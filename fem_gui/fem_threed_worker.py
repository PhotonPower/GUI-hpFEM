"""Worker for 3D structures (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_threed_worker.py <job folder> [estimate]

Reads <folder>/job.json (fem_threed.job). Tasks:
  scatter    plane wave (x-polarised, along +z) on a sphere or a cube in a box with PML, Scattering3D, scattered-field formulation; per wavelength the
             cross sections (cross_sections around the object, sphere: Mie series as reference) and |E|^2 on two cut planes. Optionally the quarter model
             x > 0, y > 0 (mirror symmetry of the wave: x = 0 PEC, y = 0 natural), as in the library benchmark (validation C).
  resonance  Resonance3D: the complex eigenfrequencies of the object (lossy or lossless) with PML; frozen permittivity at the target wavelength.
  bands      BandStructure3D: simple-cubic photonic crystal, sphere or cube in the unit cell (a = 1), lossless, path G-X-M-R-G.
With ``estimate`` only the mesh is built and hpfem.estimate_memory is printed (``ESTIMATE{json}``), mesh_<task>.npz is written. Before every run the estimate is
logged; a job whose estimate exceeds ``max_gb`` stops with ERROR.

Writes <folder>/results.json after every step and log lines on stdout (PROGRESS i/n, ERROR ..., CANCELLED, DONE). Cancellation: the file <folder>/cancel,
checked between the steps (a single solve cannot be interrupted).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

C0 = 299792458.0
TAG_OBJ = 2
TAG_SYM = 7                      # cut x = 0 of the quarter model (PEC)
GRID = 41


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------------------------------------------ meshes
def build_mesh(h, job):
    """(mesh, info). SI units for scatter / resonance (a = radius_nm), units of the lattice constant for bands."""
    task, shape = job["task"], job["shape"]
    n, hw, outer = int(job["n"]), float(job["half_width"]), float(job["outer"])
    if task == "bands":
        a = float(job["radius"])                     # radius (sphere) or half edge (cube) in units of the lattice constant
        hw = outer = 0.5
    else:
        a = float(job["radius_nm"]) * 1e-9
        hw, outer = hw * a, outer * a
    if shape == "sphere":
        if task == "bands":
            mesh = h.box_with_ball(n, a, 0.5, 0.5, TAG_OBJ)
        else:
            mesh = h.box_with_ball(n, a, hw, outer, TAG_OBJ)
    else:
        cells = int(job["cells"]) if task == "bands" else int(round(2 * (outer / a) * n))
        mesh = h.box(cells, cells, cells, np.full(3, -outer), np.full(3, outer))
        cen = np.asarray(mesh.cell_centroids)
        inside = np.all(np.abs(cen) < a, axis=1)
        for c in np.nonzero(inside)[0]:
            mesh.set_cell_tag(int(c), TAG_OBJ)
    quarter = bool(job.get("quarter", False)) and task == "scatter"
    if quarter:
        mesh = h.extract(mesh, lambda x: bool(x[0] > 0 and x[1] > 0))
        for f in mesh.boundary_facets:
            if mesh.facet_tag(int(f)) != h.NO_TAG:
                continue
            if all(abs(mesh.vertex(int(v))[0]) < 1e-9 * max(a, 1e-30) + 1e-12 * a for v in mesh.facet_vertices(int(f))):
                mesh.set_facet_tag(int(f), TAG_SYM)
    return mesh, dict(a=a, hw=hw, outer=outer, quarter=quarter)


def pml_box(h, job, info, k0):
    hw, outer = info["hw"], info["outer"]
    d = outer - hw
    prof = h.PmlProfile(int(job.get("pml_order", 3)), float(job.get("pml_reflection", 1e-8)))
    if info["quarter"]:
        return h.PmlBox3D(np.array([0.0, 0.0, -hw]), np.array([hw, hw, hw]), [0.0, d, 0.0, d, d, d], k0, 1.0, prof)
    return h.PmlBox3D.uniform(np.full(3, -hw), np.full(3, hw), d, k0, 1.0, prof)


def pec_tags(h, info):
    t = h.box_tag
    return [t.X_MAX, t.Y_MAX, t.Z_MIN, t.Z_MAX, TAG_SYM] if info["quarter"] else [t.X_MIN, t.X_MAX, t.Y_MIN, t.Y_MAX, t.Z_MIN, t.Z_MAX]


def estimate(h, job, mesh):
    nd = h.NedelecDofMap3D(mesh, int(job["order"]))
    e = h.estimate_memory(nd)
    return dict(cells=int(mesh.num_cells), dofs=int(nd.num_dofs), bytes=int(e.total_bytes), text=e.describe(), human=h.format_bytes(int(e.total_bytes)))


def save_mesh(folder, task, mesh):
    np.savez_compressed(folder / f"mesh_{task}.npz", vertices=np.asarray(mesh.vertices), cells=np.asarray(mesh.cells), tags=np.asarray(mesh.cell_tags))


# ------------------------------------------------------------------------------------------------------------------------------------ scatter
def slice_points(info):
    """Cut planes in the quarter x, y >= 0 (the fields are mirrored by the app): xz at y = 0, xy at z = 0, edges moved inwards by a tiny amount."""
    a, hw = info["a"], info["hw"]
    eps = 1e-6 * a
    xs, ys, zs = np.linspace(eps, hw - eps, GRID), np.linspace(eps, hw - eps, GRID), np.linspace(-hw + eps, hw - eps, 2 * GRID - 1)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pxz = np.stack([X.ravel(), np.full(X.size, eps), Z.ravel()], axis=1)
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    pxy = np.stack([X2.ravel(), Y2.ravel(), np.full(X2.size, eps)], axis=1)
    return (xs, zs, pxz, X.shape), (xs, ys, pxy, X2.shape)


def run_scatter(h, job, folder, mesh, info, out, flush):
    a = info["a"]
    sphere = job["shape"] == "sphere"
    lams = np.asarray(job["wavelengths_nm"], dtype=float)
    nd = h.NedelecDofMap3D(mesh, int(job["order"]))
    surface = h.Surface3D.around_cells(mesh, TAG_OBJ)
    area = np.pi * a * a if sphere else (2 * a) ** 2
    mult = 4.0 if info["quarter"] else 1.0
    locator = h.PointLocator3D(mesh)
    (xs, zs, pxz, sxz), (_, ys, pxy, sxy) = slice_points(info)
    for i, lam in enumerate(lams):
        if (folder / "cancel").exists():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return
        say(f"PROGRESS {i + 1}/{len(lams)} lambda = {lam:g} nm")
        t0 = time.time()
        pt = dict(index=i, lam_nm=float(lam), eps=job["eps"][i], size_parameter=float(2 * np.pi * a / (lam * 1e-9)))
        try:
            omega = 2 * np.pi * C0 / (lam * 1e-9)
            k0 = omega / C0
            eps = complex(job["eps"][i][0], job["eps"][i][1])
            s = h.ScatteringSetup3D()
            s.omega = omega
            s.materials.set(TAG_OBJ, h.Material(eps))
            s.incident = h.plane_wave(np.array([1.0, 0.0, 0.0], dtype=complex), np.array([0.0, 0.0, k0]))
            s.formulation = h.Formulation.SCATTERED_FIELD
            s.pml = pml_box(h, job, info, k0)
            s.pec_tags = pec_tags(h, info)
            problem = h.Scattering3D(nd, s)
            sol = problem.solve()
            cs = h.cross_sections(problem, sol, surface, 1.0)
            pt.update(sigma_sca=mult * cs.scattering, sigma_abs=mult * cs.absorption, sigma_ext=mult * cs.extinction,
                      q_sca=mult * cs.scattering / area, q_abs=mult * cs.absorption / area, q_ext=mult * cs.extinction / area)
            if sphere:
                mie = h.mie_sphere(k0, a, eps)
                pt.update(mie_q_sca=float(mie.scattering_efficiency()), mie_q_abs=float(mie.absorption_efficiency()), mie_q_ext=float(mie.extinction_efficiency()))
            if job.get("fields", True):
                fields = {}
                for key, pts, shp in (("xz", pxz, sxz), ("xy", pxy, sxy)):
                    vals, cells = problem.sample(sol, locator, pts)
                    e2 = np.sum(np.abs(vals) ** 2, axis=1)
                    e2[np.asarray(cells) < 0] = np.nan
                    fields[key] = e2.reshape(shp)
                np.savez_compressed(folder / f"field_{i}.npz", xs=xs, ys=ys, zs=zs, xz=fields["xz"], xy=fields["xy"])
                pt["field_file"] = f"field_{i}.npz"
            pt["seconds"] = time.time() - t0
        except Exception as exc:                                  # a failed wavelength must not stop the sweep
            pt["error"] = f"{type(exc).__name__}: {exc}"
            say(f"ERROR wavelength {lam:g} nm: {pt['error']}")
        out["points"].append(pt)
        flush()


# ------------------------------------------------------------------------------------------------------------------------------------ resonance
def run_resonance(h, job, folder, mesh, info, out, flush):
    a = info["a"]
    lam0 = float(job["lam0_nm"])
    omega0 = 2 * np.pi * C0 / (lam0 * 1e-9)
    k0 = omega0 / C0
    nd = h.NedelecDofMap3D(mesh, int(job["order"]))
    s = h.ResonanceSetup3D()
    s.target_omega = omega0
    s.num_modes = int(job["num_modes"])
    if int(job.get("krylov", 0)) > 0:
        s.krylov_dimension = int(job["krylov"])
    s.materials.set(TAG_OBJ, h.Material(complex(job["eps"][0], job["eps"][1])))
    s.pml = pml_box(h, job, info, k0)
    s.pec_tags = pec_tags(h, info)
    say("PROGRESS 0/1 Arnoldi (one call, cannot be interrupted)")
    t0 = time.time()
    modes = h.Resonance3D(nd, s).solve()
    locator = h.PointLocator3D(mesh)
    hw = info["hw"]
    eps = 1e-6 * a
    xs = np.linspace(-hw + eps, hw - eps, GRID)
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    pxz = np.stack([X.ravel(), np.full(X.size, eps), Z.ravel()], axis=1)
    pxy = np.stack([X.ravel(), Z.ravel(), np.full(X.size, eps)], axis=1)
    for j, m in enumerate(modes):
        om = complex(m.omega)
        rec = dict(index=j, omega=[om.real, om.imag], lam_nm=float(2 * np.pi * C0 / om.real * 1e9), Q=float(m.quality), residual=float(m.residual))
        if job.get("fields", True):
            fields = {}
            for key, pts in (("xz", pxz), ("xy", pxy)):
                e2 = np.full(len(pts), np.nan)
                for q, x in enumerate(pts):
                    v = h.evaluate_hcurl(nd, np.asarray(m.field), locator, x)
                    if v is not None:
                        e2[q] = float(np.sum(np.abs(np.asarray(v)) ** 2))
                fields[key] = e2.reshape(X.shape)
            np.savez_compressed(folder / f"field_{j}.npz", xs=xs, ys=xs, zs=xs, xz=fields["xz"], xy=fields["xy"])
            rec["field_file"] = f"field_{j}.npz"
        out["modes"].append(rec)
    out["seconds"] = time.time() - t0
    say(f"PROGRESS 1/1 {len(modes)} modes")
    flush()


# ------------------------------------------------------------------------------------------------------------------------------------ bands
def cubic_path(n_seg):
    """G-X-M-R-G of the simple cubic lattice in units of 2 pi / a: (k list, x, ticks)."""
    corners = [("Γ", (0, 0, 0)), ("X", (0.5, 0, 0)), ("M", (0.5, 0.5, 0)), ("R", (0.5, 0.5, 0.5)), ("Γ", (0, 0, 0))]
    ks, x, ticks, pos = [corners[0][1]], [0.0], [(0.0, "Γ")], 0.0
    for (_, p), (lab, q) in zip(corners[:-1], corners[1:]):
        p, q = np.array(p, float), np.array(q, float)
        for t in range(1, n_seg + 1):
            ks.append(tuple(p + (q - p) * t / n_seg))
            x.append(pos + np.linalg.norm(q - p) * t / n_seg)
        pos += float(np.linalg.norm(q - p))
        ticks.append((pos, lab))
    return ks, x, ticks


def run_bands(h, job, folder, mesh, info, out, flush):
    p = int(job["order"])
    nd, h1 = h.NedelecDofMap3D(mesh, p), h.DofMap3D(mesh, p)
    t = h.box_tag
    s = h.BandStructureSetup3D()
    s.lattice = [h.PeriodicPair3D(t.X_MIN, t.X_MAX, [1.0, 0.0, 0.0]), h.PeriodicPair3D(t.Y_MIN, t.Y_MAX, [0.0, 1.0, 0.0]),
                 h.PeriodicPair3D(t.Z_MIN, t.Z_MAX, [0.0, 0.0, 1.0])]
    s.num_bands = int(job["bands"]) + 3                        # the static zero modes at Gamma (up to three) are replaced below
    s.krylov_dimension = int(job["krylov"]) if int(job.get("krylov", 0)) > 0 else 40 + 4 * s.num_bands      # the automatic choice stalls at Gamma
    s.materials = h.MaterialMap(h.Material(complex(job["eps_bg"], 0.0)))
    s.materials.set(TAG_OBJ, h.Material(complex(job["eps"], 0.0)))
    say("assembling the Bloch problem ...")
    problem = h.BandStructure3D(nd, h1, s)
    ks, x, ticks = cubic_path(int(job["n_seg"]))
    out.update(x=x, ticks=ticks)
    nb = int(job["bands"])
    for i, k in enumerate(ks):
        if (folder / "cancel").exists():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return
        say(f"PROGRESS {i + 1}/{len(ks)} k = ({k[0]:.3f}, {k[1]:.3f}, {k[2]:.3f}) 2pi/a")
        t0 = time.time()
        pt = dict(index=i, k=list(k))
        try:
            b = problem.bands(np.array(k, float) * 2 * np.pi)
            f = np.asarray(b.normalised(1.0), dtype=float)
            if np.linalg.norm(k) == 0:                                      # static zero modes at Gamma: keep the two acoustic ones (f = 0) so the band index
                f = np.concatenate([[0.0, 0.0], f[f > 1e-3 / (2 * np.pi)]])  # stays the same along the path
            pt["f"] = [float(v) for v in f[:nb]]
            pt["residual"] = float(np.max(np.asarray(b.residual)))
            pt["seconds"] = time.time() - t0
        except Exception as exc:
            pt["error"] = f"{type(exc).__name__}: {exc}"
            say(f"ERROR k point {i}: {pt['error']}")
        out["points"].append(pt)
        flush()


# ------------------------------------------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    task = job["task"]
    try:
        mesh, info = build_mesh(hpfem, job)
        est = estimate(hpfem, job, mesh)
        save_mesh(folder, task, mesh)
    except Exception as exc:
        say(f"ERROR mesh: {type(exc).__name__}: {exc}")
        return 1
    if len(argv) > 1 and argv[1] == "estimate":
        print("ESTIMATE" + json.dumps(est), flush=True)
        return 0
    say(f"hpfem {getattr(hpfem, '__version__', '?')}, mesh: {est['cells']} cells, {est['dofs']} unknowns, memory estimate {est['human']}: {est['text']}")
    if est["bytes"] > float(job.get("max_gb", 16.0)) * 1e9:
        say(f"ERROR memory estimate {est['human']} exceeds the limit of {job.get('max_gb', 16.0)} GB: reduce n, p or the box (or raise the limit).")
        return 1
    out = dict(meta=dict(name=job.get("name", ""), task=task, shape=job["shape"], started=time.strftime("%Y-%m-%d %H:%M:%S"),
                         hpfem=str(getattr(hpfem, "__version__", "?")), quarter=info["quarter"], order=int(job["order"]), **est), points=[], modes=[])
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    flush()
    try:
        {"scatter": run_scatter, "resonance": run_resonance, "bands": run_bands}[task](hpfem, job, folder, mesh, info, out, flush)
    except Exception as exc:
        say(f"ERROR {type(exc).__name__}: {exc}")
        say(traceback.format_exc())
        return 1
    flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
