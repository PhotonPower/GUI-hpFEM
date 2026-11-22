"""Worker for 3D structures (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_threed_worker.py <job folder> [estimate]

Reads <folder>/job.json (fem_threed.job). Tasks:
  scatter    plane wave (x-polarised, along +z) on a sphere or a cube in a box with PML, Scattering3D, scattered-field formulation; per wavelength the
             cross sections (cross_sections around the object, sphere: Mie series as reference) and |E|^2 on two cut planes. Optionally the quarter model
             x > 0, y > 0 (mirror symmetry of the wave: x = 0 PEC, y = 0 natural), as in the library benchmark (validation C).
  resonance  Resonance3D: the complex eigenfrequencies of the object (lossy or lossless) with PML; frozen permittivity at the target wavelength.
  bands      BandStructure3D: simple-cubic photonic crystal, sphere or cube in the unit cell (a = 1), lossless, path G-X-M-R-G.
Extensions: ``mesh_kind`` gmsh (own geometry from fem_threed_gmsh.py: sphere, cylinder, block, optionally above a planar layer stack) or file (uploaded .msh with a tag
to material table); ``stack`` (scatter): LayerStack3D as layered background, oblique incidence s / p, R / T / A of the bare stack; ``adaptive`` (scatter):
AdaptiveMesh3D with the residual estimator, Doerfler marking and the hp decision by prediction, convergence per step in results.json.
With ``estimate`` only the mesh is built and hpfem.estimate_memory is printed (``ESTIMATE{json}``), mesh_<task>.npz is written. Before every run the estimate is
logged; a job whose estimate exceeds ``max_gb`` stops with ERROR.

With ``inspect`` the tags of the mesh file job["file"]["path"] are printed (``INSPECT{json}``).

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

C0 = 299792458.0
TAG_OBJ = 2
TAG_SYM = 7                      # cut x = 0 of the quarter model (PEC)
TAG_INSIDE = 99                  # cells inside the measurement surface (on a copy of the mesh) when the object has several tags
GRID = 41


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------------------------------------------ meshes
def build_mesh(h, job, folder=None):
    """(mesh, info). SI units for scatter / resonance (a = radius_nm), units of the lattice constant for bands. info: a (characteristic length), hw, outer, quarter,
    object_tags, interior (x_lo, x_hi, z_lo, z_hi) of the PML box, area (reference cross-section), zc (height of the object centre), sphere (radius for Mie)."""
    kind = job.get("mesh_kind", "builtin")
    if kind == "gmsh" and job["task"] != "bands":
        return build_gmsh(h, job, folder)
    if kind == "file" and job["task"] != "bands":
        return build_file(h, job)
    return build_builtin(h, job)


def build_builtin(h, job):
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
    area = np.pi * a * a if shape == "sphere" else (2 * a) ** 2
    return mesh, dict(a=a, hw=hw, outer=outer, quarter=quarter, object_tags=[TAG_OBJ], interior=[-hw, hw, -hw, hw], area=area, zc=0.0, kind="builtin",
                      sphere=a if shape == "sphere" else None)


def build_gmsh(h, job, folder):
    """Own geometry with Gmsh (fem_threed_gmsh.scene), read with hpfem.read_gmsh; the mesh file is reused when the geometry is unchanged."""
    import fem_threed_gmsh as gm

    g = dict(job["g"])
    folder = Path(folder or ".")
    path = folder / "scene.msh"
    key = json.dumps(g, sort_keys=True)
    keyfile = folder / "scene.key"
    if not (path.is_file() and keyfile.is_file() and keyfile.read_text(encoding="utf-8") == key):
        gi = gm.scene(g, path)
        keyfile.write_text(key, encoding="utf-8")
    mesh = h.read_gmsh(str(path), 1e-9, 3)
    lay = gm.layout(g)
    hw = g["hw"] * 1e-9
    ox, oy, oh = gm.object_extent(g)
    area = {"sphere": np.pi * g["r"] ** 2, "cylinder": np.pi * g["r"] ** 2, "block": g["lx"] * g["ly"]}[g["kind"]] * 1e-18
    return mesh, dict(a=g["r"] * 1e-9 if "r" in g else g["lx"] * 0.5e-9, hw=hw, outer=hw + g["pml"] * 1e-9, quarter=False, object_tags=[gm.TAG_OBJ],
                      interior=[-hw, hw, lay["z_lo"] * 1e-9, lay["z_hi"] * 1e-9], area=area, zc=lay["zc"] * 1e-9, kind="gmsh",
                      sphere=g["r"] * 1e-9 if (g["kind"] == "sphere" and not g.get("stack")) else None, layout=lay)


def build_file(h, job):
    """Uploaded mesh: scale to metres, the outer boundary facets without a tag get the box tags X_MIN ... Z_MAX (the PML box is the bounding box shrunk by the PML thickness)."""
    f = job["file"]
    mesh = h.read_gmsh(str(f["path"]), float(f["scale"]), 3)
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    tol = 1e-6 * float(np.max(hi - lo))
    t = h.box_tag
    for fc in mesh.boundary_facets:
        if mesh.facet_tag(int(fc)) != h.NO_TAG:
            continue
        pts = np.array([mesh.vertex(int(i)) for i in mesh.facet_vertices(int(fc))])
        for axis, (tmin, tmax) in enumerate(((t.X_MIN, t.X_MAX), (t.Y_MIN, t.Y_MAX), (t.Z_MIN, t.Z_MAX))):
            if np.all(np.abs(pts[:, axis] - lo[axis]) < tol):
                mesh.set_facet_tag(int(fc), tmin)
                break
            if np.all(np.abs(pts[:, axis] - hi[axis]) < tol):
                mesh.set_facet_tag(int(fc), tmax)
                break
    pml = float(f["pml_nm"]) * 1e-9
    tags = [int(x) for x in f["object_tags"]]
    cen = np.asarray(mesh.cell_centroids)
    ctags = np.asarray(mesh.cell_tags)
    sel = np.isin(ctags, tags)
    zc = float(cen[sel, 2].mean()) if sel.any() else 0.0
    ext = np.zeros(3)
    area = 1.0
    if sel.any():
        pts = np.array([mesh.vertex(int(i)) for c in np.nonzero(sel)[0] for i in mesh.cell_vertices(int(c))])
        ext = pts.max(axis=0) - pts.min(axis=0)
        area = projected_area_xy(h, mesh, tags) or float(ext[0] * ext[1])
    return mesh, dict(a=float(max(ext) / 2) if sel.any() else 1.0, hw=float(hi[0] - pml), outer=float(hi[0]), quarter=False, object_tags=tags,
                      interior=[float(lo[0] + pml), float(hi[0] - pml), float(lo[2] + pml), float(hi[2] - pml)], interior_y=[float(lo[1] + pml), float(hi[1] - pml)],
                      pml=pml, area=area, zc=zc, kind="file", sphere=None)


def pml_box(h, job, info, k0):
    prof = h.PmlProfile(int(job.get("pml_order", 3)), float(job.get("pml_reflection", 1e-8)))
    if info["kind"] == "gmsh":
        x0, x1, z0, z1 = info["interior"]
        pml = float(job["g"]["pml"]) * 1e-9
        return h.PmlBox3D(np.array([x0, x0, z0]), np.array([x1, x1, z1]), [pml] * 6, k0, 1.0, prof)
    if info["kind"] == "file":
        x0, x1, z0, z1 = info["interior"]
        y0, y1 = info["interior_y"]
        pml = info["pml"]
        return h.PmlBox3D(np.array([x0, y0, z0]), np.array([x1, y1, z1]), [pml] * 6, k0, 1.0, prof)
    hw, outer = info["hw"], info["outer"]
    d = outer - hw
    if info["quarter"]:
        return h.PmlBox3D(np.array([0.0, 0.0, -hw]), np.array([hw, hw, hw]), [0.0, d, 0.0, d, d, d], k0, 1.0, prof)
    return h.PmlBox3D.uniform(np.full(3, -hw), np.full(3, hw), d, k0, 1.0, prof)


def pec_tags(h, info):
    t = h.box_tag
    if info["quarter"]:
        return [t.X_MAX, t.Y_MAX, t.Z_MIN, t.Z_MAX, TAG_SYM]
    return [t.X_MIN, t.X_MAX, t.Y_MIN, t.Y_MAX, t.Z_MIN, t.Z_MAX]


def surface_of(h, mesh, tags):
    """Closed surface around the cells with one of `tags` (several tags: retagged on a copy of the mesh, the facet numbers are those of the mesh)."""
    if len(tags) == 1:
        return h.Surface3D.around_cells(mesh, int(tags[0]))
    cp = mesh.copy()
    ts = set(int(x) for x in tags)
    for c in range(cp.num_cells):
        if int(cp.cell_tag(c)) in ts:
            cp.set_cell_tag(c, TAG_INSIDE)
    return h.Surface3D.around_cells(cp, TAG_INSIDE)


def projected_area_xy(h, mesh, tags):
    """Area of the projection of the object (cells with `tags`) onto the x-y plane: half the sum of |n_z| dA over the closed surface (straight facets)."""
    surf = surface_of(h, mesh, tags)
    tot = 0.0
    for f in surf.facets:
        v = np.array([mesh.vertex(int(i)) for i in mesh.facet_vertices(int(f.facet))[:3]])
        tot += abs(0.5 * np.cross(v[1] - v[0], v[2] - v[0])[2])
    return 0.5 * tot


def set_materials(h, setup_or_map, job, info, i):
    """Materials by tag for a scatter / resonance setup: the object, the regions of the layer stack (gmsh) or the tag table (file)."""
    mm = setup_or_map
    if info["kind"] == "file":
        for tag, lst in job["file"].get("tag_eps", {}).items():
            e = lst[i]
            mm.set(int(tag), h.Material(complex(e[0], e[1])))
        return
    e = job["eps"][i] if job["task"] == "scatter" else job["eps"]
    mm.set(TAG_OBJ, h.Material(complex(e[0], e[1])))
    st = job.get("stack")
    if st and info["kind"] == "gmsh":
        for r, lst in enumerate(st["eps"]):
            ee = lst[i]
            mm.set(11 + r, h.Material(complex(ee[0], ee[1])))


def estimate(h, job, mesh):
    nd = h.NedelecDofMap3D(mesh, int(job["order"]))
    e = h.estimate_memory(nd)
    return dict(cells=int(mesh.num_cells), dofs=int(nd.num_dofs), bytes=int(e.total_bytes), text=e.describe(), human=h.format_bytes(int(e.total_bytes)))


def inspect_mesh(h, path, scale):
    """Tags of a mesh file: cells per tag, facets per tag, bounding box [m], geometry order."""
    mesh = h.read_gmsh(str(path), float(scale), 3)
    cells = np.asarray(mesh.cell_tags)
    ftags = np.asarray(mesh.facet_tags)
    v = np.asarray(mesh.vertices)
    return dict(n_cells=int(mesh.num_cells), cells={str(int(x)): int((cells == x).sum()) for x in np.unique(cells)},
                facets={str(int(x)): int((ftags == x).sum()) for x in np.unique(ftags) if int(x) != int(h.NO_TAG)},
                bbox=[v.min(axis=0).tolist(), v.max(axis=0).tolist()], geometry_order=int(mesh.geometry_order))


def save_mesh(folder, task, mesh, orders=None):
    extra = {} if orders is None else dict(orders=np.asarray(orders, dtype=int))
    np.savez_compressed(folder / f"mesh_{task}.npz", vertices=np.asarray(mesh.vertices), cells=np.asarray(mesh.cells), tags=np.asarray(mesh.cell_tags), **extra)


# ------------------------------------------------------------------------------------------------------------------------------------ scatter
def slice_points(info):
    """Cut planes: xz at y = 0, xy through the centre of the object. Quarter model: x, y >= 0 (the app mirrors the fields); otherwise the whole interior of the PML box.
    Returns ((xs, zs, pxz, shape), (xs, ys, pxy, shape)); edges are moved inwards by a tiny amount."""
    a, hw = info["a"], info["hw"]
    eps = 1e-6 * a
    if info["quarter"]:
        xs, ys, zs = np.linspace(eps, hw - eps, GRID), np.linspace(eps, hw - eps, GRID), np.linspace(-hw + eps, hw - eps, 2 * GRID - 1)
        y0, z0 = eps, eps
    else:
        x0, x1, zlo, zhi = info["interior"]
        xs = np.linspace(x0 + eps, x1 - eps, GRID)
        yr = info.get("interior_y", [x0, x1])
        ys = np.linspace(yr[0] + eps, yr[1] - eps, GRID)
        zs = np.linspace(zlo + eps, zhi - eps, GRID if info["kind"] == "builtin" else 2 * GRID - 1)
        y0, z0 = 0.5 * (yr[0] + yr[1]) + eps, info.get("zc", 0.0) + eps
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pxz = np.stack([X.ravel(), np.full(X.size, y0), Z.ravel()], axis=1)
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    pxy = np.stack([X2.ravel(), Y2.ravel(), np.full(X2.size, z0)], axis=1)
    return (xs, zs, pxz, X.shape), (xs, ys, pxy, X2.shape)


def scatter_setup(h, job, info, i):
    """(setup, omega, k0, eps_obj, extra) of wavelength i. With a layer stack the background is a LayerStack3D and the incident field its plane wave (oblique,
    s / p); extra then carries R, T, A of the bare stack."""
    lam = float(job["wavelengths_nm"][i])
    omega = 2 * np.pi * C0 / (lam * 1e-9)
    k0 = omega / C0
    eps = complex(*(job["eps"][i] if info["kind"] != "file" else (1.0, 0.0)))
    s = h.ScatteringSetup3D()
    s.omega = omega
    set_materials(h, s.materials, job, info, i)
    s.formulation = h.Formulation.SCATTERED_FIELD
    s.pml = pml_box(h, job, info, k0)
    s.pec_tags = pec_tags(h, info)
    extra = {}
    st = job.get("stack")
    if st and info["kind"] == "gmsh":
        mats = [h.Material(complex(e[i][0], e[i][1])) for e in st["eps"]]
        layers = [h.Layer(mats[1 + k], float(th) * 1e-9) for k, th in enumerate(st["thickness_nm"])]
        stack = h.LayerStack3D(mats[0], layers, mats[-1], 0.0)
        pol = h.Polarisation.S if st["pol"] == "s" else h.Polarisation.P
        wave = stack.plane_wave(k0, np.deg2rad(float(st["angle_deg"])), pol, 1.0, np.deg2rad(float(st["azimuth_deg"])))
        s.background = stack
        s.incident = wave.field
        extra = dict(stack_R=float(wave.reflectance), stack_T=float(wave.transmittance), stack_A=float(wave.absorptance))
    else:
        s.incident = h.plane_wave(np.array([1.0, 0.0, 0.0], dtype=complex), np.array([0.0, 0.0, k0]))
    return s, omega, k0, eps, extra


def scatter_post(h, job, info, mesh, problem, sol, pt, k0, eps, locator, planes, folder, i):
    """Cross sections (and Mie), field cuts of one solution into the point record pt."""
    mult = 4.0 if info["quarter"] else 1.0
    surface = surface_of(h, mesh, info["object_tags"])
    area = info["area"]
    cs = h.cross_sections(problem, sol, surface, 1.0)
    pt.update(sigma_sca=mult * cs.scattering, sigma_abs=mult * cs.absorption, sigma_ext=mult * cs.extinction,
              q_sca=mult * cs.scattering / area, q_abs=mult * cs.absorption / area, q_ext=mult * cs.extinction / area)
    if info.get("sphere"):
        mie = h.mie_sphere(k0, info["sphere"], eps)
        pt.update(mie_q_sca=float(mie.scattering_efficiency()), mie_q_abs=float(mie.absorption_efficiency()), mie_q_ext=float(mie.extinction_efficiency()))
    if job.get("fields", True) and locator is not None:
        (xs, zs, pxz, sxz), (_, ys, pxy, sxy) = planes
        fields = {}
        for key, pts, shp in (("xz", pxz, sxz), ("xy", pxy, sxy)):
            vals, cells = problem.sample(sol, locator, pts)
            e2 = np.sum(np.abs(vals) ** 2, axis=1)
            e2[np.asarray(cells) < 0] = np.nan
            fields[key] = e2.reshape(shp)
        np.savez_compressed(folder / f"field_{i}.npz", xs=xs, ys=ys, zs=zs, xz=fields["xz"], xy=fields["xy"])
        pt["field_file"] = f"field_{i}.npz"


def cancelled(folder, out, flush):
    if (folder / "cancel").exists():
        say("CANCELLED")
        out["cancelled"] = True
        flush()
        return True
    return False


def run_scatter(h, job, folder, mesh, info, out, flush):
    a = info["a"]
    lams = np.asarray(job["wavelengths_nm"], dtype=float)
    ad = job.get("adaptive") or {}
    if ad.get("enabled"):
        return run_scatter_adaptive(h, job, folder, mesh, info, out, flush)
    nd = h.NedelecDofMap3D(mesh, int(job["order"]))
    locator = h.PointLocator3D(mesh) if job.get("fields", True) else None
    planes = slice_points(info) if locator is not None else None
    for i, lam in enumerate(lams):
        if cancelled(folder, out, flush):
            return
        say(f"PROGRESS {i + 1}/{len(lams)} lambda = {lam:g} nm")
        t0 = time.time()
        pt = dict(index=i, lam_nm=float(lam), eps=job["eps"][i], size_parameter=float(2 * np.pi * a / (lam * 1e-9)))
        try:
            s, omega, k0, eps, extra = scatter_setup(h, job, info, i)
            pt.update(extra)
            problem = h.Scattering3D(nd, s)
            sol = problem.solve()
            scatter_post(h, job, info, mesh, problem, sol, pt, k0, eps, locator, planes, folder, i)
            pt["seconds"] = time.time() - t0
        except Exception as exc:                                  # a failed wavelength must not stop the sweep
            pt["error"] = f"{type(exc).__name__}: {exc}"
            say(f"ERROR wavelength {lam:g} nm: {pt['error']}")
        out["points"].append(pt)
        flush()


def run_scatter_adaptive(h, job, folder, mesh, info, out, flush):
    """hp-adaptive loop per wavelength: solve, estimate (residual), Doerfler marking, hp decision by prediction, refine; stops at max_steps, max_dofs, the memory
    limit or the relative estimator tolerance. The steps are recorded in the point (pt["steps"]); cross sections and fields are those of the last mesh."""
    a = info["a"]
    ad = job["adaptive"]
    lams = np.asarray(job["wavelengths_nm"], dtype=float)
    p0 = int(job["order"])
    max_steps, max_dofs, theta, tol = int(ad["max_steps"]), int(ad["max_dofs"]), float(ad["theta"]), float(ad.get("tol", 0.0))
    max_bytes = float(job.get("max_gb", 16.0)) * 1e9
    for i, lam in enumerate(lams):
        if cancelled(folder, out, flush):
            return
        t0 = time.time()
        pt = dict(index=i, lam_nm=float(lam), eps=job["eps"][i], size_parameter=float(2 * np.pi * a / (lam * 1e-9)), steps=[])
        out["points"].append(pt)
        try:
            adaptive = h.AdaptiveMesh3D(mesh)
            orders = np.full(mesh.num_cells, p0, dtype=int)
            predicted = []
            last = None
            eta0 = None
            for step in range(max_steps):
                if cancelled(folder, out, flush):
                    return
                m = adaptive.mesh
                nd = h.NedelecDofMap3D(m, orders.tolist())
                mem = h.estimate_memory(nd)
                if step > 0 and (nd.num_dofs > max_dofs or mem.total_bytes > max_bytes):
                    say(f"stop: the next mesh has {nd.num_dofs} unknowns / {h.format_bytes(int(mem.total_bytes))} (limit {max_dofs} unknowns, {max_bytes / 1e9:g} GB)")
                    pt["stop_reason"] = "Grenze für Unbekannte oder Speicher erreicht"
                    break
                say(f"PROGRESS {i * max_steps + step + 1}/{len(lams) * max_steps} lambda = {lam:g} nm, step {step}: {m.num_cells} cells, {nd.num_dofs} unknowns, "
                    f"{h.format_bytes(int(mem.total_bytes))}")
                s, omega, k0, eps, extra = scatter_setup(h, job, info, i)
                pt.update(extra)
                problem = h.Scattering3D(nd, s)
                ts = time.time()
                sol = problem.solve()
                est = problem.estimate(sol)
                eta = float(est.total())
                eta0 = eta if eta0 is None else eta0
                rec = dict(step=step, cells=int(m.num_cells), dofs=int(nd.num_dofs), eta=eta, orders_max=int(orders.max()), orders_min=int(orders.min()),
                           seconds=time.time() - ts)
                surf = surface_of(h, m, info["object_tags"])
                cs = h.cross_sections(problem, sol, surf, 1.0)
                mult = 4.0 if info["quarter"] else 1.0
                rec.update(q_sca=mult * cs.scattering / info["area"], q_abs=mult * cs.absorption / info["area"])
                if info.get("sphere"):
                    mie = h.mie_sphere(k0, info["sphere"], eps)
                    rec.update(mie_q_sca=float(mie.scattering_efficiency()))
                pt["steps"].append(rec)
                last = (m, nd, problem, sol, k0, eps, orders.copy())
                flush()
                if tol > 0 and eta / eta0 < tol:
                    pt["stop_reason"] = "Fehlerschätzer unter der Toleranz"
                    break
                if step == max_steps - 1:
                    pt["stop_reason"] = "Schrittzahl erreicht"
                    break
                marked = h.dorfler_marking(list(est.indicators), theta)
                dec = h.hp_decide_by_prediction(list(est.indicators), list(predicted), [int(c) for c in marked])
                hp = h.hp_refine(adaptive, orders.tolist(), list(dec.h_marked), list(dec.p_marked))
                predicted = list(h.predict_indicators(list(est.indicators), orders.tolist(), hp))
                rec["h_marked"], rec["p_marked"] = len(dec.h_marked), len(dec.p_marked)
                orders = np.asarray(hp.orders)
            if last is None:
                raise RuntimeError("no step was solved")
            m, nd, problem, sol, k0, eps, ords = last
            locator = h.PointLocator3D(m) if job.get("fields", True) else None
            planes = slice_points(info) if locator is not None else None
            scatter_post(h, job, info, m, problem, sol, pt, k0, eps, locator, planes, folder, i)
            save_mesh(folder, "scatter", m, ords)
            pt["dofs_final"] = int(nd.num_dofs)
            pt["seconds"] = time.time() - t0
        except Exception as exc:
            pt["error"] = f"{type(exc).__name__}: {exc}"
            say(f"ERROR wavelength {lam:g} nm: {pt['error']}")
            say(traceback.format_exc())
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
    set_materials(h, s.materials, job, info, 0)
    s.pml = pml_box(h, job, info, k0)
    s.pec_tags = pec_tags(h, info)
    say("PROGRESS 0/1 Arnoldi (one call, cannot be interrupted)")
    t0 = time.time()
    modes = h.Resonance3D(nd, s).solve()
    locator = h.PointLocator3D(mesh)
    (xs, zs, pxz, sxz), (_, ys, pxy, sxy) = slice_points(info)
    for j, m in enumerate(modes):
        om = complex(m.omega)
        rec = dict(index=j, omega=[om.real, om.imag], lam_nm=float(2 * np.pi * C0 / om.real * 1e9), Q=float(m.quality), residual=float(m.residual))
        if job.get("fields", True):
            fields = {}
            for key, pts, shp in (("xz", pxz, sxz), ("xy", pxy, sxy)):
                e2 = np.full(len(pts), np.nan)
                for q, x in enumerate(pts):
                    v = h.evaluate_hcurl(nd, np.asarray(m.field), locator, x)
                    if v is not None:
                        e2[q] = float(np.sum(np.abs(np.asarray(v)) ** 2))
                fields[key] = e2.reshape(shp)
            np.savez_compressed(folder / f"field_{j}.npz", xs=xs, ys=ys, zs=zs, xz=fields["xz"], xy=fields["xy"])
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
    if len(argv) > 1 and argv[1] == "inspect":
        try:
            print("INSPECT" + json.dumps(inspect_mesh(hpfem, job["file"]["path"], job["file"]["scale"])), flush=True)
            return 0
        except Exception as exc:
            say(f"ERROR inspect: {type(exc).__name__}: {exc}")
            return 1
    try:
        mesh, info = build_mesh(hpfem, job, folder)
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
    out = dict(meta=dict(name=job.get("name", ""), task=task, shape=job["shape"], mesh_kind=job.get("mesh_kind", "builtin"), started=time.strftime("%Y-%m-%d %H:%M:%S"),
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
