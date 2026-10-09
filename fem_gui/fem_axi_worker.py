"""Solver worker for bodies of revolution (resonators and emitters): runs in the Python that has hpfem, called by the app.

    python fem_axi_worker.py <job folder>

Reads <job>/job.json (model of fem_axi, task, solver settings) and <job>/mesh.msh (meridian mesh x = r, y = z of fem_axi.build_mesh) and
  task "resonance": the quasi-normal modes of azimuthal order m nearest the target wavelength (hpfem.AxisymmetricResonance, cylindrical
                    PML): omega, wavelength, Q, residual; the mode fields on the subdivided mesh (mode_<k>.npz)
  task "emitter":   a Gaussian-smeared point dipole on the axis (hpfem.axisymmetric_gaussian_dipole; transverse: orders m = +-1, axial: m = 0)
                    over a wavelength sweep (hpfem.AxisymmetricScattering with the current as source): Purcell factor F_P = P / P_bulk
                    (power through the closed surface around the emitter box over the Larmor power of the smeared dipole in the bulk medium
                    of the emitter), the fractions through the plane above (beta) and below the structure; optionally the sweep is centred
                    on the resonance found first (+- k linewidths); the fields at the chosen points (field_<i>.npz).
writes <job>/results.json after every point.

Progress lines: PROGRESS i/n ...  PHASE name  WARN ...  ERROR ...  CANCELLED  DONE.  Cancellation: <job>/cancel, between the points.
Field export: hpfem.FieldExporter2D (exact on the subdivided cells) parsed from its ASCII VTU; E = (E_r, E_phi, E_z) with E_phi = i v / r from
the scaled azimuthal unknown v = -i r E_phi.
"""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_axi as fa  # noqa: E402
import fem_materials as fm  # noqa: E402

NM = 1e-9


def say(*a):
    print(*a, flush=True)


class Job:
    def __init__(self, hpfem, job, folder):
        self.h, self.job, self.folder = hpfem, job, Path(folder)
        self.model = job["model"]
        self.names = list(self.model["materials"])
        self.solver = job["solver"]
        self.lay = fa.layout(self.model)
        t0 = time.time()
        self.mesh = hpfem.read_gmsh(str(self.folder / "mesh.msh"), NM, 2)
        self.tags = sorted({int(self.mesh.cell_tag(c)) for c in range(self.mesh.num_cells)})
        say(f"mesh read in {time.time() - t0:.1f} s: {self.mesh.num_cells} cells, tags {self.tags}")
        p = int(self.solver.get("order", 3))
        self.order = p
        self.nd = hpfem.NedelecDofMap2D(self.mesh, p)
        self.h1 = hpfem.DofMap2D(self.mesh, p)
        self.dofs = int(self.nd.num_dofs + self.h1.num_dofs)

    def cancelled(self):
        return (self.folder / "cancel").exists()

    def materials(self, lam_nm):
        """MaterialMap at lam_nm for every tag of the mesh (the emitter box carries 101 + index with the material of 1 + index)."""
        mm = self.h.MaterialMap()
        eps = {}
        for t in self.tags:
            name = self.names[(t - 1) % fa.TAG_SOURCE]
            e = fm.eps_at(self.model["materials"][name], lam_nm)
            eps[name] = e
            mm.set(int(t), self.h.Material(complex(e)))
        return mm, eps

    def pml(self, k0):
        h, lay = self.h, self.lay
        n_ref = float(np.sqrt(fm.eps_at(self.model["materials"][self.model["background"]], 2 * np.pi / k0 / NM) + 0j).real)
        profile = h.PmlProfile(2, float(self.solver.get("pml_reflection", 1e-10)))
        return h.PmlBox2D([0.0, lay["z_lo"] * NM], [lay["r_in"] * NM, lay["z_hi"] * NM],
                          [0.0, lay["pml"] * NM, lay["pml"] * NM, lay["pml"] * NM], k0, max(n_ref, 1.0), profile)

    def backend(self, setup):
        name = self.solver.get("backend", "AUTO")
        if name and name != "AUTO":
            try:
                setup.solver = getattr(self.h.DirectSolverBackend, name)
            except AttributeError:
                pass


# -------------------------------------------------------------------------------------------------------------- fields
def export_field(job, meridian, azimuthal, subdivisions):
    """(points [nm], simplices, E (n, 3) = (E_r, E_phi, E_z), tag per simplex) from hpfem.FieldExporter2D on the subdivided mesh, only the
    cells inside the PML box (the PML cells carry the stretched field)."""
    h = job.h
    ex = h.FieldExporter2D(job.mesh, int(subdivisions), h.VtkFormat.ASCII)
    ex.hcurl("E", job.nd, np.asarray(meridian))
    ex.h1("v", job.h1, np.asarray(azimuthal))
    root = ET.fromstring(ex.to_string())
    arrays = {da.attrib.get("Name"): da for da in root.iter("DataArray")}

    def arr(name, dtype=float):
        return np.array((arrays[name].text or "").split(), dtype=dtype)

    pts = arr("Points").reshape(-1, 3)[:, :2]
    conn = arr("connectivity", np.int64).reshape(-1, 3)
    tags = arr("cell_tag", np.int64) if "cell_tag" in arrays else np.zeros(len(conn), np.int64)
    E = (arr("E_re") + 1j * arr("E_im")).reshape(-1, 3)
    v = arr("v_re") + 1j * arr("v_im")
    r = pts[:, 0]
    tiny = 1e-6 * max(r.max(), 1e-30)
    Ephi = np.where(r > tiny, 1j * v / np.where(r > tiny, r, 1.0), np.nan)
    on_axis = np.where(~(r > tiny))[0]
    if len(on_axis):                                                 # the limit r -> 0: the value of the nearest point off the axis
        off = np.where(r > tiny)[0]
        for k in on_axis:
            j = off[np.argmin(np.abs(pts[off, 1] - pts[k, 1]) + np.abs(r[off]))]
            Ephi[k] = Ephi[j]
    field = np.column_stack([E[:, 0], Ephi, E[:, 1]])
    lay = job.lay
    tol = 1e-9 * lay["r_out"] * NM
    P = pts[conn]
    keep = (P[:, :, 0] <= lay["r_in"] * NM + tol).all(axis=1) & (P[:, :, 1] >= lay["z_lo"] * NM - tol).all(axis=1) & \
           (P[:, :, 1] <= lay["z_hi"] * NM + tol).all(axis=1)
    conn, tags = conn[keep], tags[keep]
    used = np.unique(conn)
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    field = field[used]
    scale = np.nanmax(np.abs(field)) or 1.0
    return dict(points_nm=pts[used] / NM, simplices=remap[conn].astype(np.int32), E=(field / scale).astype(np.complex64), tag=tags.astype(np.int32),
                scale=float(scale))


# ------------------------------------------------------------------------------------------------------------ resonances
def resonances(job, lam_nm, m, num_modes, krylov=0):
    """Modes of order m nearest lam_nm: list of dicts and the raw modes."""
    h = job.h
    k0 = 2 * np.pi / (lam_nm * NM)
    mm, _ = job.materials(lam_nm)
    setup = h.AxisymmetricResonanceSetup()
    setup.target_omega = k0 * h.constants.c0
    setup.materials = mm
    setup.axis_tag = fa.TAG_AXIS
    setup.pec_tags = [fa.TAG_WALL]
    setup.azimuthal_order = int(m)
    setup.pml = job.pml(k0)
    setup.num_modes = int(num_modes)
    setup.krylov_dimension = int(krylov) if krylov else max(40, 4 * int(num_modes) + 20)
    job.backend(setup)
    say("PHASE eigensolve")
    t0 = time.time()
    raw = h.AxisymmetricResonance(job.nd, job.h1, setup).solve()
    dt = time.time() - t0
    modes = []
    for k, md in enumerate(raw):
        w = complex(md.omega)
        modes.append(dict(k=k, lam_nm=float(md.wavelength) / NM, Q=float(md.quality), omega=[w.real, w.imag], residual=float(md.residual), m=int(m)))
    return modes, raw, dt


def pick_fundamental(modes, lam_nm, window=0.03):
    """The highest-Q mode within +-window of lam_nm (as the micropillar example of hp-FEM), else the nearest with Q > 0."""
    good = [md for md in modes if md["Q"] > 0]
    near = [md for md in good if abs(md["lam_nm"] - lam_nm) < window * lam_nm]
    if near:
        return max(near, key=lambda md: md["Q"])
    return min(good, key=lambda md: abs(md["lam_nm"] - lam_nm)) if good else None


# --------------------------------------------------------------------------------------------------------------- emitter
def plane_surface(job, z_nm, direction):
    """Horizontal plane z = z_nm for r < r_in with the normal along +z (direction +1, cell below) or -z (cell above)."""
    h, mesh = job.h, job.mesh
    V = np.asarray(mesh.vertices)
    zc = z_nm * NM
    r_max = job.lay["r_in"] * NM
    tol = 1e-7 * max(r_max, abs(zc), 1e-9)
    centroids = np.asarray(mesh.cell_centroids)
    facets = []
    for f in range(mesh.num_facets):
        v = V[mesh.facet_vertices(f)]
        if np.all(np.abs(v[:, 1] - zc) < tol) and np.all(v[:, 0] < r_max + tol):
            cells = [c for c in mesh.facet_cells(f) if c >= 0]
            if not cells:
                continue
            pick = min(cells, key=lambda c: centroids[c][1]) if direction > 0 else max(cells, key=lambda c: centroids[c][1])
            facets.append(h.Surface2D.Facet(f, pick))
    s = h.Surface2D()
    s.facets = facets
    return s


def emitter_point(job, lam_nm, surfaces, em, want_field=False):
    """One wavelength: emitted power, Purcell factor, fractions through the planes (and the field)."""
    h = job.h
    omega = 2 * np.pi * h.constants.c0 / (lam_nm * NM)
    k0 = 2 * np.pi / (lam_nm * NM)
    mm, eps = job.materials(lam_nm)
    axial = em["orientation"] == "axial"
    m = 0 if axial else 1
    sigma = float(em["sigma_nm"]) * NM
    setup = h.AxisymmetricScatteringSetup()
    setup.omega = omega
    setup.materials = mm
    setup.axis_tag = fa.TAG_AXIS
    setup.pec_tags = [fa.TAG_WALL]
    setup.azimuthal_order = m
    setup.pml = job.pml(k0)
    setup.current = h.axisymmetric_gaussian_dipole(float(em["z_nm"]) * NM, 1.0, h.AxisDipole.AXIAL if axial else h.AxisDipole.TRANSVERSE,
                                                   sigma, omega, m)
    setup.extra_quadrature_order = int(job.solver.get("extra_quadrature_order", 6))
    job.backend(setup)
    t0 = time.time()
    field = h.AxisymmetricScattering(job.nd, job.h1, setup).solve()
    t_solve = time.time() - t0
    factor = 1.0 if axial else 2.0                                    # orders +1 and -1 radiate equally

    def flux(surface):
        return factor * float(h.axisymmetric_poynting_flux(job.nd, job.h1, field.meridian, field.azimuthal, m, omega, mm, surface))

    total = flux(surfaces["around"])
    top = flux(surfaces["top"]) if surfaces.get("top") is not None and len(surfaces["top"]) else 0.0
    bottom = flux(surfaces["bottom"]) if surfaces.get("bottom") is not None and len(surfaces["bottom"]) else 0.0
    n_em = float(np.sqrt(eps[surfaces["emitter_material"]] + 0j).real)
    bulk = n_em * float(h.dipole_vacuum_power(1.0, omega)) * np.exp(-((n_em * k0 * sigma) ** 2))
    res = dict(lam_nm=lam_nm, P_total=total, P_top=top, P_bottom=bottom, P_bulk=bulk, purcell=total / bulk,
               purcell_radiative=(top + bottom) / bulk, beta_top=top / total if total else None, beta_bottom=bottom / total if total else None,
               n_emitter=n_em, dofs=job.dofs, time_s=t_solve)
    fld = export_field(job, field.meridian, field.azimuthal, job.job.get("subdivisions", 2)) if want_field else None
    if fld is not None:
        fld.update(lam_nm=lam_nm, m=m, kind="emitter", purcell=res["purcell"])
    return res, fld


def emitter_surfaces(job):
    h = job.h
    em = job.model["emitter"]
    src_tags = [t for t in job.tags if t > fa.TAG_SOURCE]
    if len(src_tags) != 1:
        raise RuntimeError(f"Emitterkasten: erwartet genau ein Material, gefunden Tags {src_tags} (Netz mit der Aufgabe „Emitter“ neu erzeugen)")
    around = h.Surface2D.around_cells(job.mesh, int(src_tags[0]))
    lay = job.lay
    top = plane_surface(job, lay["z_plane_top"], +1)
    bottom = plane_surface(job, lay["z_plane_bottom"], -1)
    say(f"surfaces: around the emitter {len(around)} facets, top plane {len(top)}, bottom plane {len(bottom)}")
    if not len(top):
        say("WARN keine Facetten auf der oberen Messebene: β nicht messbar")
    return dict(around=around, top=top, bottom=bottom, emitter_material=job.names[(src_tags[0] - 1) % fa.TAG_SOURCE], z=float(em["z_nm"]))


# ------------------------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    jd = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    try:
        from hpfem.run import version_info
        vi = version_info()
    except Exception:
        vi = {"hpfem": str(getattr(hpfem, "__version__", "?"))}
    say(f"hpfem {vi.get('hpfem', '?')}" + (f", backends {', '.join(vi.get('backends', []))}" if vi.get("backends") else ""))
    if not hasattr(hpfem, "AxisymmetricResonance"):
        say("ERROR Dieses hpfem hat keinen zylindersymmetrischen Löser (AxisymmetricResonance, hp-FEM ab 0.3)")
        return 1
    job = Job(hpfem, jd, folder)
    model, task = job.model, jd["task"]
    say(f"order p = {job.order}: {job.dofs} DoFs")
    out = dict(meta=dict(model=model.get("name", ""), task=task, order=job.order, cells=int(job.mesh.num_cells), dofs=job.dofs,
                         started=time.strftime("%Y-%m-%d %H:%M:%S"), version_info=vi), resonance=None, points=[])
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    sub = int(jd.get("subdivisions", 2))
    try:
        if task == "resonance":
            rs = model["resonance"]
            say(f"PROGRESS 1/1 modes of order m = {rs['m']} near {rs['wavelength_nm']:g} nm")
            modes, raw, dt = resonances(job, float(rs["wavelength_nm"]), int(rs["m"]), int(rs["num_modes"]), int(jd.get("krylov_dimension", 0)))
            out["resonance"] = dict(target_nm=float(rs["wavelength_nm"]), m=int(rs["m"]), modes=modes, time_s=dt)
            for md in modes:
                say(f"  mode {md['k']}: lambda = {md['lam_nm']:.3f} nm, Q = {md['Q']:.4g}, residual {md['residual']:.1e}")
            flush()
            if jd.get("fields", True):
                say("PHASE post")
                for md, r in zip(modes, raw):
                    fld = export_field(job, r.meridian, r.azimuthal, sub)
                    fld.update(lam_nm=md["lam_nm"], Q=md["Q"], m=md["m"], kind="mode")
                    np.savez_compressed(folder / f"mode_{md['k']}.npz", **fld)
            say(f"  {dt:.1f} s")
        else:
            em = model["emitter"]
            sw = em["sweep"]
            if sw["mode"] == "resonance":
                rs = model["resonance"]
                m = 0 if em["orientation"] == "axial" else 1
                say(f"PROGRESS 0/1 resonance of order m = {m} near {rs['wavelength_nm']:g} nm (centre of the spectrum)")
                modes, raw, dt = resonances(job, float(rs["wavelength_nm"]), m, int(rs["num_modes"]), int(jd.get("krylov_dimension", 0)))
                best = pick_fundamental(modes, float(rs["wavelength_nm"]))
                out["resonance"] = dict(target_nm=float(rs["wavelength_nm"]), m=m, modes=modes, time_s=dt, chosen=best)
                if best is None:
                    raise RuntimeError("keine Resonanz mit Q > 0 gefunden: Zielwellenlänge oder Anzahl der Moden ändern")
                span = float(sw.get("linewidths", 1.5)) * best["lam_nm"] / best["Q"]
                values = list(np.linspace(best["lam_nm"] - span, best["lam_nm"] + span, max(int(sw["n"]), 1)))
                say(f"  resonance {best['lam_nm']:.3f} nm, Q = {best['Q']:.4g}: spectrum {values[0]:.3f} ... {values[-1]:.3f} nm")
                flush()
            else:
                values = fa.sweep_values(sw)
            surfaces = emitter_surfaces(job)
            want = set(jd.get("field_indices", []))
            if jd.get("field_at_centre", True):
                want.add(len(values) // 2)
            n = len(values)
            for i, lam in enumerate(values):
                if job.cancelled():
                    say("CANCELLED")
                    out["cancelled"] = True
                    flush()
                    return 0
                say(f"PROGRESS {i + 1}/{n} lambda = {lam:.3f} nm")
                try:
                    res, fld = emitter_point(job, float(lam), surfaces, em, want_field=i in want)
                    res["index"] = i
                    out["points"].append(res)
                    if fld is not None:
                        np.savez_compressed(folder / f"field_{i}.npz", **fld)
                    say(f"  {res['time_s']:.1f} s: F_P = {res['purcell']:.4g}, beta_top = {res['beta_top']:.4f}, down = {res['beta_bottom']:.4f}")
                except Exception as exc:
                    say(f"ERROR point {i}: {exc}")
                    out["points"].append(dict(index=i, lam_nm=float(lam), error=str(exc)))
                flush()
    except Exception as exc:
        say(f"ERROR {exc}")
        out["error"] = str(exc)
        flush()
        return 1
    flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
