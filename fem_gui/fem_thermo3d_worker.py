"""Worker for heat and mechanics in 3D (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_thermo3d_worker.py <job folder> [estimate]

Reads <folder>/job.json (the model of fem_thermo3d plus ``max_gb``; SI is derived here) and writes <folder>/results.json and <folder>/fields3d.npz.

* case ``particle``: ScatteringSetup3D on hpfem.box_with_ball (plane wave along +z, E along x, in the matrix, PML) -> absorbed power load -> Thermal3D
  (T = T_ref on the outer faces); with ``selfconsistent`` the loop hpfem.ThermoOptical3D (eps(T) = eps + d eps / dT (T - T_ref)); optional Elasticity3D
  (thermal expansion alpha per tag from the temperature, shrinkage inside the sphere; outer faces clamped or free). With ``quarter`` the model x > 0, y > 0
  (x = 0 PEC for the optics, y = 0 natural; heat and elasticity use the symmetry planes) and the power and energy multiplied by 4.
* case ``block``: hpfem.box; optional Thermal3D (uniform volumetric source, sink faces at T_ref, others adiabatic); Elasticity3D with boundary conditions per face,
  temperature change (linear along z or the heat solution) and shrinkage (linear along z).

With ``estimate`` only the mesh is built and the memory estimate is printed (``ESTIMATE{json}``); a run whose estimate exceeds ``max_gb`` stops with ERROR.

fields3d.npz: x, y, z (cell centres of a regular grid, unit nm for the particle and um for the block), 3D arrays (NaN outside the mesh) of T, Eabs, ux, uy, uz, umag,
sxx, syy, szz, sxy, syz, sxz, svm; sphere_r. Stress in MPa, displacement in nm. For the quarter model the grid is mirrored to the full box (displacement
components and shears with the sign of the mirror). Log lines on stdout: PROGRESS i/n stage, ERROR ..., CANCELLED, DONE. Cancellation: the file <folder>/cancel,
checked between the stages (a single solve cannot be interrupted).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_disp as fd  # noqa: E402

NM, UM = 1e-9, 1e-6
C0 = 299792458.0
GB = 1024.0**3
TAG_SPHERE, TAG_SYM_X, TAG_SYM_Y = 2, 7, 8
SIDE_TAGS = {"xmin": 1, "xmax": 2, "ymin": 3, "ymax": 4, "zmin": 5, "zmax": 6}
SIDE_NORMAL = {"xmin": (-1.0, 0.0, 0.0), "xmax": (1.0, 0.0, 0.0), "ymin": (0.0, -1.0, 0.0), "ymax": (0.0, 1.0, 0.0), "zmin": (0.0, 0.0, -1.0), "zmax": (0.0, 0.0, 1.0)}
ODD_X, ODD_Y = ("ux", "sxy", "sxz"), ("uy", "sxy", "syz")
FIELD_NAMES = ["T", "Eabs", "ux", "uy", "uz", "umag", "sxx", "syy", "szz", "sxy", "syz", "sxz", "svm"]


def say(*a):
    print(*a, flush=True)


class Cancelled(Exception):
    pass


def check_cancel(folder):
    if (folder / "cancel").exists():
        raise Cancelled()


def material_at(job, name, omega):
    return fd.dispersive_model(job["materials"][name]).at(omega)


def elastic(h, props):
    return h.ElasticMaterial(float(props["E_GPa"]) * 1e9, float(props["nu"]))


# ---------------------------------------------------------------------------------------------------------------------- meshes
def build_mesh(h, job):
    """(mesh, info): SI units; for the quarter model the cut planes x = 0 / y = 0 get the facet tags 7 / 8."""
    if job["case"] == "block":
        b = job["block"]
        mesh = h.box(int(b["nx"]), int(b["ny"]), int(b["nz"]), [0.0, 0.0, 0.0], [b["W_um"] * UM, b["D_um"] * UM, b["H_um"] * UM])
        return mesh, dict(quarter=False)
    p = job["particle"]
    r = p["radius_nm"] * NM
    hw, outer = p["half_factor"] * r, p["outer_factor"] * r
    mesh = h.box_with_ball(int(p["n"]), r, hw, outer, TAG_SPHERE)
    quarter = bool(p["quarter"])
    if quarter:
        mesh = h.extract(mesh, lambda x: bool(x[0] > 0 and x[1] > 0))
        tol = 1e-9 * r
        for f in mesh.boundary_facets:
            if mesh.facet_tag(int(f)) != h.NO_TAG:
                continue
            vs = [mesh.vertex(int(v)) for v in mesh.facet_vertices(int(f))]
            if all(abs(v[0]) < tol for v in vs):
                mesh.set_facet_tag(int(f), TAG_SYM_X)
            elif all(abs(v[1]) < tol for v in vs):
                mesh.set_facet_tag(int(f), TAG_SYM_Y)
    return mesh, dict(quarter=quarter, r=r, hw=hw, outer=outer)


def time_hint(dofs):
    """Rough time of one optical solve (SparseLU, one thread): about 70 s at 18 000 unknowns, ~ N^1.7 (as measured for fem_threed)."""
    return 70.0 * (max(dofs, 1) / 18264.0) ** 1.7


def estimate(h, job, mesh):
    import fem_thermo3d as t3

    order = int(job["order"])
    h1 = h.DofMap3D(mesh, order)
    me = t3.mech_estimate(h1.num_dofs)
    est = dict(cells=int(mesh.num_cells), h1_dofs=int(h1.num_dofs), mech_dofs=me["dofs"], mech_seconds=me["seconds"], mech_bytes=int(me["bytes"]))
    nbytes = me["bytes"] if (job["case"] == "block" or job["mech"]["on"]) else 0.0
    if job["case"] == "particle":
        nd = h.NedelecDofMap3D(mesh, order)
        e = h.estimate_memory(nd)
        est.update(nd_dofs=int(nd.num_dofs), optics_bytes=int(e.total_bytes), optics_text=e.describe(), optics_seconds=time_hint(nd.num_dofs))
        nbytes = max(nbytes, float(e.total_bytes))
    est["bytes"] = int(nbytes)
    est["human"] = h.format_bytes(int(nbytes))
    return est


# ---------------------------------------------------------------------------------------------------------------------- optics and heat
def sink_tags(info):
    t = SIDE_TAGS
    return [t["xmax"], t["ymax"], t["zmin"], t["zmax"]] if info["quarter"] else list(t.values())


def run_optics_heat(h, job, mesh, info, nd, h1, out):
    """Returns (temperature coefficients, total E coefficients, omega)."""
    pj, optics, th = job["particle"], job["optics"], job["thermal"]
    lam = optics["wavelength_nm"] * NM
    omega = 2 * math.pi * C0 / lam
    k0 = omega / C0
    bg_name, in_name = job["background"], job["inclusion"]
    mat_bg, mat_in = material_at(job, bg_name, omega), material_at(job, in_name, omega)
    n_bg = math.sqrt(complex(mat_bg.eps_r).real)
    hw, outer = info["hw"], info["outer"]
    d = outer - hw
    prof = h.PmlProfile(3, 1e-8)
    s = h.ScatteringSetup3D()
    s.omega = omega
    s.materials = h.MaterialMap(mat_bg)
    s.materials.set(TAG_SPHERE, mat_in)
    s.incident = h.plane_wave(np.array([optics["E0_MVm"] * 1e6, 0.0, 0.0], dtype=complex), np.array([0.0, 0.0, n_bg * k0]))
    s.formulation = h.Formulation.SCATTERED_FIELD
    t = SIDE_TAGS
    if info["quarter"]:
        s.pml = h.PmlBox3D(np.array([0.0, 0.0, -hw]), np.array([hw, hw, hw]), [0.0, d, 0.0, d, d, d], k0, n_bg, prof)
        s.pec_tags = [t["xmax"], t["ymax"], t["zmin"], t["zmax"], TAG_SYM_X]
    else:
        s.pml = h.PmlBox3D.uniform(np.full(3, -hw), np.full(3, hw), d, k0, n_bg, prof)
        s.pec_tags = list(t.values())
    tp = h.ThermalSetup()
    tp.background_conductivity = float(job["props"][bg_name]["kappa"])
    tp.conductivity = {TAG_SPHERE: float(job["props"][in_name]["kappa"])}
    tref = float(th["T_ref"])
    tp.fixed_temperature = [(tg, tref) for tg in sink_tags(info)]
    mult = 4.0 if info["quarter"] else 1.0
    out["scalars"]["Freiheitsgrade Optik (Nédélec)"] = int(nd.num_dofs)
    out["scalars"]["Freiheitsgrade Wärme"] = int(h1.num_dofs)
    if th["selfconsistent"]:
        to = h.ThermoOpticalSetup3D()
        to.optical, to.thermal, to.reference_temperature = s, tp, tref
        to.relaxation, to.tolerance, to.max_iterations = float(th["relaxation"]), float(th["tolerance"]), int(th["max_iter"])
        coeff = {}
        for tag, name in ((0, bg_name), (TAG_SPHERE, in_name)):
            c = complex(job["props"][name]["deps_re"], job["props"][name]["deps_im"])
            if c != 0:
                coeff[tag] = c
        to.thermo_optic = coeff
        prob = h.ThermoOptical3D(nd, h1, to)
        state = prob.solve()
        temp = np.asarray(state.temperature, dtype=complex)
        e_tot = np.asarray(prob.total_field(state.solution), dtype=complex)
        out["history"] = [float(v) for v in state.history]
        out["scalars"]["Schleife konvergiert"] = "ja" if state.converged else "NEIN (Toleranz nicht erreicht)"
        out["scalars"]["Iterationen"] = int(state.iterations)
        out["scalars"]["Absorbierte Leistung (W)"] = float(state.absorbed_power) * mult
        out["converged"] = bool(state.converged)
    else:
        prob = h.Scattering3D(nd, s)
        sol = prob.solve()
        e_tot = np.asarray(sol.unknown, dtype=complex) + np.asarray(h.interpolate(nd, s.incident.value), dtype=complex)
        ap = h.absorbed_power_by_tag(prob, sol)
        load = h.absorbed_power_load(nd, e_tot, omega, s.materials, h1)
        temp = np.asarray(h.Thermal3D(h1, tp).solve_load(load), dtype=complex)
        out["scalars"]["Absorbierte Leistung (W)"] = float(ap.total) * mult
        for tag, w in dict(ap.by_tag).items():
            if tag == TAG_SPHERE:
                out["scalars"]["  davon in der Kugel (W)"] = float(w) * mult
    return temp, e_tot


def run_block_heat(h, job, mesh, h1, out):
    """Thermal3D of the block: uniform source q, sink faces at T_ref, the others adiabatic. Returns the temperature coefficients."""
    th = job["thermal"]
    tp = h.ThermalSetup()
    tp.background_conductivity = float(job["props"][job["background"]]["kappa"])
    tp.fixed_temperature = [(SIDE_TAGS[s], float(th["T_ref"])) for s, v in th["sinks"].items() if v]
    q = float(th["q_mWum3"]) * 1e15
    qc = np.asarray(h.interpolate(h1, lambda x: complex(q)), dtype=complex)
    prob = h.Thermal3D(h1, tp)
    t = np.asarray(prob.solve(qc), dtype=complex)
    out["scalars"]["Freiheitsgrade Wärme"] = int(h1.num_dofs)
    out["scalars"]["Wärmeleistung im Quader (W)"] = float(prob.total_power(qc))
    return t


# ---------------------------------------------------------------------------------------------------------------------- mechanics
def side_conditions(h, job):
    """(displacement conditions, traction conditions, any_fixed) from the per-face boundary conditions of the block."""
    disp, trac, fixed = [], [], False
    for side, tag in SIDE_TAGS.items():
        c = job["mech"]["bc"][side]
        kind = c["kind"]
        n = SIDE_NORMAL[side]
        if kind == "clamped":
            disp.append(h.DisplacementCondition3D(tag, [True, True, True]))
            fixed = True
        elif kind == "symmetry":
            disp.append(h.DisplacementCondition3D(tag, [abs(v) > 0 for v in n]))
            fixed = True
        elif kind == "traction" and float(c["value"]) != 0.0:
            sg = float(c["value"]) * 1e6
            tv = np.array([n[0] * sg, n[1] * sg, n[2] * sg], dtype=complex)
            trac.append(h.TractionCondition3D(tag, lambda x, tv=tv: tv))
    return disp, trac, fixed


def solve_mechanics(h, job, mesh, info, h1, temp):
    """Elasticity3D for the job: returns (problem, u)."""
    case, mech, props = job["case"], job["mech"], job["props"]
    es = h.ElasticitySetup3D()
    tref = float(job["thermal"]["T_ref"])
    es.reference_temperature = tref
    es.static_condensation = False
    if case == "particle":
        r = info["r"]
        s = float(mech["shrink_a_pct"]) / 100.0
        es.background = elastic(h, props[job["background"]])
        es.materials = {TAG_SPHERE: elastic(h, props[job["inclusion"]])}
        es.background_expansion = float(props[job["background"]]["alpha"]) * 1e-6
        es.thermal_expansion = {TAG_SPHERE: float(props[job["inclusion"]]["alpha"]) * 1e-6}
        es.temperature = temp
        if s:
            es.shrinkage = lambda x, r=r, s=s: s if (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) < r * r else 0.0
        disp = []
        if info["quarter"]:
            disp += [h.DisplacementCondition3D(TAG_SYM_X, [True, False, False]), h.DisplacementCondition3D(TAG_SYM_Y, [False, True, False])]
        if mech["wall"] == "clamped":
            disp += [h.DisplacementCondition3D(tg, [True, True, True]) for tg in sink_tags(info)]
        else:
            es.remove_rigid_modes = True
        es.displacements = disp
    else:
        p = props[job["background"]]
        es.background = elastic(h, p)
        es.background_expansion = float(p["alpha"]) * 1e-6
        W, D, H = job["block"]["W_um"] * UM, job["block"]["D_um"] * UM, job["block"]["H_um"] * UM
        if temp is not None:
            es.temperature = temp
        else:
            da, db = float(mech["dT_a"]), float(mech["dT_b"])
            if da != 0 or db != 0:
                es.temperature = np.asarray(h.interpolate(h1, lambda x: complex(tref + da + (db - da) * x[2] / H)), dtype=complex)
        sa, sb = float(mech["shrink_a_pct"]) / 100.0, float(mech["shrink_b_pct"]) / 100.0
        if sa or sb:
            es.shrinkage = lambda x, sa=sa, sb=sb, H=H: sa + (sb - sa) * x[2] / H
        disp, trac, fixed = side_conditions(h, job)
        es.displacements, es.tractions = disp, trac
        if not fixed:
            es.remove_rigid_modes = True
    prob = h.Elasticity3D(h1, es)
    u = np.asarray(prob.solve(), dtype=complex)
    return prob, u


# ---------------------------------------------------------------------------------------------------------------------- fields
def von_mises(s):
    sx, sy, sz, sxy, syz, sxz = s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2]
    return math.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2))


def grid_axes(job, mesh, info):
    """Cell-centre axes (SI) of the sample grid: the whole box; for the quarter model only the quadrant x, y > 0 (mirrored afterwards)."""
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    g = int(job["grid"])
    ext = hi - lo
    if info["quarter"]:
        ext_full = np.array([2 * ext[0], 2 * ext[1], ext[2]])
        n_full = np.maximum(4, np.round(g * ext_full / ext_full.max()).astype(int))
        n = np.array([max(2, n_full[0] // 2), max(2, n_full[1] // 2), n_full[2]])
    else:
        n = np.maximum(3, np.round(g * ext / ext.max()).astype(int))
    return [lo[i] + (np.arange(n[i]) + 0.5) * ext[i] / n[i] for i in range(3)]


def sample_fields(h, job, mesh, info, nd, h1, temp, e_tot, mech):
    xs, ys, zs = grid_axes(job, mesh, info)
    loc = h.PointLocator3D(mesh)
    shape = (len(xs), len(ys), len(zs))
    f = {k: np.full(shape, np.nan) for k in FIELD_NAMES}
    prob, u = mech if mech else (None, None)
    e0 = float(job["optics"]["E0_MVm"]) * 1e6 if job["case"] == "particle" else 1.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                lp = loc.locate(np.array([x, y, z]))
                if lp is None:
                    continue
                c, xi = lp.cell, np.asarray(lp.xi)
                if temp is not None:
                    f["T"][i, j, k] = h.evaluate_h1(h1, temp, c, xi).real
                if e_tot is not None:
                    ev = np.asarray(h.evaluate_hcurl(nd, e_tot, c, xi))
                    f["Eabs"][i, j, k] = math.sqrt(float(np.sum(np.abs(ev) ** 2))) / e0
                if prob is not None:
                    d = np.asarray(prob.displacement(u, c, xi)).real / NM
                    f["ux"][i, j, k], f["uy"][i, j, k], f["uz"][i, j, k] = d
                    f["umag"][i, j, k] = float(np.linalg.norm(d))
                    s = np.asarray(prob.stress(u, c, xi)).real / 1e6
                    f["sxx"][i, j, k], f["syy"][i, j, k], f["szz"][i, j, k] = s[0, 0], s[1, 1], s[2, 2]
                    f["sxy"][i, j, k], f["syz"][i, j, k], f["sxz"][i, j, k] = s[0, 1], s[1, 2], s[0, 2]
                    f["svm"][i, j, k] = von_mises(s)
    return [xs, ys, zs], f


def mirror(axes, f, info):
    """Quarter model -> full box: mirror about x = 0 and y = 0 (displacement components and shears flip sign across their mirror)."""
    if not info["quarter"]:
        return axes, f
    xs, ys, zs = axes
    out = {}
    for k, a in f.items():
        sx = -1.0 if k in ODD_X else 1.0
        a = np.concatenate([sx * a[::-1, :, :], a], axis=0)
        sy = -1.0 if k in ODD_Y else 1.0
        out[k] = np.concatenate([sy * a[:, ::-1, :], a], axis=1)
    return [np.concatenate([-xs[::-1], xs]), np.concatenate([-ys[::-1], ys]), zs], out


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem as h
    h.set_log_level("warn")
    case = job["case"]
    particle = case == "particle"
    order = int(job["order"])
    heat_on = particle or bool(job["thermal"]["heat"])
    mech_on = bool(job["mech"]["on"]) or not particle
    out = dict(meta=dict(name=job.get("name", ""), case=case, dim=3, started=time.strftime("%Y-%m-%d %H:%M:%S"), hpfem=str(getattr(h, "__version__", "?")), order=order),
               scalars={}, history=None, probes={})
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float, ensure_ascii=False), encoding="utf-8")

    stages = (["Optik und Wärme"] if particle else (["Wärme"] if heat_on else [])) + (["Mechanik"] if mech_on else []) + ["Felder"]

    def stage(i):
        check_cancel(folder)
        say(f"PROGRESS {i + 1}/{len(stages)} {stages[i]}")

    try:
        t0 = time.time()
        if not (len(argv) > 1 and argv[1] == "estimate"):
            f3 = folder / "fields3d.npz"
            if f3.exists():
                f3.unlink()
        mesh, info = build_mesh(h, job)
        est = estimate(h, job, mesh)
        if len(argv) > 1 and argv[1] == "estimate":
            print("ESTIMATE" + json.dumps(est), flush=True)
            return 0
        say(f"hpfem {getattr(h, '__version__', '?')}, mesh: {est['cells']} cells, p = {order}, {est['h1_dofs']} H1 unknowns"
            + (f", {est['nd_dofs']} optical unknowns ({est['optics_text']})" if particle else "")
            + (f", elasticity ~{est['mech_dofs']} unknowns (~{est['mech_seconds']:.0f} s, ~{est['mech_bytes'] / GB:.2f} GB)" if mech_on else ""))
        if est["bytes"] / GB > float(job.get("max_gb", 8.0)):
            say(f"ERROR memory estimate {est['human']} exceeds the limit of {job.get('max_gb', 8.0)} GB: reduce the mesh or the order (or raise the limit).")
            out["error"] = f"Speicherschätzung {est['human']} über der Grenze von {job.get('max_gb', 8.0)} GB"
            flush()
            return 1
        out["meta"].update(cells=int(est["cells"]), dofs_h1=int(est["h1_dofs"]), estimate=est)
        h1 = h.DofMap3D(mesh, order)
        nd = h.NedelecDofMap3D(mesh, order) if particle else None
        i = 0
        temp = e_tot = None
        if heat_on:
            stage(i)
            t1 = time.time()
            if particle:
                temp, e_tot = run_optics_heat(h, job, mesh, info, nd, h1, out)
                out["time_optics_s"] = time.time() - t1
                say(f"  {time.time() - t1:.1f} s, P_abs = {out['scalars'].get('Absorbierte Leistung (W)', float('nan')):.4g} W")
            else:
                temp = run_block_heat(h, job, mesh, h1, out)
                say(f"  {time.time() - t1:.1f} s")
            flush()
            i += 1
        mech = None
        if mech_on:
            stage(i)
            t1 = time.time()
            prob, u = solve_mechanics(h, job, mesh, info, h1, temp if (heat_on or particle) else None)
            mech = (prob, u)
            mult = 4.0 if info["quarter"] else 1.0
            out["scalars"]["Dehnungsenergie (J)"] = float(prob.strain_energy(u)) * mult
            out["scalars"]["Freiheitsgrade Mechanik"] = int(prob.num_dofs)
            say(f"  {time.time() - t1:.1f} s")
            flush()
            i += 1
        stage(i)
        loc = h.PointLocator3D(mesh)
        if particle:
            r = info["r"]
            lp = loc.locate(np.array([1e-3 * r, 1e-3 * r, 1e-3 * r]))
            if lp is not None and temp is not None:
                out["probes"]["T_center"] = float(h.evaluate_h1(h1, temp, lp.cell, np.asarray(lp.xi)).real)
        else:
            b = job["block"]
            W, D, H = b["W_um"] * UM, b["D_um"] * UM, b["H_um"] * UM
            if temp is not None:
                for name, pt in (("T_mid", [W / 2, D / 2, H / 2]), ("T_end", [W * (1 - 1e-9), D / 2, H / 2])):
                    lp = loc.locate(np.array(pt))
                    if lp is not None:
                        out["probes"][name] = float(h.evaluate_h1(h1, temp, lp.cell, np.asarray(lp.xi)).real)
            if mech:
                d = np.asarray(mech[0].displacements_at(mech[1], np.array([[W, D / 2, H / 2], [W / 2, D, H / 2], [W / 2, D / 2, H]]))).real
                out["probes"].update(ux_end=float(d[0, 0]), uy_end=float(d[1, 1]), uz_end=float(d[2, 2]))
        axes, f = sample_fields(h, job, mesh, info, nd, h1, temp, e_tot, mech)
        axes, f = mirror(axes, f, info)
        arrays = {k: v for k, v in f.items() if np.isfinite(v).any()}
        scale = 1.0 / (NM if particle else UM)
        if "T" in arrays:
            tmax = float(np.nanmax(arrays["T"]))
            out["scalars"]["Maximale Temperatur auf dem Raster (K)"] = tmax
            out["scalars"]["Temperaturanstieg max. auf dem Raster (K)"] = tmax - float(job["thermal"]["T_ref"])
        if "T_center" in out["probes"]:
            out["scalars"]["Temperaturanstieg im Kugelmittelpunkt (K)"] = out["probes"]["T_center"] - float(job["thermal"]["T_ref"])
        if "umag" in arrays:
            out["scalars"]["Größte Verschiebung auf dem Raster (nm)"] = float(np.nanmax(arrays["umag"]))
            out["scalars"]["Größte Von-Mises-Spannung auf dem Raster (MPa)"] = float(np.nanmax(arrays["svm"]))
        np.savez_compressed(folder / "fields3d.npz", x=axes[0] * scale, y=axes[1] * scale, z=axes[2] * scale, unit=np.array("nm" if particle else "um"),
                            sphere_r=np.array(job["particle"]["radius_nm"] if particle else 0.0), **arrays)
        out["meta"]["total_s"] = time.time() - t0
        out["finished"] = True
        flush()
        say("DONE")
    except Cancelled:
        say("CANCELLED")
        out["cancelled"] = True
        flush()
    except Exception as exc:
        say(f"ERROR {type(exc).__name__}: {exc}")
        out["error"] = f"{type(exc).__name__}: {exc}"
        flush()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
