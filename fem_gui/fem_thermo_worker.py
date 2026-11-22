"""Worker for heat and mechanics in 2D (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_thermo_worker.py <job folder>

Reads <folder>/job.json (the model of fem_thermo, SI is derived here) and writes <folder>/results.json and <folder>/fields.npz.

* case ``wire``: ScatteringSetup2D on hpfem.square_with_disc (plane wave in the matrix, PML) -> absorbed power load -> Thermal2D (T = T_ref on the
  outer wall); with ``selfconsistent`` the loop hpfem.ThermoOptical2D (eps(T) = eps + d eps / dT (T - T_ref)); optional Elasticity2D (plane strain,
  thermal expansion alpha per tag from the temperature, shrinkage inside the disc).
* cases ``block`` / ``axi``: Elasticity2D on hpfem.rectangle (plane strain, or meridian half-plane with the axis on the side x = 0) with boundary
  conditions per side, temperature change (linear along x resp. z) and shrinkage.

fields.npz: x, y (grid, cell centres, unit nm for the wire and um otherwise), 2D arrays (NaN outside the mesh) of T, Eabs, ux, uy, umag, sxx, syy, sxy, szz, svm,
the deformed mesh (mesh_vertices, mesh_cells, mesh_disp in the length unit), circle_r. Stress in MPa, displacement in nm.
Log lines on stdout: PROGRESS i/n stage, ERROR ..., CANCELLED, DONE. Cancellation: the file <folder>/cancel, checked between the stages.
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
import fem_materials as fm  # noqa: E402

NM, UM = 1e-9, 1e-6
TWO_PI = 2 * math.pi


def say(*a):
    print(*a, flush=True)


class Cancelled(Exception):
    pass


def check_cancel(folder):
    if (folder / "cancel").exists():
        raise Cancelled()


def material_at(job, name, omega):
    """hpfem.Material of the job's material at the angular frequency omega (dispersion evaluated there)."""
    return fd.dispersive_model(job["materials"][name]).at(omega)


def elastic(h, props):
    return h.ElasticMaterial(float(props["E_GPa"]) * 1e9, float(props["nu"]))


# ---------------------------------------------------------------------------------------------------------------------- optics and heat
def run_optics_heat(h, job, mesh, nd, h1, out):
    """Returns (temperature coefficients, total E coefficients or None). Fills out['scalars'] and out['history']."""
    wire, optics, th = job["wire"], job["optics"], job["thermal"]
    r = wire["radius_nm"] * NM
    hw, outer = wire["half_factor"] * r, wire["outer_factor"] * r
    lam = optics["wavelength_nm"] * NM
    omega = 2 * math.pi * 299792458.0 / lam
    k0 = omega / 299792458.0
    bg_name, in_name = job["background"], job["inclusion"]
    mat_bg, mat_in = material_at(job, bg_name, omega), material_at(job, in_name, omega)
    n_bg = math.sqrt(complex(mat_bg.eps_r).real)
    sides = [1, 2, 3, 4]
    s = h.ScatteringSetup2D()
    s.omega = omega
    s.pec_tags = sides
    s.materials = h.MaterialMap(mat_bg)
    s.materials.set(2, mat_in)
    s.incident = h.plane_wave([0.0, optics["E0_MVm"] * 1e6], [n_bg * k0, 0.0])
    s.formulation = h.Formulation.SCATTERED_FIELD
    s.pml = h.PmlBox2D.uniform([-hw, -hw], [hw, hw], outer - hw, k0, n_bg, h.PmlProfile(2, 1e-10))
    tp = h.ThermalSetup()
    tp.background_conductivity = float(job["props"][bg_name]["kappa"])
    tp.conductivity = {2: float(job["props"][in_name]["kappa"])}
    tref = float(th["T_ref"])
    tp.fixed_temperature = [(t, tref) for t in sides]
    out["scalars"]["Zellen (Optik und Wärme)"] = int(mesh.num_cells)
    out["scalars"]["Freiheitsgrade Optik (Nédélec)"] = int(nd.num_dofs)
    out["scalars"]["Freiheitsgrade Wärme"] = int(h1.num_dofs)
    if th["selfconsistent"]:
        to = h.ThermoOpticalSetup2D()
        to.optical, to.thermal, to.reference_temperature = s, tp, tref
        to.relaxation, to.tolerance, to.max_iterations = float(th["relaxation"]), float(th["tolerance"]), int(th["max_iter"])
        coeff = {}
        for tag, name in ((0, bg_name), (2, in_name)):
            c = complex(job["props"][name]["deps_re"], job["props"][name]["deps_im"])
            if c != 0:
                coeff[tag] = c
        to.thermo_optic = coeff
        prob = h.ThermoOptical2D(nd, h1, to)
        state = prob.solve()
        temp = np.asarray(state.temperature, dtype=complex)
        e_tot = np.asarray(prob.total_field(state.solution), dtype=complex)
        out["history"] = [float(v) for v in state.history]
        out["scalars"]["Schleife konvergiert"] = "ja" if state.converged else "NEIN (Toleranz nicht erreicht)"
        out["scalars"]["Iterationen"] = int(state.iterations)
        out["scalars"]["Absorbierte Leistung (W/m)"] = float(state.absorbed_power)
        out["converged"] = bool(state.converged)
    else:
        prob = h.Scattering2D(nd, s)
        sol = prob.solve()
        e_tot = np.asarray(sol.unknown, dtype=complex) + np.asarray(h.interpolate(nd, s.incident.value), dtype=complex)
        ap = h.absorbed_power_by_tag(prob, sol)
        load = h.absorbed_power_load(nd, e_tot, omega, s.materials, h1)
        temp = np.asarray(h.Thermal2D(h1, tp).solve_load(load), dtype=complex)
        out["scalars"]["Absorbierte Leistung (W/m)"] = float(ap.total)
        for tag, w in dict(ap.by_tag).items():
            if tag == 2:
                out["scalars"]["  davon im Draht (W/m)"] = float(w)
    return temp, e_tot


# ---------------------------------------------------------------------------------------------------------------------- mechanics
def side_conditions(h, job):
    """(displacement conditions, traction conditions, any_fixed) from the per-side boundary conditions of block / axi."""
    from fem_thermo import SIDE_NORMAL, SIDE_TAGS

    disp, trac, fixed = [], [], False
    for side, tag in SIDE_TAGS.items():
        c = job["mech"]["bc"][side]
        kind = c["kind"]
        if kind == "clamped":
            disp.append(h.DisplacementCondition2D(tag, [True, True]))
            fixed = True
        elif kind == "symmetry":
            nx, ny = SIDE_NORMAL[side]
            disp.append(h.DisplacementCondition2D(tag, [abs(nx) > 0, abs(ny) > 0]))
            fixed = True
        elif kind == "traction" and float(c["value"]) != 0.0:
            sg = float(c["value"]) * 1e6
            t = np.array([SIDE_NORMAL[side][0] * sg, SIDE_NORMAL[side][1] * sg], dtype=complex)
            trac.append(h.TractionCondition2D(tag, lambda x, t=t: t))
    return disp, trac, fixed


def solve_mechanics(h, job, mesh, h1, temp, out):
    """Elasticity2D for the job: returns (problem, u, axisymmetric flag)."""
    case, mech = job["case"], job["mech"]
    props = job["props"]
    es = h.ElasticitySetup2D()
    shrink = float(mech["shrink_pct"]) / 100.0
    tref = float(job["thermal"]["T_ref"])
    if case == "wire":
        r = job["wire"]["radius_nm"] * NM
        es.background = elastic(h, props[job["background"]])
        es.materials = {2: elastic(h, props[job["inclusion"]])}
        es.background_expansion = float(props[job["background"]]["alpha"]) * 1e-6
        es.thermal_expansion = {2: float(props[job["inclusion"]]["alpha"]) * 1e-6}
        es.temperature = temp
        es.reference_temperature = tref
        if shrink:
            es.shrinkage = lambda x, r=r, s=shrink: s if (x[0] ** 2 + x[1] ** 2) < r * r else 0.0
        if mech["wall"] == "clamped":
            es.displacements = [h.DisplacementCondition2D(t, [True, True]) for t in (1, 2, 3, 4)]
        else:
            es.remove_rigid_modes = True
    else:
        axi = case == "axi"
        p = props[job["background"]]
        es.background = elastic(h, p)
        es.background_expansion = float(p["alpha"]) * 1e-6
        es.reference_temperature = tref
        b = job["block"]
        length = (b["height_um"] if axi else b["width_um"]) * UM
        da, db = float(mech["dT_a"]), float(mech["dT_b"])
        if da != 0 or db != 0:
            es.temperature = np.asarray(h.interpolate(h1, lambda x: complex(tref + da + (db - da) * (x[1] if axi else x[0]) / length)), dtype=complex)
        if shrink:
            es.shrinkage = lambda x, s=shrink: s
        disp, trac, fixed = side_conditions(h, job)
        es.displacements, es.tractions = disp, trac
        if not fixed:
            es.remove_rigid_modes = True
        if axi:
            es.axisymmetric, es.axis_tag = True, 1
    es.static_condensation = False
    prob = h.Elasticity2D(h1, es)
    u = np.asarray(prob.solve(), dtype=complex)
    return prob, u


# ---------------------------------------------------------------------------------------------------------------------- fields
def grid_points(mesh, n, unit):
    v = np.asarray(mesh.vertices)[:, :2]
    lo, hi = v.min(axis=0), v.max(axis=0)
    nx = int(n)
    ny = max(2, int(round(n * (hi[1] - lo[1]) / (hi[0] - lo[0]))))
    xs = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
    ys = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    return xs, ys


def von_mises(sx, sy, sxy, sz):
    return math.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * sxy**2)


def sample_fields(h, job, mesh, nd, h1, temp, e_tot, mech, out, unit):
    """Raster fields on a regular grid (cell centres), NaN outside the mesh."""
    xs, ys = grid_points(mesh, job["grid"], unit)
    loc = h.PointLocator2D(mesh)
    shape = (len(ys), len(xs))
    names = ["T", "Eabs", "ux", "uy", "umag", "sxx", "syy", "sxy", "szz", "svm"]
    f = {k: np.full(shape, np.nan) for k in names}
    prob, u, axi = mech if mech else (None, None, False)
    nus = None
    if prob is not None:
        nus = {0: float(job["props"][job["background"]]["nu"])}
        if job["case"] == "wire":
            nus[2] = float(job["props"][job["inclusion"]]["nu"])
    e0 = float(job["optics"]["E0_MVm"]) * 1e6 if job["case"] == "wire" else 1.0
    tags = np.asarray(mesh.cell_tags)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            lp = loc.locate(np.array([x, y]))
            if lp is None:
                continue
            c, xi = lp.cell, np.asarray(lp.xi)
            if temp is not None:
                f["T"][j, i] = h.evaluate_h1(h1, temp, c, xi).real
            if e_tot is not None:
                ev = h.evaluate_hcurl(nd, e_tot, c, xi)
                f["Eabs"][j, i] = math.sqrt(float(np.sum(np.abs(np.asarray(ev)) ** 2))) / e0
            if prob is not None:
                d = np.asarray(prob.displacement(u, c, xi)).real
                f["ux"][j, i], f["uy"][j, i] = d[0] / NM, d[1] / NM
                f["umag"][j, i] = math.hypot(d[0], d[1]) / NM
                s = np.asarray(prob.stress(u, c, xi)).real / 1e6
                sx, sy, sxy = s[0, 0], s[1, 1], s[0, 1]
                if axi:
                    sz = prob.hoop_stress(u, c, xi).real / 1e6
                else:
                    sz = nus.get(int(tags[c]), nus[0]) * (sx + sy)
                f["sxx"][j, i], f["syy"][j, i], f["sxy"][j, i], f["szz"][j, i] = sx, sy, sxy, sz
                f["svm"][j, i] = von_mises(sx, sy, sxy, sz)
    return xs, ys, f


def probes(prob, u, job):
    """Displacements at the points the closed-form checks use (fem_thermo.analytic_checks)."""
    b = job["block"]
    W, H = b["width_um"] * UM, b["height_um"] * UM
    d = np.asarray(prob.displacements_at(u, np.array([[W, H / 2], [W / 2, H]]))).real
    return {"ux_end": float(d[0, 0]), "uy_top": float(d[1, 1])}


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem as h
    h.set_log_level("warn")
    for f in ("fields.npz",):
        if (folder / f).exists():
            (folder / f).unlink()
    case = job["case"]
    wire = case == "wire"
    mech_on = bool(job["mech"]["on"]) or not wire
    order = int(job["order"])
    stages = (["Optik und Wärme"] if wire else []) + (["Mechanik"] if mech_on else []) + ["Felder"]
    out = dict(meta=dict(name=job.get("name", ""), case=case, started=time.strftime("%Y-%m-%d %H:%M:%S"), hpfem=str(getattr(h, "__version__", "?")), order=order),
               scalars={}, history=None)
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float, ensure_ascii=False), encoding="utf-8")

    def stage(i):
        check_cancel(folder)
        say(f"PROGRESS {i + 1}/{len(stages)} {stages[i]}")

    try:
        t0 = time.time()
        unit = "nm" if wire else "um"
        if wire:
            w = job["wire"]
            r = w["radius_nm"] * NM
            mesh = h.square_with_disc(int(w["cells_per_radius"]), r, w["half_factor"] * r, w["outer_factor"] * r, 2)
        else:
            b = job["block"]
            mesh = h.rectangle(int(b["nx"]), int(b["ny"]), [0.0, 0.0], [b["width_um"] * UM, b["height_um"] * UM])
        h1 = h.DofMap2D(mesh, order)
        nd = h.NedelecDofMap2D(mesh, order) if wire else None
        say(f"hpfem {getattr(h, '__version__', '?')}, mesh: {mesh.num_cells} cells, p = {order}")
        out["meta"].update(cells=int(mesh.num_cells), dofs_h1=int(h1.num_dofs))
        i = 0
        temp = e_tot = None
        if wire:
            stage(i)
            t1 = time.time()
            temp, e_tot = run_optics_heat(h, job, mesh, nd, h1, out)
            out["scalars"]["Maximale Temperatur (K)"] = None
            out["time_optics_s"] = time.time() - t1
            say(f"  {time.time() - t1:.1f} s, P_abs = {out['scalars'].get('Absorbierte Leistung (W/m)', float('nan')):.4g} W/m")
            flush()
            i += 1
        mech = None
        if mech_on:
            stage(i)
            t1 = time.time()
            prob, u = solve_mechanics(h, job, mesh, h1, temp, out)
            mech = (prob, u, case == "axi")
            out["scalars"]["Dehnungsenergie (J/m)" if case != "axi" else "Dehnungsenergie (J pro rad)"] = float(prob.strain_energy(u))
            out["scalars"]["Freiheitsgrade Mechanik"] = int(prob.num_dofs)
            if not wire:
                out["probes"] = probes(prob, u, job)
            say(f"  {time.time() - t1:.1f} s")
            i += 1
        stage(i)
        xs, ys, f = sample_fields(h, job, mesh, nd, h1, temp, e_tot, mech, out, unit)
        scale = 1.0 / (NM if wire else UM)
        arrays = {k: v for k, v in f.items() if np.isfinite(v).any()}
        if "T" in arrays:
            out["scalars"]["Maximale Temperatur (K)"] = float(np.nanmax(arrays["T"]))
            out["scalars"]["Temperaturanstieg max. (K)"] = float(np.nanmax(arrays["T"]) - float(job["thermal"]["T_ref"]))
        else:
            out["scalars"].pop("Maximale Temperatur (K)", None)
        if "umag" in arrays:
            out["scalars"]["Größte Verschiebung (nm)"] = float(np.nanmax(arrays["umag"]))
            out["scalars"]["Größte Von-Mises-Spannung (MPa)"] = float(np.nanmax(arrays["svm"]))
        meshv = np.asarray(mesh.vertices)[:, :2]
        extra = {}
        if mech:
            dv = np.asarray(mech[0].deformed_mesh(mech[1], 1.0).vertices)[:, :2] - meshv
            extra = dict(mesh_vertices=meshv * scale, mesh_cells=np.asarray(mesh.cells, dtype=int), mesh_disp=dv * scale)
        np.savez_compressed(folder / "fields.npz", x=xs * scale, y=ys * scale, unit=np.array(unit), circle_r=np.array(job["wire"]["radius_nm"] if wire else 0.0),
                            **arrays, **extra)
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
