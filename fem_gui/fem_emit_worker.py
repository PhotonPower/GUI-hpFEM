"""Task "emitter" of the periodic mode: a Gaussian dipole in the cell (hp-FEM 0.5, M17, ADR-0013). Runs inside fem_worker.py (called from its main),
in the Python that has hpfem.

Stages (job["model"]["emitter"]["stage"]):
  A  hpfem.grating.emit: a Bloch-periodic array of dipoles at one (kx, beta), one cell problem per orientation
  B  hpfem.grating.dipole_emission: the single dipole by array scanning (Purcell factor, channels up / down / non-radiated),
     hpfem.grating.emission_cone for every collection aperture
  C  hpfem.grating.emission_pattern: dP/dOmega on a (theta, phi) grid, apertures from the grid

The cost (number of solves, memory of one factorisation, calibrated time) is estimated first (``COST {...}`` in the log, results["cost"]); with
``dry_run`` nothing else is computed. The mesh, the layer stack and the PML layers are those of the periodic mode (fem_worker.Case, prepare,
grating_options); cover_line and substrate_line are left to the library (it moves them away from the dipole).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import fem_emit as fe

NM = 1e-9


def say(*a):
    print(*a, flush=True)


def _listed(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def moment_spec(spec):
    """(key, vector or None) of "isotropic" / "x" / "y" / "z"."""
    if spec == "isotropic":
        return "isotropic", None
    if spec in ("x", "y", "z"):
        return spec, np.eye(3, dtype=complex)["xyz".index(spec)]
    raise ValueError(f"Orientierung {spec!r}: isotropic, x, y oder z")


def aperture_from_grid(theta, phi, d_omega, na, n):
    """Power into theta <= asin(NA / n) from dP/dOmega on a (theta, phi) grid: trapezoidal rule in theta, the phi samples as a uniform circle. The
    last interval is cut at the cone edge by linear interpolation of the ring integrand, so a grid that reaches just past the edge (30 degrees for
    NA = 0.5 in air, whose edge lies at 29.9993 degrees) gives the full cone. A grid that ends before the edge counts only what it covers."""
    theta_max = math.asin(min(1.0, na / n))
    theta = np.asarray(theta, dtype=float)
    ring = np.asarray(d_omega).mean(axis=1) * 2 * math.pi * np.sin(theta)                 # integral over phi per theta
    inside = theta <= theta_max + 1e-12
    t, f = theta[inside], ring[inside]
    if t.size and t[-1] < theta_max - 1e-12 and theta[-1] > theta_max:
        t, f = np.append(t, theta_max), np.append(f, np.interp(theta_max, theta, ring))
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t))) if len(t) > 1 else 0.0


def run(h, case, job, folder, Control, fw):
    """Runs the emitter task of the job; returns the exit code. fw is the module fem_worker (Case, prepare, grating_options)."""
    import hpfem.grating as grating

    model, solver = job["model"], job["solver"]
    em = fe.emitter_complete(model["emitter"])
    stage = str(em["stage"]).upper()
    inc, sw = job["incidence"], job["sweep"]
    wavelengths = [float(v) for v in sw["values"]] if sw["mode"] == "wavelength" else [float(inc["wavelength_nm"])]
    key, vector = moment_spec(em["moment"])
    dipole = {"position": [float(em["x_nm"]) * NM, float(em["y_nm"]) * NM], "sigma": float(em["sigma_nm"]) * NM}
    if vector is not None:
        dipole["moment"] = vector
    scan = em["scan"]
    scan_kwargs = dict(nodes=int(scan["nodes"]), kx_nodes=int(scan["kx_nodes"]), depth=float(scan["depth"]),
                       angle_nodes=tuple(int(v) for v in scan["angle_nodes"]), symmetric=bool(scan["symmetric"]), workers=int(scan["workers"]),
                       guided={"auto": "auto", "on": True, "off": False}.get(scan.get("guided", "auto"), "auto"))
    bmax = scan.get("beta_max_over_k0")
    apertures = [float(v) for v in _listed(em.get("apertures", []))]
    aperture_side = str(em.get("aperture_side", "up"))
    order = int(solver["order"])
    out = dict(meta=dict(model=model.get("name", ""), task="emitter", order=order, cells=int(case.mesh.num_cells), engine="grating",
                         started=time.strftime("%Y-%m-%d %H:%M:%S")),
               task="emitter",
               emitter=dict(stage=stage, moment=key, x_nm=float(em["x_nm"]), y_nm=float(em["y_nm"]), sigma_nm=float(em["sigma_nm"]), wavelengths=wavelengths,
                            apertures=apertures, aperture_side=aperture_side, scan={**scan_kwargs, "angle_nodes": list(scan_kwargs["angle_nodes"]),
                                                                                      "beta_max_over_k0": bmax}),
               points=[])
    path = Path(folder) / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    def contexts(lam):
        ctx = fw.prepare(case, lam)
        if ctx.crystals:
            raise ValueError("Emitter in Kristallen: nicht in dieser Oberfläche (isotrope Materialien)")
        materials = {1 + i: ctx.mat[n] for i, n in enumerate(case.names)}
        opts = fw.grating_options(case, ctx, solver)
        pml = opts.pop("pml")
        opts.pop("cover_line", None)
        opts.pop("substrate_line", None)
        return ctx, materials, pml, opts

    # ---- cost estimate (first wavelength)
    ctx, materials, pml, opts = contexts(wavelengths[0])
    k0 = ctx.k0
    est = grating.estimate_memory(case.mesh, order)
    n_theta, n_phi = scan_kwargs["angle_nodes"]
    if stage == "B":
        cost = grating.emission_cost(case.mesh, materials, ctx.stack, dipole, ctx.omega, order=order, pml=pml,
                                     beta_max=None if bmax is None else float(bmax) * k0, calibrate=bool(em.get("calibrate", True)),
                                     **scan_kwargs, **opts)
        cone = len(apertures) * 2 * len(grating._hemisphere_directions(n_theta, n_phi, scan_kwargs["symmetric"], True, (), 0.5)) if apertures else 0
        per = dict(samples=cost.samples, plane_wave_solves=cost.plane_wave_solves + cone)
        calib = dict(seconds_per_sample=cost.seconds_per_sample, seconds_per_plane_wave=cost.seconds_per_plane_wave)
    elif stage == "A":
        per, calib = dict(samples=len(["x", "y", "z"] if key == "isotropic" else [key]), plane_wave_solves=0), {}
    else:
        d = em["directions"]
        per, calib = dict(samples=0, plane_wave_solves=2 * len(d["theta_deg"]) * len(d["phi_deg"]) * len(d["sides"])), {}
    solves = per["samples"] + per["plane_wave_solves"]
    seconds = None
    if calib.get("seconds_per_sample") is not None:
        seconds = len(wavelengths) * (per["samples"] * calib["seconds_per_sample"] / max(scan_kwargs["workers"], 1)
                                      + per["plane_wave_solves"] * (calib["seconds_per_plane_wave"] or 0.0))
    out["cost"] = dict(stage=stage, per_wavelength=dict(per, solves=solves), wavelengths=len(wavelengths), solves=solves * len(wavelengths),
                       dofs=int(est.dofs), total_bytes=int(est.total_bytes), memory=est.describe(), seconds=seconds, **calib)
    say("COST " + json.dumps(out["cost"], default=float))
    flush()
    if em.get("dry_run"):
        out["dry_run"] = True
        flush()
        say("DONE")
        return 0

    Cancelled = getattr(h, "Cancelled", ())
    n = len(wavelengths)
    for i, lam in enumerate(wavelengths):
        if Control.cancelled():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return 0
        say(f"PROGRESS {i + 1}/{n} lambda = {lam:.3f} nm, stage {stage}")
        t0 = time.time()
        try:
            ctx, materials, pml, opts = contexts(lam)
            omega, k0 = ctx.omega, ctx.k0

            def progress(step, num, phase="", i=i):
                say(f"PHASE {phase} {int(step)}/{int(num)}")

            if stage == "A":
                point = stage_a(grating, case.mesh, materials, ctx.stack, dipole, key, omega, k0, em, order, pml, opts, progress, Control.cancelled)
            elif stage == "B":
                point = stage_b(grating, case.mesh, materials, ctx.stack, dipole, key, omega, k0, scan_kwargs, bmax, apertures, aperture_side, order, pml,
                                opts, progress, Control.cancelled)
            else:
                point = stage_c(grating, case.mesh, materials, ctx.stack, dipole, key, vector, omega, em, apertures, order, pml, opts, progress, Control.cancelled)
            point.update(i=i, wavelength=lam, seconds=time.time() - t0)
            out["points"].append(point)
            if stage == "B":
                say(f"  {point['seconds']:.1f} s: F_P = {point['purcell']:.5g}" + (f", up {point['fractions']['up']:.4f}, down {point['fractions']['down']:.4f}"
                                                                                   if point.get("fractions") else ""))
            else:
                say(f"  {point['seconds']:.1f} s")
        except Cancelled:
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return 0
        except Exception as exc:
            say(f"ERROR point {i}: {exc}")
            out["points"].append(dict(i=i, wavelength=lam, error=str(exc)))
        flush()
    say("DONE")
    return 0


def stage_a(grating, mesh, materials, stack, dipole, key, omega, k0, em, order, pml, opts, progress, cancel):
    """The Bloch array at (kx, beta), averaged over the orientations (one cell problem each)."""
    kx, beta = float(em["kx_over_k0"]) * k0, float(em["beta_over_k0"]) * k0
    moments = ["x", "y", "z"] if key == "isotropic" else [key]
    fields = ("P_cell", "up", "down", "absorbed", "guided")
    sums = dict.fromkeys(fields, 0.0)
    up, down, leak = {}, {}, 0.0
    for j, k in enumerate(moments):
        if cancel():
            raise _cancelled()
        progress(j, len(moments), "emit")
        r = grating.emit(mesh, materials, stack, {**dipole, "moment": np.eye(3)["xyz".index(k)]}, omega, kx, beta, order, pml=pml, **opts)
        for f in fields:
            sums[f] += getattr(r, f) / len(moments)
        for o in r.orders_up:
            up[o.m] = up.get(o.m, 0.0) + o.power / len(moments)
        for o in r.orders_down:
            down[o.m] = down.get(o.m, 0.0) + o.power / len(moments)
        leak += (r.pml_leak or 0.0) / len(moments)
    return dict(stage="A", kx=kx, beta=beta, **sums, pml_leak=leak, orders_up=[[m, p] for m, p in sorted(up.items())],
                orders_down=[[m, p] for m, p in sorted(down.items())])


def _cancelled():
    import hpfem
    return hpfem.Cancelled("emitter cancelled")


def stage_b(grating, mesh, materials, stack, dipole, key, omega, k0, scan_kwargs, bmax, apertures, side, order, pml, opts, progress, cancel):
    r = grating.dipole_emission(mesh, materials, stack, dipole, omega, order=order, pml=pml, beta_max=None if bmax is None else float(bmax) * k0,
                                progress=lambda i, n: progress(i, n, "scan"), cancel=cancel, **scan_kwargs, **opts)
    p_em = r.P_em[key]
    point = dict(stage="B", purcell=r.purcell[key], P_em=p_em, P_bulk=r.P_bulk if key != "moment" else p_em / r.purcell[key], n_host=r.n_host,
                 lossy=bool(r.lossy), samples=int(r.samples), plane_wave_solves=int(r.directions), purcell_by_orientation=dict(r.purcell),
                 timing=dict(r.timing))
    if r.guided is not None:
        point["guided"] = float(r.guided[key])
    if r.up is not None:
        point["fractions"] = dict(up=r.up[key] / p_em, down=r.down[key] / p_em, nonradiated=r.nonradiated[key] / p_em)
    point["aperture"] = []
    for na in apertures:
        cone = grating.emission_cone(mesh, materials, stack, dipole, omega, na, side=side, angle_nodes=scan_kwargs["angle_nodes"],
                                     symmetric=scan_kwargs["symmetric"], order=order, pml=pml,
                                     progress=lambda i, n, na=na: progress(i, n, f"aperture {na:g}"), cancel=cancel, **opts)
        power = cone.power[key]
        point["aperture"].append(dict(NA=na, side=side, theta_max_deg=math.degrees(cone.theta_max), power=power, fraction=power / p_em))
    return point


def stage_c(grating, mesh, materials, stack, dipole, key, vector, omega, em, apertures, order, pml, opts, progress, cancel):
    d = em["directions"]
    thetas, phis, sides = [float(v) for v in d["theta_deg"]], [float(v) for v in d["phi_deg"]], [str(v) for v in d["sides"]]
    spec = {**dipole, "moment": key if key != "moment" else dipole["moment"]}
    directions = [(math.radians(t), math.radians(f), s) for s in sides for t in thetas for f in phis]
    pat = grating.emission_pattern(mesh, materials, stack, spec, omega, directions, order=order, pml=pml,
                                   progress=lambda i, n: progress(i, n, "pattern"), cancel=cancel, **opts)
    shape = (len(thetas), len(phis))
    grid, by_pol, n_side, aps = {}, {}, {}, []
    for si, s in enumerate(sides):
        blk = slice(si * shape[0] * shape[1], (si + 1) * shape[0] * shape[1])
        total = pat.total[blk].reshape(shape)
        grid[s] = total.tolist()
        by_pol[s] = {pol: pat.dP_dOmega[blk, j].reshape(shape).tolist() for j, pol in enumerate(pat.pol)}
        n_side[s] = float(pat.n[blk][0])
        for na in apertures:
            pw = aperture_from_grid(np.radians(thetas), np.radians(phis), total, na, n_side[s])
            aps.append(dict(NA=na, side=s, power=pw, per_P_bulk=pw / pat.P_bulk))
    return dict(stage="C", theta_deg=thetas, phi_deg=phis, sides=sides, dP_dOmega=grid, dP_dOmega_by_pol=by_pol, P_bulk=pat.P_bulk, n=n_side,
                aperture=aps, timing=dict(pat.timing))
