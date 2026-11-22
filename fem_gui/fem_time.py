"""Time domain (hp-FEM 0.5) without hpfem and without Streamlit: models, meshing with Gmsh, job documents, results and figures.

The solver is ``hpfem.TimeDomain2D`` / ``TimeDomain3D`` (Newmark-beta on the Nedelec space, default beta = 1/4, gamma = 1/2: the trapezoidal rule,
unconditionally stable and energy conserving, no numerical damping): the second-order wave equation for E

    eps E'' + sigma E' + curl(mu^-1 curl E) = -J g'(t)

with real materials by cell tag (eps_r, mu_r = 1), a conductivity sigma [S/m] by tag, PEC facets (tangential E = 0), absorbing facets (first-order
Silver-Mueller condition: exact only for normal incidence, oblique waves are partly reflected), natural (untagged) facets = PMC, and a
current density J(x) with the time signal g(t). Nothing is dispersive: the permittivities are evaluated once at the reference wavelength lambda_0.

Models (nm, fs in the models; SI in the solver):

* ``dim = "2d"``: a rectangle (x from 0 to L, y from -W/2 to W/2) with the shapes of the waveguide mode (``fem_wg.SHAPES``; priority by order,
  later ones cover earlier ones), E in the plane (E_x, E_y), H_z out of the plane (one z-invariant section; results per unit length).
  The sides carry the tags 101 / 102 / 103 / 104 (left / right / bottom / top).
* ``dim = "3d"``: the cube [-R, R]^3 (R = outer_factor * radius) with a spherical inclusion (``hpfem.box_with_ball``, curved cells) in a
  background medium; the faces carry the hpfem box tags 1 ... 6 (x_min, x_max, y_min, y_max, z_min, z_max).

Source: ``J(x) = J0 exp(-|x - x_s|^2 / (2 s^2)) p`` with the unit polarisation p; the signal is a modulated Gaussian ``sin(w (t - t0)) exp(-(t - t0)^2 /
(2 tau^2))`` (carrier lambda_0, width tau), a Gaussian pulse ``exp(-(t - t0)^2 / (2 tau^2))`` or a continuous wave ``sin(w t)`` with soft
switch-on (logistic ramp, time constant tau, centred at t0 = delay * tau). The solver feeds -J g'(t): a pulse has no DC part.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd

import fem_materials as fm
import fem_wg as fw

C0 = 299792458.0
EPS0 = 8.8541878128e-12
NM, FS = 1e-9, 1e-15
TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP = fw.TAG_LEFT, fw.TAG_RIGHT, fw.TAG_BOTTOM, fw.TAG_TOP
DIMS = {"2d": "2D: Ebene (z-invariant, E in der Ebene)", "3d": "3D: Kasten mit Kugel (sehr kleines Netz)"}
SIGNALS = {"modulated": "Modulierter Gauß-Puls (Trägerwelle)", "gaussian": "Gauß-Puls (breitbandig)", "cw": "Dauerstrich mit weichem Einschalten"}
BOUNDARY = {"pec": "ideal leitend (PEC)", "abs": "absorbierend (Silver-Müller)", "pmc": "magnetisch (PMC, natürlich)"}
SIDES_2D = {"left": ("links (x = 0)", TAG_LEFT), "right": ("rechts (x = L)", TAG_RIGHT), "bottom": ("unten (y = −B/2)", TAG_BOTTOM), "top": ("oben (y = +B/2)", TAG_TOP)}
SIDES_3D = {"xmin": ("x −", 1), "xmax": ("x +", 2), "ymin": ("y −", 3), "ymax": ("y +", 4), "zmin": ("z −", 5), "zmax": ("z +", 6)}
AXES_3D = {"x": "x", "y": "y", "z": "z"}
QUANTITIES = {"abs": "|E|", "ex": "E_x", "ey": "E_y", "ez": "E_z"}
WINDOWS = {"none": "kein Fenster", "hann": "Hann-Fenster"}
MAX_STEPS = 20000
MAX_DOFS_3D = 30000


# ===================================================================================================================== models
def default_model():
    """Pulse in a vacuum strip: a modulated Gaussian (500 nm carrier) from a point-like source in the middle, absorbing ends, PEC top and bottom."""
    return {
        "name": "Pulsausbreitung im Streifen", "dim": "2d",
        "materials": {"Vakuum": {"type": "eps", "re": 1.0, "im": 0.0}, "Glas": {"type": "index", "n": 1.5, "k": 0.0}},
        "background": "Vakuum", "sigma": {}, "loss_as_sigma": False,
        "length_nm": 4000.0, "width_nm": 800.0, "shapes": [],
        "boundary": {"left": "abs", "right": "abs", "bottom": "pec", "top": "pec"},
        "sphere": {"material": "Glas", "radius_nm": 300.0, "outer_factor": 2, "half_factor": 1.2, "cells": 1},
        "boundary3": {k: "pec" for k in SIDES_3D},
        "source": {"kind": "modulated", "lambda_nm": 500.0, "width_fs": 3.0, "delay": 4.0, "x_nm": 2000.0, "y_nm": 0.0, "z_nm": 0.0, "size_nm": 60.0,
                   "angle_deg": 90.0, "axis": "z", "amplitude": 1.0},
        "probes": [[1000.0, 0.0], [3000.0, 0.0]],
        "time": {"steps_per_period": 40, "dt_fs": 0.0, "t_end_fs": 0.0, "f_max_THz": 0.0, "frames": 60, "gamma": 0.5, "beta": 0.25},
        "frame_z_nm": 0.0, "grid": 61,
        "mesh": {"cells_per_wavelength": 4.0, "order": 2, "interface_factor": 0.8},
    }


def presets():
    base = default_model()
    out = {base["name"]: base}
    glass = copy.deepcopy(base)
    glass.update(name="Puls trifft einen Glasblock (n = 1,5)", shapes=[{"type": "rect", "material": "Glas", "x0": 2600.0, "x1": 3400.0, "y0": -500.0, "y1": 500.0}],
                 probes=[[1200.0, 0.0], [3700.0, 0.0]])
    glass["source"].update(x_nm=1200.0)
    out[glass["name"]] = glass
    cav = copy.deepcopy(base)
    cav.update(name="Scheibenresonator n = 3 in PEC-Kasten (Ausschwingen, Spektrum)", length_nm=1800.0, width_nm=1800.0,
               shapes=[{"type": "disc", "material": "Si-ähnlich", "cx": 900.0, "cy": 0.0, "r": 500.0}], probes=[[1250.0, 150.0]])
    cav["materials"]["Si-ähnlich"] = {"type": "index", "n": 3.0, "k": 0.0}
    cav["boundary"] = {k: "pec" for k in SIDES_2D}
    cav["source"].update(kind="gaussian", width_fs=1.0, x_nm=700.0, y_nm=100.0, size_nm=40.0, angle_deg=90.0)
    cav["time"].update(t_end_fs=60.0)
    out[cav["name"]] = cav
    lossy = copy.deepcopy(base)
    lossy.update(name="Leitfähiger Block (σ = 1·10⁵ S/m): Dämpfung des Pulses",
                 shapes=[{"type": "rect", "material": "Leiter", "x0": 2500.0, "x1": 3300.0, "y0": -500.0, "y1": 500.0}], probes=[[1200.0, 0.0], [3700.0, 0.0]])
    lossy["materials"]["Leiter"] = {"type": "eps", "re": 2.0, "im": 0.0}
    lossy["sigma"] = {"Leiter": 1.0e5}
    lossy["source"].update(x_nm=1200.0)
    out[lossy["name"]] = lossy
    sph = copy.deepcopy(base)
    sph.update(name="3D: dielektrische Kugel (n = 2) im PEC-Kasten", dim="3d", probes=[[250.0, 0.0, 0.0]])
    sph["materials"]["Kugelmaterial"] = {"type": "index", "n": 2.0, "k": 0.0}
    sph["sphere"] = {"material": "Kugelmaterial", "radius_nm": 300.0, "outer_factor": 2, "half_factor": 1.2, "cells": 1}
    sph["source"].update(kind="gaussian", width_fs=1.0, x_nm=-150.0, y_nm=0.0, z_nm=0.0, size_nm=60.0, axis="z")
    sph["time"].update(steps_per_period=30, t_end_fs=30.0)
    sph["mesh"].update(order=2, cells_per_wavelength=4.0)
    out[sph["name"]] = sph
    return out


# ===================================================================================================================== signal and estimates
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def signal_value(kind, omega, t0, tau, t):
    """g(t) as in the hpfem signals: ``gaussian_pulse`` / ``modulated_gaussian`` and the continuous wave with logistic switch-on."""
    t = np.asarray(t, dtype=float)
    if kind == "gaussian":
        return np.exp(-((t - t0) ** 2) / (2 * tau**2))
    if kind == "modulated":
        return np.sin(omega * (t - t0)) * np.exp(-((t - t0) ** 2) / (2 * tau**2))
    return np.sin(omega * t) * sigmoid((t - t0) / tau)


def signal_derivative(kind, omega, t0, tau, t):
    t = np.asarray(t, dtype=float)
    if kind == "gaussian":
        return -(t - t0) / tau**2 * np.exp(-((t - t0) ** 2) / (2 * tau**2))
    if kind == "modulated":
        env = np.exp(-((t - t0) ** 2) / (2 * tau**2))
        return omega * np.cos(omega * (t - t0)) * env - (t - t0) / tau**2 * np.sin(omega * (t - t0)) * env
    s = sigmoid((t - t0) / tau)
    return omega * np.cos(omega * t) * s + np.sin(omega * t) * s * (1 - s) / tau


def domain_bounds(m):
    """((x0, x1), (y0, y1), (z0, z1)) of the computational domain in nm (z only in 3D)."""
    if m["dim"] == "2d":
        L, W = float(m["length_nm"]), float(m["width_nm"])
        return (0.0, L), (-W / 2, W / 2), (0.0, 0.0)
    R = float(m["sphere"]["radius_nm"]) * float(m["sphere"]["outer_factor"])
    return (-R, R), (-R, R), (-R, R)


def reference_indices(m):
    """Largest refractive index n of the used materials at lambda_0 (for the resolution)."""
    lam = float(m["source"]["lambda_nm"])
    ns = [1.0]
    names = used_materials(m)
    for n in names:
        try:
            ns.append(math.sqrt(max(fm.eps_at(m["materials"][n], lam).real, 1e-12)))
        except Exception:
            pass
    return max(ns)


def used_materials(m):
    names = {m["background"]}
    if m["dim"] == "2d":
        names |= {s["material"] for s in m["shapes"]}
    else:
        names.add(m["sphere"]["material"])
    return names


def estimate(m):
    """Derived quantities of the model (nm / fs / THz): carrier, highest frequency, shortest wavelength, time step, number of steps, times."""
    s, tm = m["source"], m["time"]
    lam0 = float(s["lambda_nm"])
    f0 = C0 / (lam0 * NM) / 1e12
    tau = float(s["width_fs"])
    spread = 3.0 / (2 * math.pi * tau) * 1e3                                                                 # THz (tau in fs)
    kind = s["kind"]
    f_auto = (4.0 / (2 * math.pi * tau) * 1e3) if kind == "gaussian" else f0 + spread
    f_max = float(tm.get("f_max_THz", 0.0)) or f_auto
    dt = float(tm["dt_fs"]) or 1.0 / (f_max * 1e12 * int(tm["steps_per_period"])) / FS
    t0 = float(s["delay"]) * tau
    (x0, x1), (y0, y1), (z0, z1) = domain_bounds(m)
    diag = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + ((z1 - z0) ** 2 if m["dim"] == "3d" else 0.0))
    n_ref = reference_indices(m)
    t_auto = t0 + 3 * tau + 1.0 * diag * NM * n_ref / C0 / FS
    t_end = float(tm["t_end_fs"]) or t_auto
    steps = int(math.ceil(t_end / dt)) if dt > 0 else 0
    return dict(f0_THz=f0, f_max_THz=f_max, lam_min_nm=C0 / (f_max * 1e12) / NM, dt_fs=dt, t0_fs=t0, t_end_fs=t_end, steps=steps, n_ref=n_ref,
                bandwidth_THz=spread, period_fs=1e3 / f_max, diag_nm=diag, resolution_f_THz=1e3 / t_end if t_end > 0 else float("inf"))


def tag_of_material(m, name):
    return 1 + list(m["materials"]).index(name)


def eps_sigma(m, name):
    """(Re eps, sigma [S/m]) of a material at lambda_0; with ``loss_as_sigma`` the loss Im eps becomes eps0 omega_0 Im eps."""
    lam = float(m["source"]["lambda_nm"])
    e = complex(fm.eps_at(m["materials"][name], lam))
    sig = float(m.get("sigma", {}).get(name, 0.0))
    if m.get("loss_as_sigma") and e.imag > 0:
        sig += EPS0 * 2 * math.pi * C0 / (lam * NM) * e.imag
    return float(e.real), sig


# ===================================================================================================================== validation
def validate(m):
    """List of (level, text) with level "error" or "warn"."""
    out = []
    s, tm = m["source"], m["time"]
    mats = m["materials"]
    lam = float(s["lambda_nm"])
    if lam <= 0 or float(s["width_fs"]) <= 0 or float(s["size_nm"]) <= 0:
        out.append(("error", "Wellenlänge, Pulsbreite und Quellgröße müssen positiv sein."))
        return out
    for n in used_materials(m):
        if n not in mats:
            out.append(("error", f"Material „{n}“ fehlt."))
            continue
        if fm.is_crystal(mats[n]):
            out.append(("error", f"{n}: Kristalle sind im Zeitbereich nicht möglich (isotrope, reelle Materialien)."))
            continue
        try:
            e = complex(fm.eps_at(mats[n], lam))
        except Exception as exc:
            out.append(("error", f"{n}: {exc}"))
            continue
        if e.real <= 0:
            out.append(("error", f"{n}: Re ε = {e.real:.3g} ≤ 0 bei λ₀: der Zeitbereichslöser braucht positive, konstante ε (kein Metall)."))
        if e.imag > 1e-9 and not m.get("loss_as_sigma"):
            out.append(("error", f"{n}: verlustbehaftet (Im ε = {e.imag:.3g}); der Löser rechnet Verlust nur als Leitfähigkeit σ. "
                                 "Leitfähigkeit eintragen oder die Umrechnung Im ε → σ einschalten."))
        elif e.imag > 1e-9:
            out.append(("warn", f"{n}: Im ε wird als σ = ε₀ ω₀ Im ε abgebildet: nur für schmalbandige Pulse bei λ₀ richtig."))
        if fm.is_dispersive(mats[n]):
            out.append(("warn", f"{n}: ε ist dispersiv, wird aber nur bei λ₀ = {lam:g} nm ausgewertet (konstant im Zeitbereich)."))
    for n, v in m.get("sigma", {}).items():
        if float(v) < 0:
            out.append(("error", f"{n}: die Leitfähigkeit muss ≥ 0 sein."))
    if any(lv == "error" for lv, _ in out):
        return out
    (x0, x1), (y0, y1), (z0, z1) = domain_bounds(m)
    if m["dim"] == "2d":
        if float(m["length_nm"]) <= 0 or float(m["width_nm"]) <= 0:
            out.append(("error", "Länge und Breite des Gebiets müssen positiv sein."))
            return out
        for i, sh in enumerate(m["shapes"]):
            if sh["material"] not in mats:
                out.append(("error", f"Form {i + 1}: Material „{sh['material']}“ fehlt."))
                continue
            try:
                a0, a1, b0, b1 = fw.shape_bounds(sh)
            except Exception:
                out.append(("error", f"Form {i + 1}: ungültige Angaben."))
                continue
            if sh["type"] in ("strip", "rect") and (a1 <= a0 or b1 <= b0):
                out.append(("error", f"Form {i + 1}: Ausdehnung ≤ 0."))
            if sh["type"] in ("disc", "ring") and (sh["r"] <= 0 or (sh["type"] == "ring" and not 0 < sh["w"] < 2 * sh["r"])):
                out.append(("error", f"Form {i + 1}: Radius und Breite müssen 0 < w < 2r erfüllen."))
            if sh["type"] == "poly" and len(sh["points"]) < 3:
                out.append(("error", f"Form {i + 1}: ein Polygon braucht mindestens 3 Punkte."))
            if a1 < x0 or a0 > x1 or b1 < y0 or b0 > y1:
                out.append(("warn", f"Form {i + 1} liegt außerhalb des Gebiets."))
        bd = m["boundary"]
    else:
        sp = m["sphere"]
        if sp["material"] not in mats:
            out.append(("error", f"Kugelmaterial „{sp['material']}“ fehlt."))
        if float(sp["radius_nm"]) <= 0:
            out.append(("error", "Der Kugelradius muss positiv sein."))
        if int(sp["outer_factor"]) < 2 or int(sp["cells"]) < 1 or float(sp["half_factor"]) < 1.0:
            out.append(("error", "3D-Gebiet: Außenmaß ≥ 2 Radien (ganzzahlig), Zellen je Radius ≥ 1, Gleichmaß-Grenze ≥ 1 Radius."))
        if float(sp["half_factor"]) * 1.0 > int(sp["outer_factor"]):
            out.append(("error", "Die Grenze des gleichmäßigen Netzes liegt außerhalb des Kastens."))
        bd = m["boundary3"]
    if not any(v in ("pec", "abs") for v in bd.values()):
        out.append(("warn", "Nur natürliche (PMC) Ränder: Das Gebiet ist ein geschlossener magnetischer Resonator."))
    pos = [float(s["x_nm"]), float(s["y_nm"])] + ([float(s["z_nm"])] if m["dim"] == "3d" else [])
    lo, hi = [x0, y0] + ([z0] if m["dim"] == "3d" else []), [x1, y1] + ([z1] if m["dim"] == "3d" else [])
    if any(p < a or p > b for p, a, b in zip(pos, lo, hi)):
        out.append(("error", "Die Quelle liegt außerhalb des Gebiets."))
    for i, p in enumerate(m["probes"]):
        need = 3 if m["dim"] == "3d" else 2
        if len(p) < need:
            out.append(("error", f"Sonde {i + 1}: {need} Koordinaten nötig."))
        elif any(float(c) < a or float(c) > b for c, a, b in zip(p[:need], lo, hi)):
            out.append(("error", f"Sonde {i + 1} liegt außerhalb des Gebiets."))
    if not m["probes"]:
        out.append(("warn", "Keine Sonde: es gibt dann kein Zeitsignal und kein Spektrum (Energie und Felder schon)."))
    if float(tm["dt_fs"]) < 0 or float(tm["t_end_fs"]) < 0 or float(tm.get("f_max_THz", 0.0)) < 0:
        out.append(("error", "Zeitschritt, Endzeit und höchste Frequenz dürfen nicht negativ sein."))
        return out
    if not 0.5 <= float(tm["gamma"]) <= 1.0 or float(tm["beta"]) <= 0:
        out.append(("error", "Newmark: γ ≥ 1/2 und β > 0 (β = 1/4, γ = 1/2 ist die Trapezregel)."))
    elif float(tm["beta"]) < (float(tm["gamma"]) + 0.5) ** 2 / 4 - 1e-12:
        out.append(("error", "Newmark: β ≥ (γ + 1/2)² / 4 für unbedingte Stabilität (β = 1/4, γ = 1/2; oder γ = 0,6, β = 0,3025)."))
    elif float(tm["gamma"]) > 0.5:
        out.append(("warn", "γ > 1/2: numerische Dämpfung; die Energie nimmt auch ohne Verlust ab, hohe Frequenzen am stärksten."))
    est = estimate(m)
    if est["steps"] > MAX_STEPS:
        out.append(("error", f"{est['steps']} Zeitschritte sind zu viel (Grenze {MAX_STEPS}): Endzeit kürzen oder den Zeitschritt vergrößern."))
    elif est["steps"] < 2:
        out.append(("error", "Weniger als 2 Zeitschritte."))
    spp = est["period_fs"] / est["dt_fs"] if est["dt_fs"] > 0 else 0.0
    if spp < 10:
        out.append(("warn", f"Nur {spp:.1f} Zeitschritte je Periode der höchsten Frequenz: die Trapezregel ist stabil, aber ihr Phasenfehler "
                            f"(ω Δt)²/12 je Bogenmaß wird groß; ≥ 20–40 wählen."))
    mesh = m["mesh"]
    if int(mesh["order"]) < 1 or float(mesh["cells_per_wavelength"]) <= 0:
        out.append(("error", "Ordnung ≥ 1 und Zellen je Wellenlänge > 0."))
    if m["dim"] == "3d" and int(mesh["order"]) > 3:
        out.append(("warn", "3D mit p > 3: die Faktorisierung braucht viel Zeit und Speicher."))
    if float(s["size_nm"]) < est["lam_min_nm"] / 40:
        out.append(("warn", "Die Quelle ist viel kleiner als die kleinste Wellenlänge und das Netz: sie wird nur durch wenige Quadraturpunkte erfasst."))
    if int(tm["frames"]) < 0 or int(tm["frames"]) > 400:
        out.append(("error", "Zahl der Feldbilder 0 … 400."))
    return out


# ===================================================================================================================== mesh
def wg_model(m):
    """The model as the waveguide module wants it (geometry, materials, mesh settings) for ``fem_wg.build_mesh`` and ``fem_wg.fig_preview``."""
    d = copy.deepcopy(m)
    lam = float(m["source"]["lambda_nm"])
    d.update(task="ports", ports={"left": {"enabled": True, "modes": 1}, "right": {"enabled": True, "modes": 1}}, modes={"num": 1}, field="none",
             sweep={"mode": "single", "start": lam, "stop": lam, "n": 1}, excite={"port": 0, "mode": 0})
    return d


def build_mesh(m, out_dir):
    """2D: Gmsh mesh (MSH 4.1, nm) in out_dir/mesh.msh; 3D: no file (the worker builds ``box_with_ball``); returns statistics."""
    est = estimate(m)
    if m["dim"] == "3d":
        return {"cells": None, "h_max_nm": None, "lam_min_nm": est["lam_min_nm"]}
    stats = fw.build_mesh(wg_model(m), est["lam_min_nm"], out_dir)
    stats["lam_min_nm"] = est["lam_min_nm"]
    return stats


# ===================================================================================================================== job
def pol_vector(m):
    s = m["source"]
    if m["dim"] == "2d":
        a = math.radians(float(s["angle_deg"]))
        return [math.cos(a), math.sin(a)]
    return [1.0 if s["axis"] == a else 0.0 for a in "xyz"]


def job(m):
    """The job document for the worker: all derived quantities in SI (m, s) plus the model."""
    est = estimate(m)
    s, tm = m["source"], m["time"]
    names = list(m["materials"])
    mats = {n: dict(zip(("eps", "sigma"), eps_sigma(m, n))) for n in names if n in used_materials(m)}
    d = dict(name=m["name"], dim=m["dim"], materials=mats, background=m["background"], names=names, dt=est["dt_fs"] * FS, steps=est["steps"],
             gamma=float(tm["gamma"]), beta=float(tm["beta"]), order=int(m["mesh"]["order"]), frames=int(tm["frames"]),
             source=dict(kind=s["kind"], omega=2 * math.pi * C0 / (float(s["lambda_nm"]) * NM), t0=est["t0_fs"] * FS, tau=float(s["width_fs"]) * FS,
                         pos=[float(s["x_nm"]) * NM, float(s["y_nm"]) * NM] + ([float(s["z_nm"]) * NM] if m["dim"] == "3d" else []),
                         size=float(s["size_nm"]) * NM, pol=pol_vector(m), amplitude=float(s["amplitude"])),
             probes=[[float(c) * NM for c in p[: (3 if m["dim"] == "3d" else 2)]] for p in m["probes"]], estimate=est, max_dofs=MAX_DOFS_3D,
             pec_tags=[], absorbing_tags=[])
    sides = SIDES_2D if m["dim"] == "2d" else SIDES_3D
    bd = m["boundary"] if m["dim"] == "2d" else m["boundary3"]
    for k, (_, tag) in sides.items():
        if bd[k] == "pec":
            d["pec_tags"].append(tag)
        elif bd[k] == "abs":
            d["absorbing_tags"].append(tag)
    if m["dim"] == "2d":
        d["material_tags"] = {n: tag_of_material(m, n) for n in mats}
    else:
        sp = m["sphere"]
        d["sphere"] = dict(material=sp["material"], radius=float(sp["radius_nm"]) * NM, outer=float(sp["radius_nm"]) * NM * int(sp["outer_factor"]),
                           half=float(sp["radius_nm"]) * NM * float(sp["half_factor"]), cells=int(sp["cells"]))
        d["frame_z"] = float(m["frame_z_nm"]) * NM
        d["grid"] = int(m["grid"])
    return d


# ===================================================================================================================== results
def energy_dataframe(res):
    return pd.DataFrame({"t (fs)": np.asarray(res["t_energy_fs"]), "Energie (J)": np.asarray(res["energy"])})


def probe_dataframe(res):
    """One row per time step: t (fs), the drive g'(t) and the components of every probe."""
    t = np.asarray(res["t_fs"])
    cols = {"t (fs)": t, "Quelle g'(t)": np.asarray(res["drive"])}
    for i, p in enumerate(res["probes"]):
        for comp in ("ex", "ey", "ez"):
            if comp in p:
                cols[f"Sonde {i + 1} E_{comp[1]} (V/m)"] = np.asarray(p[comp])
    return pd.DataFrame(cols)


def spectrum(t_fs, y, window="none", pad=8):
    """(f [THz], |Y(f)|) of the real signal y(t) sampled at the uniform times t_fs (rfft, zero padded to pad times the length, trapezoid weights
    dt): |Y(f)| in (V/m) fs."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t_fs, dtype=float)
    dt = float(t[1] - t[0])
    if window == "hann":
        y = y * np.hanning(len(y))
    n = int(2 ** math.ceil(math.log2(max(len(y) * pad, 8))))
    f = np.fft.rfftfreq(n, d=dt * 1e-15) / 1e12
    return f, np.abs(np.fft.rfft(y, n)) * dt


def transfer(res, probe, comp, window="none", floor=0.02):
    """(f [THz], |E(f)| / |G'(f)|) where the source spectrum G' (the drive g'(t)) is above ``floor`` of its maximum, else NaN."""
    t = res["t_fs"]
    f, e = spectrum(t, res["probes"][probe][comp], window)
    _, s = spectrum(t, res["drive"], window)
    ok = s > floor * s.max()
    out = np.full_like(e, np.nan)
    out[ok] = e[ok] / s[ok]
    return f, out


def peak_frequency(f, a, lo=None, hi=None):
    """Frequency of the maximum of the spectrum a(f) between lo and hi."""
    f, a = np.asarray(f), np.asarray(a)
    ok = np.ones_like(f, dtype=bool)
    if lo is not None:
        ok &= f >= lo
    if hi is not None:
        ok &= f <= hi
    return float(f[ok][np.nanargmax(a[ok])])


def envelope(y):
    """Envelope |y + i H[y]| of a real signal (analytic signal by FFT)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    spec = np.fft.fft(y)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spec * h))


def arrival_time(t_fs, y):
    """Time of the maximum of the envelope of y (parabola through the largest sample and its neighbours), fs."""
    y = envelope(y)
    t = np.asarray(t_fs, dtype=float)
    k = int(np.argmax(y))
    if 0 < k < len(y) - 1:
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        if den != 0:
            return float(t[k] + 0.5 * (a - c) / den * (t[1] - t[0]))
    return float(t[k])


def lam_axis(f_thz):
    with np.errstate(divide="ignore"):
        return C0 / (np.asarray(f_thz) * 1e12) / NM


# ===================================================================================================================== figures
def _outline_box(ax, m):
    (x0, x1), (y0, y1), _ = domain_bounds(m)
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1.2)


def fig_preview(m):
    """The domain with materials, source and probes (2D: shapes; 3D: the section z = 0 of the box and the sphere)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if m["dim"] == "2d":
        fig = fw.fig_preview(wg_model(m))
        ax = fig.axes[0]
        for ln in list(ax.lines):                                     # drop the red port lines of the waveguide preview
            if ln.get_color() == "r":
                ln.remove()
        for tx in list(ax.texts):
            tx.remove()
    else:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        _outline_box(ax, m)
        sp = m["sphere"]
        ax.add_patch(Circle((0, 0), float(sp["radius_nm"]), facecolor="tab:orange", alpha=0.6, edgecolor="k", label=f"Kugel ({sp['material']})"))
        (x0, x1), (y0, y1), _ = domain_bounds(m)
        ax.set_xlim(x0 * 1.1, x1 * 1.1)
        ax.set_ylim(y0 * 1.1, y1 * 1.1)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm), Schnitt z = 0")
    s = m["source"]
    ax.plot([float(s["x_nm"])], [float(s["y_nm"])], "r*", ms=12, label="Quelle")
    ax.add_patch(Circle((float(s["x_nm"]), float(s["y_nm"])), float(s["size_nm"]), fill=False, edgecolor="r", lw=0.8))
    for i, p in enumerate(m["probes"]):
        ax.plot([float(p[0])], [float(p[1])], "bo", ms=5)
        ax.annotate(str(i + 1), (float(p[0]), float(p[1])), textcoords="offset points", xytext=(4, 4), fontsize=8, color="b")
    sides = SIDES_2D if m["dim"] == "2d" else {k: v for k, v in SIDES_3D.items() if k.startswith(("x", "y"))}
    bd = m["boundary"] if m["dim"] == "2d" else m["boundary3"]
    (x0, x1), (y0, y1), _ = domain_bounds(m)
    for k in sides:
        mode = bd[k]
        col = {"pec": "k", "abs": "tab:green", "pmc": "tab:purple"}[mode]
        if k in ("left", "xmin"):
            ax.plot([x0, x0], [y0, y1], color=col, lw=3, alpha=0.7)
        elif k in ("right", "xmax"):
            ax.plot([x1, x1], [y0, y1], color=col, lw=3, alpha=0.7)
        elif k in ("bottom", "ymin"):
            ax.plot([x0, x1], [y0, y0], color=col, lw=3, alpha=0.7)
        else:
            ax.plot([x0, x1], [y1, y1], color=col, lw=3, alpha=0.7)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="r", marker="*", ls="", label="Quelle"), Line2D([], [], color="b", marker="o", ls="", label="Sonden"),
               Line2D([], [], color="k", lw=3, label="PEC"), Line2D([], [], color="tab:green", lw=3, label="absorbierend"),
               Line2D([], [], color="tab:purple", lw=3, label="PMC")]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig_probes(res, which=None, comps=("ex", "ey", "ez")):
    """E(t) at the probes (and the drive g'(t) scaled to the maximum of the first curve)."""
    import matplotlib.pyplot as plt

    t = np.asarray(res["t_fs"])
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    shown = which if which is not None else range(len(res["probes"]))
    for i in shown:
        for c in comps:
            if c in res["probes"][i]:
                ax.plot(t, res["probes"][i][c], lw=1.0, label=f"Sonde {i + 1}, E_{c[1]}")
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("E (V/m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_energy(res, log=False):
    """Discrete energy (v^T M v + u^T S u)/2 versus time (J; in 2D per metre of z)."""
    import matplotlib.pyplot as plt

    t, e = np.asarray(res["t_energy_fs"]), np.asarray(res["energy"])
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.plot(t, e, "k-", lw=1.2)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(max(e.max() * 1e-8, 1e-300), e.max() * 2)
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("Energie (J)" if res["meta"]["dim"] == "3d" else "Energie (J je m in z)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_spectrum(res, probes, comp, window="none", mode="spectrum", axis="f", show_source=True):
    """FFT of the probe signals: |E(f)| (mode "spectrum") or |E(f)| / |G'(f)| ("transfer"); x axis frequency (THz) or vacuum wavelength (nm)."""
    import matplotlib.pyplot as plt

    est = res["meta"]["estimate"]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    t = res["t_fs"]
    f_hi = 1.6 * est["f_max_THz"]
    for i in probes:
        if mode == "transfer":
            f, a = transfer(res, i, comp, window)
        else:
            f, a = spectrum(t, res["probes"][i][comp], window)
        ok = (f > 0.05 * est["f0_THz"]) & (f < f_hi) if axis == "lam" else f < f_hi
        x = lam_axis(f[ok]) if axis == "lam" else f[ok]
        ax.plot(x, a[ok], lw=1.1, label=f"Sonde {i + 1}, E_{comp[1]}")
    if show_source and mode == "spectrum":
        f, s = spectrum(t, res["drive"], window)
        ok = (f > 0.05 * est["f0_THz"]) & (f < f_hi) if axis == "lam" else f < f_hi
        a0 = max(np.nanmax(np.abs(ax.lines[0].get_ydata())) if ax.lines else 1.0, 1e-300)
        x = lam_axis(f[ok]) if axis == "lam" else f[ok]
        ax.plot(x, s[ok] / s.max() * a0, "k--", lw=0.9, label="Quelle |G′(f)| (skaliert)")
    f0 = est["f0_THz"]
    ax.axvline(C0 / (f0 * 1e12) / NM if axis == "lam" else f0, color="gray", lw=0.8, ls=":", label="Referenz λ₀")
    ax.set_xlabel("Wellenlänge (nm)" if axis == "lam" else "Frequenz (THz)")
    ax.set_ylabel("|E(f)| / |G′(f)|" if mode == "transfer" else "|E(f)| (V/m · fs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def frame_values(frames, k, quantity):
    """The quantity of frame k as an array (2D: (npts,), 3D: (ny, nx)); |E| from the components."""
    comps = {c: np.asarray(frames[c][k], dtype=float) for c in ("ex", "ey", "ez") if c in frames}
    if quantity == "abs":
        return np.sqrt(sum(v**2 for v in comps.values()))
    return comps[quantity]


def fig_frame(frames, k, quantity="ey", meta=None, scale=None, model=None):
    """Field map of frame k (2D triangulation or 3D slice); ``scale`` the fixed colour limit (None = this frame)."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    z = frame_values(frames, k, quantity)
    limit = scale if scale else (float(np.nanmax(np.abs(z))) or 1.0)
    cmap, vmin = ("inferno", 0.0) if quantity == "abs" else ("RdBu_r", -limit)
    t = float(frames["times_fs"][k])
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    if "points" in frames:
        tri = mtri.Triangulation(frames["points"][:, 0], frames["points"][:, 1], frames["simplices"])
        h = ax.tripcolor(tri, z, shading="gouraud", cmap=cmap, vmin=vmin, vmax=limit)
        ax.set_ylabel("y (nm)")
    else:
        gx, gy = frames["gx"], frames["gy"]
        h = ax.pcolormesh(gx, gy, z, shading="auto", cmap=cmap, vmin=vmin, vmax=limit)
        ax.set_ylabel("y (nm), Schnitt z = %g nm" % float(meta.get("frame_z_nm", 0.0)) if meta else "y (nm)")
    fig.colorbar(h, ax=ax, label=f"{QUANTITIES[quantity]} (V/m)")
    if model is not None and model["dim"] == "2d":
        fw._outline(ax, model)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_title(f"t = {t:.2f} fs", fontsize=9)
    fig.tight_layout()
    return fig


def fig_source(m):
    """The drive g'(t) and its spectrum |G'(f)| with the resolved band (f_max) and the carrier: shows what the mesh and the time step must resolve."""
    import matplotlib.pyplot as plt

    s, est = m["source"], estimate(m)
    omega = 2 * math.pi * C0 / (float(s["lambda_nm"]) * NM)
    tau, t0 = float(s["width_fs"]) * FS, est["t0_fs"] * FS
    dt = max(est["dt_fs"], 1e-3) * FS
    t_end = max(est["t_end_fs"], est["t0_fs"] + 6 * float(s["width_fs"])) * FS
    n = int(min(max(math.ceil(t_end / dt), 64), 40000))
    t = np.arange(n + 1) * (t_end / n)
    g = signal_derivative(s["kind"], omega, t0, tau, t)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    a1.plot(t / FS, g / max(np.max(np.abs(g)), 1e-300), "k-", lw=0.9)
    a1.set_xlabel("t (fs)")
    a1.set_ylabel("g′(t) (normiert)")
    a1.grid(alpha=0.3)
    f, sp = spectrum(t / FS, g, "none", pad=8)
    ok = f < 2.0 * est["f_max_THz"]
    a2.plot(f[ok], sp[ok] / max(sp.max(), 1e-300), "k-", lw=1.0)
    a2.axvline(est["f0_THz"], color="gray", ls=":", lw=1.0, label="Referenz λ₀")
    a2.axvline(est["f_max_THz"], color="tab:red", ls="--", lw=1.0, label="höchste Frequenz (Netz, Δt)")
    a2.set_xlabel("Frequenz (THz)")
    a2.set_ylabel("|G′(f)| (normiert)")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    return fig
