"""Emitters (hp-FEM 0.5) without hpfem and without Streamlit: models, job documents, reading of results and figures.

The emitter of a periodic cell is handled here:

* **grating** (``hpfem.grating.emit`` / ``dipole_emission`` / ``emission_pattern`` / ``emission_cone``, M17): a Gaussian dipole in the
  cell of a periodic structure. The emitter belongs to the model of the periodic mode (``model["emitter"]``, see ``emitter_defaults``);
  the worker is ``fem_worker.py`` with the task ``emitter`` (``fem_emit_worker.py``). Stage A: Bloch array at one (kx, beta); stage B: single
  dipole by array scanning (Purcell factor, fractions up / down / non-radiated, power into collection apertures); stage C: dP/dOmega.

Units: nm in the models, SI in the solver; angles in degrees in the models. Conventions as everywhere: time factor exp(-i omega t),
loss means Im eps > 0; free dipole in the host medium = 1.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

import fem_materials as fm

MOMENTS = {"isotropic": "isotrop (Mittel über x, y, z)", "x": "x (quer zu den Linien, in der Ebene)", "y": "y (Schichtnormale)",
           "z": "z (längs der Linien)"}
STAGES = {"B": "B: einzelner Dipol (Array-Scan): Purcell-Faktor, Anteile, Aperturen",
          "A": "A: Bloch-Array bei festem (kx, β)", "C": "C: Abstrahlmuster dP/dΩ"}
C0 = 299792458.0


def _empty(text="keine Ergebnisse", size=(7, 3)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, text, ha="center")
    ax.axis("off")
    return fig


# ===================================================================================================================== grating
def emitter_defaults():
    """Emitter of a periodic cell (model["emitter"]); positions in nm of the model (x along the period, y vertical, y = 0 the top of the stack)."""
    return {"x_nm": 200.0, "y_nm": 300.0, "sigma_nm": 8.0, "moment": "isotropic", "stage": "B",
            "scan": {"nodes": 6, "kx_nodes": 4, "depth": 0.5, "angle_nodes": [16, 32], "symmetric": False, "workers": 1, "beta_max_over_k0": None,
                     "guided": "auto"},
            "apertures": [0.5, 0.9], "aperture_side": "up", "kx_over_k0": 0.0, "beta_over_k0": 0.0,
            "directions": {"theta_deg": [0.0, 15.0, 30.0, 45.0, 60.0, 75.0], "phi_deg": [0.0], "sides": ["up"]}, "calibrate": True}


def emitter_complete(em):
    d = emitter_defaults()
    out = copy.deepcopy(em)
    for k, v in d.items():
        if k not in out:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                out[k].setdefault(k2, copy.deepcopy(v2))
    return out


def emitter_validate(model, em, lam_nm=None):
    """Checks of the emitter position in the periodic model: inside the cell, one lossless medium within 6 sigma. List of (level, text)."""
    out = []
    P = float(model["period_nm"])
    x, y, s = float(em["x_nm"]), float(em["y_nm"]), float(em["sigma_nm"])
    if s <= 0:
        out.append(("error", "σ (Breite der Gauß-Quelle) muss positiv sein."))
        return out
    if not 6 * s <= x <= P - 6 * s:
        out.append(("error", f"Die Gauß-Quelle muss 6σ = {6 * s:g} nm innerhalb der Zelle liegen ({6 * s:g} ≤ x ≤ {P - 6 * s:g} nm)."))
    if model.get("domain", {}).get("lateral") == "pml":
        out.append(("error", "Emitter rechnet nur periodische Zellen (Ränder „periodisch“), nicht isolierte Strukturen."))
    try:
        import fem_geometry as fg
        lay = fg.layout(model)
        if not lay["y_sub_bottom"] < y < lay["y_cover_top"]:
            out.append(("error", f"y = {y:g} nm liegt außerhalb des physikalischen Gebiets ({lay['y_sub_bottom']:g} … {lay['y_cover_top']:g} nm)."))
        else:
            ang = np.linspace(0, 2 * np.pi, 33)[:-1]
            X = [x] + [x + 6 * s * np.cos(a) for a in ang]
            Y = [y] + [y + 6 * s * np.sin(a) for a in ang]
            idx = fg.material_index_map(model, np.array(X), np.array(Y))
            if len(set(np.asarray(idx).ravel().tolist())) != 1:
                out.append(("error", f"Im Umkreis von 6σ = {6 * s:g} nm um den Emitter liegt mehr als ein Material (Grenzfläche, Form): Abstand vergrößern "
                                     "oder σ verkleinern. Die Gauß-Quelle muss in einem verlustfreien Medium liegen."))
            else:
                name = list(model["materials"])[int(np.asarray(idx).ravel()[0])]
                if lam_nm is not None:
                    try:
                        e = fm.eps_at(model["materials"][name], lam_nm)
                        if abs(e.imag) > 1e-9:
                            out.append(("error", f"Das Material am Emitter ({name}) ist bei {lam_nm:g} nm verlustbehaftet."))
                    except Exception:
                        pass
    except Exception as exc:  # layout of an incomplete model
        out.append(("error", f"Emitterposition nicht prüfbar: {exc}"))
    sw = model.get("sweep", {})
    if sw.get("mode") in ("theta", "phi"):
        out.append(("warn", "Der Emitter hat keinen Einfallswinkel: gerechnet wird bei der Wellenlänge der Beleuchtung (Durchlauf über θ oder φ wird ignoriert)."))
    return out


def emitter_wavelengths(model):
    sw, inc = model["sweep"], model["incidence"]
    if sw["mode"] == "wavelength":
        return [float(round(v, 6)) for v in np.linspace(sw["start"], sw["stop"], max(int(sw["n"]), 1))]
    return [float(inc["wavelength_nm"])]


def emission_points(res):
    """Points of an emitter result without errors."""
    return [p for p in res.get("points", []) if "error" not in p]


def emitter_dataframe(res):
    """Table of an emitter result (stages A and B: one row per wavelength)."""
    pts = emission_points(res)
    stage = (res.get("emitter") or {}).get("stage", "B")
    rows = []
    for p in pts:
        row = {"λ (nm)": p["wavelength"]}
        if stage == "B":
            for k, v in p["purcell_by_orientation"].items():
                row[f"F_P {k}"] = v
            fr = p.get("fractions")
            if fr:
                row.update({"oben": fr["up"], "unten": fr["down"], "nicht abgestrahlt": fr["nonradiated"]})
            for a in p.get("aperture", []):
                row[f"NA {a['NA']:g} ({a['side']})"] = a["fraction"]
            row["Zellen-Lösungen"] = p["samples"]
        elif stage == "A":
            row.update({"kx/k0": p["kx"] / (2 * math.pi / (p["wavelength"] * 1e-9)), "P_Zelle (W/m)": p["P_cell"], "oben": p["up"], "unten": p["down"],
                        "absorbiert": p["absorbed"], "geführt": p["guided"]})
        else:
            for a in p.get("aperture", []):
                row[f"NA {a['NA']:g} ({a['side']})"] = a["per_P_bulk"]
        row["Zeit (s)"] = p.get("seconds")
        rows.append(row)
    return pd.DataFrame(rows)


def fig_emit_purcell(res):
    import matplotlib.pyplot as plt
    pts = emission_points(res)
    if not pts or "purcell_by_orientation" not in pts[0]:
        return _empty()
    lam = np.array([p["wavelength"] for p in pts])
    keys = list(pts[0]["purcell_by_orientation"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k in keys:
        y = np.array([p["purcell_by_orientation"][k] for p in pts])
        ax.plot(lam, y, "o-", ms=4, lw=2.0 if k == (res.get("emitter") or {}).get("moment") else 1.0, label=k)
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("Purcell-Faktor F_P")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(title="Orientierung", fontsize=8)
    fig.tight_layout()
    return fig


def fig_emit_fractions(res):
    import matplotlib.pyplot as plt
    pts = [p for p in emission_points(res) if p.get("fractions")]
    if not pts:
        return _empty("keine Kanalanteile (ohne „channels“ gerechnet oder Stufe A/C)")
    lam = np.array([p["wavelength"] for p in pts])
    up = np.array([p["fractions"]["up"] for p in pts])
    down = np.array([p["fractions"]["down"] for p in pts])
    non = np.array([p["fractions"]["nonradiated"] for p in pts])
    lossy = bool(pts[0].get("lossy"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    if len(lam) > 1:
        ax.stackplot(lam, up, down, np.clip(non, 0, None), labels=["nach oben", "nach unten", "absorbiert" if lossy else "geführt"],
                     colors=["#1f77b4", "#2ca02c", "#d62728"], alpha=0.85)
        ax.set_xlabel("Wellenlänge (nm)")
    else:
        ax.bar(["oben", "unten", "absorbiert" if lossy else "geführt"], [up[0], down[0], non[0]], color=["#1f77b4", "#2ca02c", "#d62728"])
    ax.set_ylabel("Anteil der abgegebenen Leistung")
    ax.set_title("Wohin die Leistung geht", fontsize=10)
    if len(lam) > 1:
        ax.legend(fontsize=8, loc="center right")
    ax = axs[1]
    names = sorted({f"NA {a['NA']:g} ({a['side']})" for p in pts for a in p.get("aperture", [])})
    for nme in names:
        y = [next((a["fraction"] for a in p.get("aperture", []) if f"NA {a['NA']:g} ({a['side']})" == nme), np.nan) for p in pts]
        ax.plot(lam, y, "o-", ms=4, label=nme)
    ax.set_title("In Sammelaperturen", fontsize=10)
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("Anteil der abgegebenen Leistung")
    ax.grid(alpha=0.3)
    if names:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_emit_orders(res, index=0):
    """Stage A: powers of the radiated orders."""
    import matplotlib.pyplot as plt
    pts = [p for p in emission_points(res) if "orders_up" in p]
    if not pts:
        return _empty("nur für Stufe A")
    p = pts[min(index, len(pts) - 1)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in ((axs[0], "orders_up", "nach oben"), (axs[1], "orders_down", "nach unten")):
        if p[key]:
            ms, ps = zip(*p[key])
            ax.bar([str(m) for m in ms], ps, color="#1f77b4" if key == "orders_up" else "#2ca02c")
        ax.set_title(f"Ordnungen {title} (λ = {p['wavelength']:g} nm)", fontsize=10)
        ax.set_xlabel("Ordnung m")
        ax.set_ylabel("Leistung (W/m je Einheit β)")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_emit_pattern(res, index=0, log=False):
    """Stage C: dP/dOmega (per P_bulk) versus theta for each side, mean over the phi samples and each phi."""
    import matplotlib.pyplot as plt
    pts = [p for p in emission_points(res) if "dP_dOmega" in p]
    if not pts:
        return _empty("nur für Stufe C")
    p = pts[min(index, len(pts) - 1)]
    th = np.array(p["theta_deg"])
    sides = p["sides"]
    fig, axs = plt.subplots(1, len(sides), figsize=(5.5 * len(sides), 3.8), squeeze=False)
    for ax, side in zip(axs[0], sides):
        g = np.array(p["dP_dOmega"][side]) / (p["P_bulk"] / (4 * math.pi))
        for j, ph in enumerate(p["phi_deg"]):
            ax.plot(th, g[:, j], "o-", ms=3, label=f"φ = {ph:g}°")
        ax.set_title(f"{'oben' if side == 'up' else 'unten'} (λ = {p['wavelength']:g} nm)", fontsize=10)
        ax.set_xlabel("Polarwinkel θ (°)")
        ax.set_ylabel("Gewinn dP/dΩ / (P_bulk/4π)")
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def cost_text(cost):
    """One line about the cost estimate of the emitter task."""
    if not cost:
        return ""
    pw = cost.get("per_wavelength", {})
    s = (f"{cost.get('solves', 0)} Lösungen insgesamt ({pw.get('samples', 0)} Zellenprobleme + {pw.get('plane_wave_solves', 0)} ebene Wellen je Wellenlänge, "
         f"{cost.get('wavelengths', 1)} Wellenlänge(n)); ein Faktorisierungsschritt: {cost.get('dofs', 0):,} Freiheitsgrade, {cost.get('memory', '')}".replace(",", "."))
    if cost.get("seconds"):
        m, sec = divmod(int(cost["seconds"]), 60)
        s += f"; geschätzte Rechenzeit {m} min {sec} s"
    return s
