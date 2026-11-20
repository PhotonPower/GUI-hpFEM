"""Emitters (hp-FEM 0.5) without hpfem and without Streamlit: models, job documents, reading of results and figures.

Two kinds of emitters are handled here:

* **planar** (``hpfem.dataports.planar_emitter_response``, semi-analytic, no mesh): a dipole at several heights above a stack of layers.
  Model: ``default_planar()``; job ``planar_job``; worker ``fem_planar_worker.py``; results ``results.json`` with the Purcell factors
  (A_tot = diag(F_par, F_par, F_perp)), the radiated power, the detector regions (numerical apertures) and the far field per half-space.
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


# ===================================================================================================================== planar
def default_planar():
    """Water above 50 nm of gold on glass (a classic: quenching and plasmon coupling), dipole heights 20 ... 400 nm."""
    return {
        "name": "Dipol über Au-Film",
        "materials": {"Wasser": {"type": "library", "name": "water"}, "Au": {"type": "library", "name": "Au"},
                      "Glas": {"type": "library", "name": "SiO2"}},
        "cover": "Wasser", "layers": [{"material": "Au", "thickness_nm": 50.0}], "substrate": "Glas",
        "sweep": {"mode": "single", "start": 600.0, "stop": 700.0, "n": 5},
        "heights": {"start": 20.0, "stop": 400.0, "n": 12},
        "detectors": [{"name": "Objektiv NA 1,0 (oben)", "side": "cover", "na_min": 0.0, "na_max": 1.0},
                      {"name": "Objektiv NA 1,4 (Glas)", "side": "substrate", "na_min": 0.0, "na_max": 1.4}],
        "theta_points": 96, "phi_points": 16,
    }


def planar_presets():
    glass = {"type": "library", "name": "SiO2"}
    air = {"type": "library", "name": "air"}
    base = default_planar()
    free = {"name": "Dipol über Glas (Luft/Glas)", "materials": {"Luft": air, "Glas": glass}, "cover": "Luft", "layers": [], "substrate": "Glas",
            "sweep": {"mode": "single", "start": 600.0, "stop": 700.0, "n": 5}, "heights": {"start": 10.0, "stop": 600.0, "n": 15},
            "detectors": [{"name": "Luft NA 0,9", "side": "cover", "na_min": 0.0, "na_max": 0.9},
                          {"name": "Glas", "side": "substrate", "na_min": 0.0, "na_max": 1.45}], "theta_points": 96, "phi_points": 16}
    ag = {"name": "Dipol über Ag-Film mit Ti-Haftschicht", "materials": {"Wasser": {"type": "library", "name": "water"},
                                                                      "Ag": {"type": "library", "name": "Ag"}, "Ti": {"type": "library", "name": "Ti"},
                                                                      "Glas": glass},
          "cover": "Wasser", "layers": [{"material": "Ag", "thickness_nm": 45.0}, {"material": "Ti", "thickness_nm": 2.0}], "substrate": "Glas",
          "sweep": {"mode": "single", "start": 550.0, "stop": 700.0, "n": 5}, "heights": {"start": 15.0, "stop": 300.0, "n": 12},
          "detectors": base["detectors"], "theta_points": 96, "phi_points": 16}
    return {base["name"]: base, free["name"]: free, ag["name"]: ag}


def sweep_list(sw):
    if sw["mode"] == "single":
        return [float(sw["start"])]
    return [float(round(v, 6)) for v in np.linspace(float(sw["start"]), float(sw["stop"]), max(int(sw["n"]), 1))]


def heights_list(h):
    return [float(round(v, 6)) for v in np.linspace(float(h["start"]), float(h["stop"]), max(int(h["n"]), 1))]


def planar_validate(m):
    """List of (level, text) with level "error" or "warn"."""
    out = []
    mats = m["materials"]
    for name in [m["cover"], m["substrate"]] + [ly["material"] for ly in m["layers"]]:
        if name not in mats:
            out.append(("error", f"Material {name!r} fehlt."))
    if out:
        return out
    hs = heights_list(m["heights"])
    if hs[0] <= 0:
        out.append(("error", "Die Höhen müssen größer als 0 sein (Abstand des Dipols von der Oberkante des Stapels)."))
    if len(hs) > 1 and not all(b > a for a, b in zip(hs, hs[1:])):
        out.append(("error", "Die Höhen müssen streng aufsteigen (Ende > Anfang)."))
    for ly in m["layers"]:
        if float(ly["thickness_nm"]) <= 0:
            out.append(("error", f"Schicht {ly['material']}: Dicke muss positiv sein."))
    for lam in (min(sweep_list(m["sweep"])), max(sweep_list(m["sweep"]))):
        for name in {m["cover"], m["substrate"]} | {ly["material"] for ly in m["layers"]}:
            try:
                e = fm.eps_at(mats[name], lam)
                if name == m["cover"] and (abs(e.imag) > 1e-9 or e.real <= 0):
                    out.append(("error", f"Das Einfallsmedium {name} muss bei {lam:g} nm verlustfrei sein (Wirt des Emitters)."))
                if fm.is_crystal(mats[name]):
                    out.append(("error", f"{name}: Kristalle sind bei planaren Emittern nicht möglich (isotrope Schichten)."))
            except Exception as exc:
                out.append(("error", str(exc)))
    if not m["detectors"]:
        out.append(("error", "Mindestens ein Detektorbereich nötig."))
    for side in ("cover", "substrate"):
        spans = sorted((d["na_min"], d["na_max"], d["name"]) for d in m["detectors"] if d["side"] == side)
        for (_, b0, n0), (a1, _, n1) in zip(spans, spans[1:]):
            if a1 < b0 - 1e-12:
                out.append(("error", f"Die Detektorbereiche {n0!r} und {n1!r} überlappen."))
    for d in m["detectors"]:
        if not d["na_min"] < d["na_max"]:
            out.append(("error", f"Detektor {d['name']!r}: NA min muss kleiner als NA max sein."))
    return list(dict.fromkeys(out))


def planar_job(m):
    """The job document of fem_planar_worker.py."""
    used = {m["cover"], m["substrate"]} | {ly["material"] for ly in m["layers"]}
    return {"planar": {"name": m["name"], "wavelengths_nm": sweep_list(m["sweep"]), "heights_nm": heights_list(m["heights"]),
                       "cover": m["cover"], "substrate": m["substrate"], "layers": copy.deepcopy(m["layers"]),
                       "materials": {k: v for k, v in m["materials"].items() if k in used},
                       "detectors": copy.deepcopy(m["detectors"]), "theta_points": int(m.get("theta_points", 96)),
                       "phi_points": int(m.get("phi_points", 16))}}


def planar_dataframe(points, lam_index=0):
    """Table for one wavelength: heights and the response (F_par, F_perp, F_iso, radiated fractions, detectors)."""
    ok = [p for p in points if "error" not in p]
    if not ok:
        return pd.DataFrame()
    p = ok[min(lam_index, len(ok) - 1)]
    df = pd.DataFrame({"λ (nm)": p["lam_nm"], "Höhe (nm)": p["heights_nm"], "F_∥": p["F_par"], "F_⊥": p["F_perp"]})
    df["F_iso"] = (2 * df["F_∥"] + df["F_⊥"]) / 3
    df["rad_∥ / F_∥"] = np.divide(p["rad_par"], p["F_par"])
    df["rad_⊥ / F_⊥"] = np.divide(p["rad_perp"], p["F_perp"])
    for k, d in enumerate(p["detectors"]):
        df[f"{d['name']} (∥)"] = np.array(p["det_par"])[:, k] / np.array(p["F_par"])
        df[f"{d['name']} (⊥)"] = np.array(p["det_perp"])[:, k] / np.array(p["F_perp"])
    return df


def planar_fractions(p):
    """Power fractions of the total emission per height for the isotropic dipole: dict of arrays cover, substrate, absorbed_or_guided, plus
    the detector shares (each of the total emission)."""
    fp_, fz = np.array(p["F_par"]), np.array(p["F_perp"])
    tot = (2 * fp_ + fz) / 3
    out = {}
    far = p.get("far", {})
    for side in ("cover", "substrate"):
        if side in far:
            out[side] = (2 * np.array(far[side]["power_par"]) + np.array(far[side]["power_perp"])) / 3 / tot
        else:
            out[side] = np.zeros_like(tot)
    out["lost"] = np.clip(1.0 - out["cover"] - out["substrate"], 0.0, None)
    for k, d in enumerate(p["detectors"]):
        out[d["name"]] = (2 * np.array(p["det_par"])[:, k] + np.array(p["det_perp"])[:, k]) / 3 / tot
    return out


def _empty(text="keine Ergebnisse", size=(7, 3)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, text, ha="center")
    ax.axis("off")
    return fig


def fig_planar_purcell(points):
    """Purcell factors versus height (one line per wavelength): F_par, F_perp, F_iso (log axis)."""
    import matplotlib.pyplot as plt
    ok = [p for p in points if "error" not in p]
    if not ok:
        return _empty()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, p in enumerate(ok):
        c = cmap(i / max(len(ok) - 1, 1))
        h = np.array(p["heights_nm"])
        fpar, fz = np.array(p["F_par"]), np.array(p["F_perp"])
        for ax, y in zip(axs, (fpar, fz, (2 * fpar + fz) / 3)):
            ax.plot(h, y, "o-", color=c, ms=3, label=f"{p['lam_nm']:g} nm")
    for ax, t in zip(axs, ("F_∥ (Dipol parallel zur Oberfläche)", "F_⊥ (senkrecht)", "F_iso (isotrop)")):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Höhe über der Oberfläche (nm)")
        ax.set_yscale("log")
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("Purcell-Faktor (frei = 1)")
    if len(ok) > 1:
        axs[-1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_planar_channels(p):
    """Where the power of the isotropic dipole goes, versus height: into the cover, into the substrate, lost (absorbed or guided);
    below: the shares collected by the detector regions."""
    import matplotlib.pyplot as plt
    fr = planar_fractions(p)
    h = np.array(p["heights_nm"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    ax = axs[0]
    ax.stackplot(h, fr["cover"], fr["substrate"], fr["lost"], labels=["ins Einfallsmedium (oben)", "ins Substrat (unten)", "absorbiert oder geführt"],
                 colors=["#1f77b4", "#2ca02c", "#d62728"], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title(f"Aufteilung der Leistung bei {p['lam_nm']:g} nm (isotroper Dipol)", fontsize=10)
    ax.set_ylabel("Anteil der Gesamtleistung")
    ax.legend(fontsize=8, loc="center right")
    ax = axs[1]
    for d in p["detectors"]:
        ax.plot(h, fr[d["name"]], "o-", ms=3, label=d["name"])
    ax.set_title("In die Detektorbereiche (Anteil der Gesamtleistung)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    for a in axs:
        a.set_xlabel("Höhe über der Oberfläche (nm)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_planar_pattern(p, height_index, orientation="isotropic"):
    """Polar plot of dP/dOmega (in units of the free dipole's power per sr) in the cover (above) and the substrate (below) at one height."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 5.6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    k = int(np.clip(height_index, 0, len(p["heights_nm"]) - 1))
    far = p.get("far", {})
    for side, sign, color in (("cover", 1, "#1f77b4"), ("substrate", -1, "#2ca02c")):
        if side not in far:
            continue
        f = far[side]
        th = np.array(f["theta"])
        par, perp = np.array(f["par"])[k], np.array(f["perp"])[k]
        y = {"isotropic": (2 * par + perp) / 3, "parallel": par, "perp": perp}[orientation]
        # the substrate half-space is drawn below: angle measured from the downward normal, mirrored to the left half for the lower half-plane
        ang = th if side == "cover" else np.pi - th
        ax.plot(ang, y, color=color, label="oben (Einfallsmedium)" if side == "cover" else "unten (Substrat)")
        ax.plot(-ang, y, color=color)
    ax.set_title(f"dP/dΩ bei {p['lam_nm']:g} nm, Höhe {p['heights_nm'][k]:g} nm", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.9, 0.0))
    fig.tight_layout()
    return fig


def planar_table_json(folder, i):
    """The BindOptiX envelope (EmitterResponseTable) of wavelength i as text, or None."""
    from pathlib import Path
    p = Path(folder) / f"table_{i}.json"
    return p.read_text(encoding="utf-8") if p.is_file() else None


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
