"""Waveguides and ports in 3D (hp-FEM 0.5): models, job, checks and figures. Pure Python (no hpfem needed); worker: fem_ports3d_worker.py, user interface:
fem_threed_app.py (task ``ports`` of the mode "3D-Strukturen").

Devices (z along the guide, x across, y up; PEC walls on the four side faces, modal ports on the two end faces z = 0 and z = L, as in the library example
``directional_coupler_3d``):

* ``straight``: one rectangular core (width x height) in a cladding; one port per end face with ``num_modes`` modes; reference: the fundamental mode of the
  same cross-section from the 2D mode solver (``PropagatingMode``): n_eff, and the phase of the transmission against beta L.
* ``coupler``: two parallel cores with a gap (directional coupler); ``ports = "half"`` splits each end face into two ports (core A / core B, one mode each),
  ``"full"`` uses one two-mode port per face (even and odd supermode); reference: coupled-mode theory P_cross = sin^2(kappa L), kappa = (beta_e - beta_o) / 2 from the
  supermodes of the cross-section.
* ``file``: a Gmsh mesh of the user (physical volumes = materials, physical surfaces = PEC walls and ports); no reference.
* ``ring``: the add-drop ring resonator of the library (examples/ring_resonator_3d): ring and two bus waveguides, Gmsh mesh (fem_ring3d_mesh.py), four single-mode ports,
  launch at the input port, through / drop / add / reflection over the wavelength, the resonance as the common complex pole of through and drop (``fit_pole``), the
  isolated ring as an axisymmetric cavity (``AxisymmetricCavity``) as the reference and the free spectral range from the group index of the bus (FSR = lambda^2 / (n_g L)).

Structured meshes (hpfem.box, tetrahedra, cells aligned with the core edges) so that no Gmsh is needed for the first two.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

import fem_materials as fm

DEVICES = {"straight": "Streifenwellenleiter", "coupler": "Richtkoppler (zwei Kerne)", "ring": "Ringresonator (Add-Drop, Gmsh)", "file": "Eigene Gmsh-Datei (.msh)"}
PORT_STYLES = {"half": "vier Halbflächen-Ports (Kern A, Kern B je Stirnfläche, eine Mode)", "full": "ein Port je Stirnfläche mit zwei Moden (gerade und ungerade Supermode)"}
SCRIPT = "fem_ports3d_worker.py"
TAG_CLAD, TAG_A, TAG_B = 1, 2, 3
PORT_TAGS = (11, 12, 13, 14)
GB = 1e9


def default_model():
    return {"name": "Streifenwellenleiter Si/SiO2, λ = 1550 nm", "device": "straight",
            "materials": {"Kern": {"type": "index", "n": 3.48, "k": 0.0}, "Mantel": {"type": "index", "n": 1.44, "k": 0.0}}, "core": "Kern", "clad": "Mantel",
            "width_nm": 450.0, "height_nm": 220.0, "gap_nm": 200.0, "length_nm": 1200.0, "margin_x_nm": 600.0, "margin_y_nm": 450.0, "cell_nm": 150.0, "cell_z_nm": 0.0,
            "order": 1, "ports": "full", "num_modes": 1, "excite_port": 0, "excite_mode": 0,
            "sweep": {"start": 1550.0, "stop": 1550.0, "count": 1}, "field": True, "max_gb": 16.0,
            "file": {"name": "", "scale": "nm", "tag_material": {}, "pec_tags": [], "ports": [], "tags": {}}, "ring": ring_defaults()}


def ring_defaults():
    """Parameters of the ring device (the cross-section, the gap and the materials are those of the model). Defaults: the cheap test case of the library (R = 1.5 µm, 150 nm
    cells, p = 1): it exercises the whole chain but resolves nothing of the resonator."""
    return {"radius_nm": 1500.0, "margin_x_nm": 1000.0, "margin_y_nm": 750.0, "margin_z_nm": 605.0, "cell_core_nm": 150.0, "cell_far_nm": 400.0, "grade_nm": 600.0, "half": True,
            "order_core": 1, "order_far": 1, "reach_nm": 250.0, "backend": "auto", "scan": "manual", "target_nm": 1550.0, "search_nm": 4.0,
            "reference": True, "ref_p": 2, "ref_cell_nm": 100.0, "field_res": False}


def presets():
    p = {}
    m = default_model()
    p[m["name"]] = m
    m = default_model()
    m.update(name="Richtkoppler Si/SiO2 (Halbflächen-Ports, klein)", device="coupler", ports="half", gap_nm=200.0, length_nm=1000.0, cell_nm=150.0)
    p[m["name"]] = m
    m = default_model()
    m.update(name="Richtkoppler Si/SiO2 (Supermoden-Ports)", device="coupler", ports="full", num_modes=2, gap_nm=200.0, length_nm=1000.0, cell_nm=150.0)
    p[m["name"]] = m
    m = default_model()
    m.update(name="Ringresonator Add-Drop (Test: R = 1,5 µm, p = 1, nur Ablauf)", device="ring", gap_nm=200.0, sweep={"start": 1530.0, "stop": 1570.0, "count": 3})
    p[m["name"]] = m
    m = default_model()
    m.update(name="Ringresonator Add-Drop (mittel: R = 2 µm, 150-nm-Zellen, p = 2/1)", device="ring", gap_nm=200.0, max_gb=64.0)
    m["ring"].update(radius_nm=2000.0, order_core=2, order_far=1, scan="auto", target_nm=1557.0)
    p[m["name"]] = m
    m = default_model()
    m.update(name="Ringresonator Add-Drop (Bibliotheksbeispiel: R = 2 µm, 100-nm-Zellen, p = 3/2; Server)", device="ring", gap_nm=200.0, max_gb=400.0)
    m["ring"].update(radius_nm=2000.0, cell_core_nm=100.0, cell_far_nm=300.0, order_core=3, order_far=2, scan="auto", target_nm=1557.0, ref_p=4, ref_cell_nm=50.0)
    p[m["name"]] = m
    return p


def wavelengths(m):
    if m["device"] == "ring" and m["ring"]["scan"] == "auto":
        return []
    s = m["sweep"]
    return [float(v) for v in np.linspace(s["start"], s["stop"], int(s["count"]))] if int(s["count"]) > 1 else [float(s["start"])]


def eps_pair(m, name, lam):
    e = fm.eps_at(m["materials"][name], lam)
    return [float(e.real), float(e.imag)]


def grid_lines(edges, cell):
    """Grid lines through all edges with about `cell` spacing in between (same rule as the worker)."""
    lines = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(1, round((b - a) / cell))
        lines.extend(a + (b - a) * (i + 1) / n for i in range(n))
    return np.array(lines)


def x_edges(m):
    w, g, mx = m["width_nm"], m["gap_nm"], m["margin_x_nm"]
    if m["device"] == "straight":
        return [-w / 2 - mx, -w / 2, w / 2, w / 2 + mx]
    a0, b1 = -g / 2 - w, g / 2 + w
    return [a0 - mx, a0, -g / 2, g / 2, b1, b1 + mx]


def y_edges(m):
    return [-m["margin_y_nm"], 0.0, m["height_nm"], m["height_nm"] + m["margin_y_nm"]]


def cell_count(m):
    """Approximate number of hexahedral cells of the structured mesh (each is split into tetrahedra) for a quick size hint."""
    if m["device"] in ("file", "ring"):
        return None
    cell = m["cell_nm"]
    nx, ny = len(grid_lines(x_edges(m), cell)) - 1, len(grid_lines(y_edges(m), cell)) - 1
    nz = max(1, round(m["length_nm"] / (m["cell_z_nm"] or cell)))
    return nx * ny * nz


def validate(m):
    out = []
    d = m["device"]
    if d == "file":
        f = m["file"]
        if not f.get("name"):
            out.append(("error", "Bitte eine .msh-Datei hochladen (Gmsh 4.1 ASCII)."))
        elif not f.get("tags"):
            out.append(("error", "Die Datei wurde noch nicht gelesen: „Datei lesen“ im Reiter 2."))
        elif not f["ports"]:
            out.append(("error", "Mindestens ein Port (Flächenmarke) wird gebraucht."))
        if not f.get("pec_tags"):
            out.append(("warning", "Keine PEC-Wand gewählt: die Seitenflächen sind dann natürlich (PMC)."))
    elif d == "ring":
        out += validate_ring(m)
    else:
        if min(m["width_nm"], m["height_nm"], m["length_nm"], m["cell_nm"]) <= 0:
            out.append(("error", "Breite, Höhe, Länge und Zelle müssen positiv sein."))
        if m["cell_nm"] > 0.6 * min(m["width_nm"], m["height_nm"]) * 1.0 and m["order"] == 1:
            out.append(("warning", "Die Zelle ist größer als etwa 60 % der kleineren Kernkante: mit p = 1 ist der Kern kaum aufgelöst (Reflexion und Phasenfehler groß)."))
        if d == "coupler" and m["gap_nm"] < m["cell_nm"]:
            out.append(("warning", "Der Spalt ist kleiner als eine Zelle: er wird mit einer Zelle vernetzt (der Spalt im Modell ist dann die Zellkante)."))
        if d == "coupler" and m["ports"] == "full" and int(m["num_modes"]) < 2:
            out.append(("error", "Supermoden-Ports brauchen zwei Moden je Port."))
        if m["length_nm"] < 2 * m["cell_nm"]:
            out.append(("error", "Die Länge muss mindestens zwei Zellen umfassen."))
        n = cell_count(m)
        if n and n > 60000:
            out.append(("warning", f"Etwa {n} Würfelzellen (je sechs Tetraeder): das wird ein großes Problem, vorher Speicher schätzen."))
    try:
        for lam in (wavelengths(m) or [lam_ref(m)]):
            if d != "file":
                ec, ek = fm.eps_at(m["materials"][m["core"]], lam), fm.eps_at(m["materials"][m["clad"]], lam)
                if ec.real <= ek.real:
                    out.append(("warning", f"Bei {lam:g} nm ist Re ε des Kerns nicht größer als der des Mantels: es gibt keine geführte Mode."))
                    break
    except Exception as exc:
        out.append(("error", f"Material: {exc}"))
    if m["order"] < 1:
        out.append(("error", "Ordnung ≥ 1."))
    return out


def lam_ref(m):
    """The wavelength at which the materials are shown and checked: the first of the sweep, for the automatic ring scan the target wavelength."""
    if m["device"] == "ring" and m["ring"]["scan"] == "auto":
        return float(m["ring"]["target_nm"])
    if m["device"] == "ring":
        w = wavelengths(m)
        return float(np.mean(w))
    return wavelengths(m)[0]


def validate_ring(m):
    r = m["ring"]
    out = []
    w, h, g = m["width_nm"], m["height_nm"], m["gap_nm"]
    if min(w, h, g, r["radius_nm"], r["cell_core_nm"], r["cell_far_nm"], r["grade_nm"]) <= 0:
        out.append(("error", "Breite, Höhe, Spalt, Radius, Zellgrößen und Staffelweg müssen positiv sein."))
        return out
    if r["radius_nm"] <= w / 2 + 100:
        out.append(("error", "Der Ringradius muss deutlich größer als die halbe Breite sein."))
    if r["cell_far_nm"] < r["cell_core_nm"]:
        out.append(("error", "Die ferne Zelle muss mindestens so groß sein wie die Zelle am Kern."))
    if min(r["margin_x_nm"], r["margin_y_nm"], r["margin_z_nm"]) < 200:
        out.append(("warning", "Ein Rand unter 200 nm klemmt die geführte Mode an der PEC-Wand ein (die Felder fallen über etwa 150 nm ab; Bibliothek: 750 nm seitlich, 605 nm oben und unten)."))
    if min(int(r["order_core"]), int(r["order_far"])) < 1:
        out.append(("error", "Ordnung ≥ 1."))
    if int(r["order_core"]) == 1:
        out.append(("warning", "Mit p = 1 am Kern löst das Netz den Resonator nicht auf (Bibliothek: n_eff 7 % zu klein, Scheinresonanzen alle paar Nanometer, Rückreflexion bis 0,47 statt eines "
                               "sauberen Drop-Peaks). Die Vorlage prüft nur die Rechenkette (Energiebilanz, Portmoden)."))
    if r["cell_core_nm"] > 0.6 * min(w, h):
        out.append(("warning", "Die Zelle am Kern ist größer als 60 % der kleineren Kernkante: der Kern ist kaum aufgelöst."))
    if r["scan"] == "auto" and (is_disp_pair(m)):
        out.append(("warning", "Bei der automatischen Suche ist das Material bei der Zielwellenlänge eingefroren (die Wellenlängen stehen vorher nicht fest); in der Streuung über wenige Nanometer ist das gut."))
    if r["scan"] == "manual":
        n = int(m["sweep"]["count"])
        if n < 4:
            out.append(("warning", "Weniger als vier Wellenlängen: die Resonanz (komplexer Pol) lässt sich nicht anpassen; es gibt nur die Spektren."))
    if r["scan"] == "auto" and float(r["search_nm"]) <= 0:
        out.append(("error", "Das Suchfenster muss positiv sein."))
    if r["backend"] != "auto":
        out.append(("warning", f"Löser {r['backend']}: nur nutzbar, wenn die Bibliothek damit gebaut ist (das Wheel enthält nur SparseLU); sonst bricht der Worker ab."))
    if int(r["order_core"]) >= 3 or r["cell_core_nm"] <= 100:
        out.append(("warning", "Das Produktionsnetz hat 0,4 bis 1,7 Millionen Unbekannte und braucht einen direkten Löser mit sehr viel Speicher (Bibliothek: MUMPS auf einem Server); "
                               "mit SparseLU sind Stunden und hunderte GB zu erwarten. Die Schätzung steht nach „Netz prüfen“."))
    return out


def is_disp_pair(m):
    return any(fm.is_dispersive(m["materials"][k]) for k in (m["core"], m["clad"]))


def job(m, max_gb=None):
    j = {"task": "ports", "name": m["name"], "device": m["device"], "order": int(m["order"]), "max_gb": float(max_gb if max_gb is not None else m["max_gb"]),
         "wavelengths_nm": wavelengths(m), "field": bool(m["field"]), "excite_port": int(m["excite_port"]), "excite_mode": int(m["excite_mode"])}
    if m["device"] == "file":
        f = m["file"]
        j.update(path=f.get("path", ""), scale=_scale(f["scale"]),
                 pec_tags=[int(t) for t in f["pec_tags"]], ports=[dict(tag=int(p["tag"]), modes=int(p["modes"])) for p in f["ports"]],
                 tag_eps={str(t): [eps_pair(m, nm, lam) for lam in j["wavelengths_nm"]] for t, nm in f["tag_material"].items() if nm in m["materials"]})
        return j
    if m["device"] == "ring":
        return ring_job(m, j)
    lams = j["wavelengths_nm"]
    j.update(width_nm=float(m["width_nm"]), height_nm=float(m["height_nm"]), gap_nm=float(m["gap_nm"]), length_nm=float(m["length_nm"]),
             margin_x_nm=float(m["margin_x_nm"]), margin_y_nm=float(m["margin_y_nm"]), cell_nm=float(m["cell_nm"]), cell_z_nm=float(m["cell_z_nm"]),
             style=m["ports"], num_modes=int(m["num_modes"]) if m["device"] == "straight" or m["ports"] == "full" else 1,
             eps_core=[eps_pair(m, m["core"], lam) for lam in lams], eps_clad=[eps_pair(m, m["clad"], lam) for lam in lams])
    return j


NG_STEP_NM = 5.0


def ring_job(m, j):
    """Job of the ring device: geometry [nm], orders, scan mode, the materials (per wavelength of a manual scan; frozen at the target wavelength in the automatic scan), the
    reference (axisymmetric cavity) and the group-index triple for the free spectral range."""
    r = m["ring"]
    auto = r["scan"] == "auto"
    lams = j["wavelengths_nm"]
    target = lam_ref(m)
    j.update(width_nm=float(m["width_nm"]), height_nm=float(m["height_nm"]), gap_nm=float(m["gap_nm"]), radius_nm=float(r["radius_nm"]), margin_x_nm=float(r["margin_x_nm"]),
             margin_y_nm=float(r["margin_y_nm"]), margin_z_nm=float(r["margin_z_nm"]), cell_core_nm=float(r["cell_core_nm"]), cell_far_nm=float(r["cell_far_nm"]),
             grade_nm=float(r["grade_nm"]), half=bool(r["half"]), order_core=int(r["order_core"]), order_far=int(r["order_far"]), reach_nm=float(r["reach_nm"]),
             backend=str(r["backend"]), auto=auto, target_nm=float(target), search_nm=float(r["search_nm"]), frozen=auto,
             eps_core=[eps_pair(m, m["core"], lam) for lam in (lams or [target])], eps_clad=[eps_pair(m, m["clad"], lam) for lam in (lams or [target])],
             reference=bool(r["reference"]), ref_p=int(r["ref_p"]), ref_cell_nm=float(r["ref_cell_nm"]), field_res=bool(r["field_res"]), field=False,
             ng=dict(lams=[target - NG_STEP_NM, target, target + NG_STEP_NM],
                     core=[eps_pair(m, m["core"], lam) for lam in (target - NG_STEP_NM, target, target + NG_STEP_NM)],
                     clad=[eps_pair(m, m["clad"], lam) for lam in (target - NG_STEP_NM, target, target + NG_STEP_NM)]))
    return j


def _scale(unit):
    return {"nm": 1e-9, "µm": 1e-6, "mm": 1e-3, "m": 1.0}[unit]


def channel_label(c):
    return f"P{c['port'] + 1}·M{c['mode'] + 1}"


# ===================================================================================================================== results
def results_dataframe(points, device, style):
    rows = []
    for p in points:
        r = {"λ (nm)": p["lam_nm"]}
        if "error" in p:
            r["Fehler"] = p["error"]
            rows.append(r)
            continue
        if "through" in p:                                  # ring: launch at port 1
            r.update({"Reflexion R (Port 1)": p["R"], "Through (Port 2)": p["through"], "Drop (Port 3)": p["drop"], "Add (Port 4)": p["add"], "Summe": p["power_sum"],
                      "n_eff (Port 1)": p.get("n_eff0", np.nan), "Unbekannte": p.get("dofs", np.nan), "Zeit (s)": p.get("seconds", np.nan)})
            rows.append(r)
            continue
        r["n_eff (Port 1, Mode 1)"] = p.get("n_eff0", np.nan)
        if "n_eff_ref" in p:
            r["n_eff Referenz (2D, fein)"] = p["n_eff_ref"]
        r["Reflexion R"] = p["R"]
        r["Transmission T"] = p["T"]
        r["Verlust 1−R−T"] = p["loss"]
        if "bar" in p:
            r["Durchgang (bar)"] = p["bar"]
            r["Kopplung (cross)"] = p["cross"]
            r["cross (gekoppelte Moden)"] = p.get("cross_cmt", np.nan)
            if "cross_cmt_ref" in p:
                r["cross Referenz (fein)"] = p["cross_cmt_ref"]
            r["Rückkopplung"] = p.get("back", np.nan)
        if "phase_error" in p:
            r["Phasenfehler (rad)"] = p["phase_error"]
        r["Unbekannte"] = p.get("dofs", np.nan)
        r["Zeit (s)"] = p.get("seconds", np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def s_matrix_dataframe(p):
    if "channels" not in p:                                 # ring: only the launch at port 1
        s = np.array([complex(*v) for v in p["s"]])
        names = ["Port 1 (Eingang, Reflexion)", "Port 2 (Through)", "Port 3 (Drop)", "Port 4 (Add)"]
        return pd.DataFrame({"Ziel": names, "|S|² (Leistung)": np.abs(s) ** 2, "Phase (rad)": np.angle(s), "Betrag |S|": np.abs(s)})
    ch = p["channels"]
    labels = [channel_label(c) for c in ch]
    s = np.array(p["S_re"]) + 1j * np.array(p["S_im"])
    df = pd.DataFrame(np.abs(s) ** 2, columns=[f"nach {lab}" for lab in labels])
    df.insert(0, "Anregung", [f"aus {lab}" for lab in labels])
    return df


def fig_sparams(points):
    import matplotlib.pyplot as plt

    ok = [p for p in points if "R" in p]
    lam = [p["lam_nm"] for p in ok]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if "bar" in ok[0]:
        ax.plot(lam, [p["bar"] for p in ok], "o-", label="Durchgang (bar)")
        ax.plot(lam, [p["cross"] for p in ok], "s-", label="Kopplung (cross)")
        if all("cross_cmt" in p for p in ok):
            ax.plot(lam, [p["cross_cmt"] for p in ok], "k--", label="cross (gekoppelte Moden)")
    else:
        ax.plot(lam, [p["T"] for p in ok], "o-", label="Transmission T")
    ax.plot(lam, [p["R"] for p in ok], "^-", color="C3", label="Reflexion R")
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("Leistungsanteil der Anregung")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def ring_summary_dataframe(ring):
    """Resonance fit, reference and free spectral range of the ring as a two-column table."""
    f = ring.get("fit")
    rows = []
    if f:
        rows += [("Resonanz λ (Anpassung, nm)", f"{f['wavelength']:.4f}"), ("Halbwertsbreite (nm)", f"{f['fwhm']:.4f}"), ("Güte Q (belastet)", f"{f['q']:.0f}"),
                 ("größter Drop", f"{f['max_drop']:.4f}"), ("kleinster Through", f"{f['min_through']:.4f}"), ("Rest der Anpassung", f"{f['residual']:.1e}"), ("Punkte in der Anpassung", f"{f['points']}")]
    if ring.get("lam_ref_nm"):
        rows.append((f"isolierter Ring, m = {ring['chosen_m']}: λ (nm)", f"{ring['lam_ref_nm']:.3f}"))
        if f:
            rows.append(("Differenz 3D − isolierter Ring (nm)", f"{f['wavelength'] - ring['lam_ref_nm']:+.3f}"))
    if ring.get("fsr_ref_nm"):
        rows.append(("FSR aus dem isolierten Ring (nm)", f"{ring['fsr_ref_nm']:.2f}"))
    if ring.get("n_g"):
        rows += [("n_eff der Zuleitung", f"{ring['n_eff_bus']:.4f}"), ("Gruppenindex n_g der Zuleitung", f"{ring['n_g']:.4f}"), ("FSR = λ²/(n_g·L), L = 2πR (nm)", f"{ring['fsr_ng_nm']:.2f}")]
    return pd.DataFrame(rows, columns=["Größe", "Wert"])


def fig_ring(points, ring):
    """Through, drop, add and reflection of the launch at port 1 over the wavelength; the fitted resonance and the reference of the isolated ring as vertical lines."""
    import matplotlib.pyplot as plt

    ok = sorted((p for p in points if "through" in p), key=lambda p: p["lam_nm"])
    lam = [p["lam_nm"] for p in ok]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(lam, [p["through"] for p in ok], "o-", label="Through (Port 2)")
    ax.plot(lam, [p["drop"] for p in ok], "s-", label="Drop (Port 3)")
    ax.plot(lam, [p["add"] for p in ok], "d-", ms=4, label="Add (Port 4)")
    ax.plot(lam, [p["R"] for p in ok], "^-", ms=4, color="C3", label="Reflexion (Port 1)")
    lo, hi = min(lam), max(lam)
    f = ring.get("fit") if ring else None
    if f and lo - 0.05 * (hi - lo) <= f["wavelength"] <= hi + 0.05 * (hi - lo):
        ax.axvline(f["wavelength"], color="k", ls="-.", lw=1.0, label=f"Resonanz (Anpassung) {f['wavelength']:.3f} nm")
    if ring and ring.get("lam_ref_nm") and lo <= ring["lam_ref_nm"] <= hi:
        ax.axvline(ring["lam_ref_nm"], color="gray", ls=":", lw=1.3, label=f"isolierter Ring (m = {ring['chosen_m']}) {ring['lam_ref_nm']:.3f} nm")
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("Leistungsanteil der Anregung (Port 1)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_ring_field(npz, m, title=""):
    """|E|^2 (arbitrary scale) of the launch at the strongest drop on the x-y plane through the mid-height of the cores; white dashed: ring and bus edges."""
    import matplotlib.pyplot as plt

    xs, ys = np.asarray(npz["xs"]) * 1e9, np.asarray(npz["ys"]) * 1e9
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.pcolormesh(xs, ys, np.asarray(npz["e2"], dtype=float).T, shading="auto", cmap="inferno")
    fig.colorbar(im, ax=ax, label="|E|² (willkürliche Einheit)")
    r, w, g = m["ring"]["radius_nm"], m["width_nm"], m["gap_nm"]
    th = np.linspace(0, 2 * np.pi, 361)
    for rr in (r - w / 2, r + w / 2):
        ax.plot(rr * np.cos(th), rr * np.sin(th), "w--", lw=0.8)
    for s in (-1, 1):
        yc = s * (r + w + g)
        ax.plot([xs[0], xs[-1]], [yc - w / 2] * 2, "w--", lw=0.8)
        ax.plot([xs[0], xs[-1]], [yc + w / 2] * 2, "w--", lw=0.8)
    ax.set_xlabel("x, längs der Zuleitungen (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_field(npz, key, outline=None, title=""):
    """|E|^2 (arbitrary scale) on the x-z plane through the cores (key xz: z horizontal, x vertical) or the x-y section at mid length (key xy)."""
    import matplotlib.pyplot as plt

    f = np.asarray(npz[key], dtype=float)
    xs = np.asarray(npz["xs"]) * 1e9
    other = np.asarray(npz["zs"] if key == "xz" else npz["ys"]) * 1e9
    fig, ax = plt.subplots(figsize=(6.4, 3.6) if key == "xz" else (5.4, 4.2))
    if key == "xz":
        im = ax.pcolormesh(other, xs, f, shading="auto", cmap="inferno")
        ax.set_xlabel("z, längs des Wellenleiters (nm)")
        ax.set_ylabel("x (nm)")
    else:
        im = ax.pcolormesh(xs, other, f.T, shading="auto", cmap="inferno")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="|E|² (willkürliche Einheit)")
    if outline:
        for seg in outline.get(key, []):
            ax.plot(*seg, "w--", lw=0.9)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def outline(m):
    """Core outlines [nm] for the two cut views: dict key -> list of (xs, ys) to plot."""
    if m["device"] == "file":
        return {}
    h, L = m["height_nm"], m["length_nm"]
    ex = x_edges(m)
    cores = [(ex[1], ex[2])] if m["device"] == "straight" else [(ex[1], ex[2]), (ex[3], ex[4])]
    xz, xy = [], []
    for a, b in cores:
        xz += [([0, L], [a, a]), ([0, L], [b, b])]
        xy.append(([a, b, b, a, a], [0, 0, h, h, 0]))
    return {"xz": xz, "xy": xy}


def copy_model(m):
    return copy.deepcopy(m)
