"""Waveguides and ports in 3D (hp-FEM 0.5): models, job, checks and figures. Pure Python (no hpfem needed); worker: fem_ports3d_worker.py, user interface:
fem_threed_app.py (task ``ports`` of the mode "3D-Strukturen").

Devices (z along the guide, x across, y up; PEC walls on the four side faces, modal ports on the two end faces z = 0 and z = L, as in the library example
``directional_coupler_3d``):

* ``straight``: one rectangular core (width x height) in a cladding; one port per end face with ``num_modes`` modes; reference: the fundamental mode of the
  same cross-section from the 2D mode solver (``PropagatingMode``): n_eff, and the phase of the transmission against beta L.
* ``coupler``: two parallel cores with a gap (directional coupler); ``ports = "half"`` splits each end face into two ports (core A / core B, one mode each),
  ``"full"`` uses one two-mode port per face (even and odd supermode); reference: coupled-mode theory P_cross = sin^2(kappa L), kappa = (beta_e - beta_o) / 2 from the
  supermodes of the cross-section.
* ``file``: a Gmsh mesh of the user (physical volumes = materials, physical surfaces = PEC walls and ports), e.g. the ring resonator of the library
  (examples/ring_resonator_3d/ring_mesh.py); no reference.

Structured meshes (hpfem.box, tetrahedra, cells aligned with the core edges) so that no Gmsh is needed for the first two.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

import fem_materials as fm

DEVICES = {"straight": "Streifenwellenleiter", "coupler": "Richtkoppler (zwei Kerne)", "file": "Eigene Gmsh-Datei (.msh)"}
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
            "file": {"name": "", "scale": "nm", "tag_material": {}, "pec_tags": [], "ports": [], "tags": {}}}


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
    return p


def wavelengths(m):
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
    if m["device"] == "file":
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
        for lam in wavelengths(m):
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


def job(m, max_gb=None):
    j = {"task": "ports", "name": m["name"], "device": m["device"], "order": int(m["order"]), "max_gb": float(max_gb if max_gb is not None else m["max_gb"]),
         "wavelengths_nm": wavelengths(m), "field": bool(m["field"]), "excite_port": int(m["excite_port"]), "excite_mode": int(m["excite_mode"])}
    if m["device"] == "file":
        f = m["file"]
        j.update(path=f.get("path", ""), scale=_scale(f["scale"]),
                 pec_tags=[int(t) for t in f["pec_tags"]], ports=[dict(tag=int(p["tag"]), modes=int(p["modes"])) for p in f["ports"]],
                 tag_eps={str(t): [eps_pair(m, nm, lam) for lam in j["wavelengths_nm"]] for t, nm in f["tag_material"].items() if nm in m["materials"]})
        return j
    lams = j["wavelengths_nm"]
    j.update(width_nm=float(m["width_nm"]), height_nm=float(m["height_nm"]), gap_nm=float(m["gap_nm"]), length_nm=float(m["length_nm"]),
             margin_x_nm=float(m["margin_x_nm"]), margin_y_nm=float(m["margin_y_nm"]), cell_nm=float(m["cell_nm"]), cell_z_nm=float(m["cell_z_nm"]),
             style=m["ports"], num_modes=int(m["num_modes"]) if m["device"] == "straight" or m["ports"] == "full" else 1,
             eps_core=[eps_pair(m, m["core"], lam) for lam in lams], eps_clad=[eps_pair(m, m["clad"], lam) for lam in lams])
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
