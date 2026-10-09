"""Model of the FEM model builder: periodic unit cell with layers and shapes, preview and meshing with Gmsh.

Coordinates in nm, x along the period (0 ... P), y vertical (up). y = 0 is the top of the layer stack: the lower side of the cover (incidence
medium, lossless), then finite layers (top to bottom), then the substrate. Shapes (rectangle, trapezoid, circle, ellipse, polygon) lie anywhere
in the cell, may reach into the cover or into the layers and the substrate and wrap around the cell edges (periodic continuation); a later
shape covers earlier shapes and the layers.

Vertical layout (bottom to top): [PML below | PEC wall] substrate | layers | cover | PML above. The mesh is conforming: every interface of the
stack and every shape boundary is a mesh line. Physical groups for the hpfem Gmsh reader (MSH 4.1 ASCII):
    surfaces: tag = 1 + index of the material in model["materials"]   (name = material name)
    curves:   left = 1, right = 2 (periodic pair), bottom = 3, top = 4  (hpfem.box_tag, as hpfem.grating expects)
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

import fem_materials as fm

TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP = 1, 2, 3, 4                  # hpfem.box_tag X_MIN, X_MAX, Y_MIN, Y_MAX (grating.solve)

SHAPE_TYPES = {
    "rect": ("Rechteck", [("x_center", "Mitte x (nm)"), ("y_bottom", "Unterkante y (nm)"), ("width", "Breite (nm)"), ("height", "Höhe (nm)")]),
    "trapezoid": ("Trapez", [("x_center", "Mitte x (nm)"), ("y_bottom", "Unterkante y (nm)"), ("width_bottom", "Breite unten (nm)"),
                             ("width_top", "Breite oben (nm)"), ("height", "Höhe (nm)")]),
    "circle": ("Kreis", [("x_center", "Mitte x (nm)"), ("y_center", "Mitte y (nm)"), ("radius", "Radius (nm)")]),
    "ellipse": ("Ellipse", [("x_center", "Mitte x (nm)"), ("y_center", "Mitte y (nm)"), ("rx", "Halbachse x (nm)"), ("ry", "Halbachse y (nm)")]),
    "polygon": ("Polygon", []),
}
DEFAULT_SHAPE = {
    "rect": dict(x_center=200.0, y_bottom=0.0, width=200.0, height=100.0),
    "trapezoid": dict(x_center=200.0, y_bottom=0.0, width_bottom=250.0, width_top=150.0, height=150.0),
    "circle": dict(x_center=200.0, y_center=60.0, radius=60.0),
    "ellipse": dict(x_center=200.0, y_center=60.0, rx=90.0, ry=50.0),
    "polygon": dict(points=[[100.0, 0.0], [300.0, 0.0], [200.0, 170.0]]),
}


# ------------------------------------------------------------------------------------------------------------------ model
def new_shape(kind, material):
    s = {"type": kind, "material": material}
    s.update(copy.deepcopy(DEFAULT_SHAPE[kind]))
    return s


def default_model():
    return {
        "name": "Neues Modell",
        "period_nm": 400.0,
        "materials": {"Luft": {"type": "library", "name": "air"}, "Si": {"type": "library", "name": "Si"}},
        "cover": "Luft", "layers": [], "substrate": "Si", "shapes": [],
        "domain": {"cover_nm": 600.0, "substrate_nm": 600.0, "pml_top_nm": 600.0, "pml_bottom_nm": 0.0, "bottom": "pec"},
        "incidence": {"pol": "TM", "theta": 50.0, "phi": 0.0, "wavelength_nm": 405.0},
        "sweep": {"mode": "none", "start": 400.0, "stop": 700.0, "n": 7},
    }


def shape_polygon(s, n=96):
    """Outline of a shape as (N, 2) array in nm (circle and ellipse as polygons)."""
    t = s["type"]
    if t == "rect":
        x0, y0 = s["x_center"] - s["width"] / 2, s["y_bottom"]
        return np.array([[x0, y0], [x0 + s["width"], y0], [x0 + s["width"], y0 + s["height"]], [x0, y0 + s["height"]]])
    if t == "trapezoid":
        xc, y0, wb, wt, h = s["x_center"], s["y_bottom"], s["width_bottom"], s["width_top"], s["height"]
        return np.array([[xc - wb / 2, y0], [xc + wb / 2, y0], [xc + wt / 2, y0 + h], [xc - wt / 2, y0 + h]])
    if t in ("circle", "ellipse"):
        rx, ry = (s["radius"], s["radius"]) if t == "circle" else (s["rx"], s["ry"])
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([s["x_center"] + rx * np.cos(a), s["y_center"] + ry * np.sin(a)])
    if t == "polygon":
        return np.array(s["points"], dtype=float)
    raise ValueError(f"unbekannte Form: {t}")


def shape_bbox(s):
    p = shape_polygon(s)
    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())


def contains(s, X, Y):
    """Boolean array: points (X, Y) in nm inside the shape (analytic for circle / ellipse)."""
    t = s["type"]
    if t == "circle":
        return (X - s["x_center"]) ** 2 + (Y - s["y_center"]) ** 2 <= s["radius"] ** 2
    if t == "ellipse":
        return ((X - s["x_center"]) / s["rx"]) ** 2 + ((Y - s["y_center"]) / s["ry"]) ** 2 <= 1.0
    from matplotlib.path import Path as MPath
    pts = np.column_stack([np.ravel(X), np.ravel(Y)])
    return MPath(shape_polygon(s)).contains_points(pts).reshape(np.shape(X))


def shift_shapes(model, dx_nm):
    """Moves every shape by dx_nm along x (the layers do not depend on x). Shapes that leave the cell are continued periodically."""
    for s in model["shapes"]:
        if s["type"] == "polygon":
            s["points"] = [[float(x) + dx_nm, float(y)] for x, y in s["points"]]
        else:
            s["x_center"] = float(s["x_center"]) + dx_nm


def shapes_x_extent(model):
    """(x_min, x_max) over all shapes in nm, or None without shapes."""
    if not model["shapes"]:
        return None
    boxes = [shape_bbox(s) for s in model["shapes"]]
    return min(b[0] for b in boxes), max(b[1] for b in boxes)


def centering_shift(model):
    """Shift in nm that puts the middle of the x extent of all shapes at the middle of the cell (P / 2); 0 without shapes."""
    ext = shapes_x_extent(model)
    return 0.0 if ext is None else model["period_nm"] / 2 - (ext[0] + ext[1]) / 2


def parse_points(text):
    """'x, y' per line (or 'x y') -> list of [x, y]."""
    pts = []
    for line in text.replace(";", "\n").splitlines():
        line = line.strip().replace(",", " ")
        if line:
            a = line.split()
            if len(a) != 2:
                raise ValueError(f"Zeile '{line}': erwartet 'x, y'")
            pts.append([float(a[0]), float(a[1])])
    return pts


def layout(model):
    """Vertical layout in nm: interfaces of the stack, physical region and PML, slabs bottom to top."""
    d, P = model["domain"], model["period_nm"]
    thick = [l["thickness_nm"] for l in model["layers"]]
    interfaces = [0.0]
    for t in thick:
        interfaces.append(interfaces[-1] - t)
    tops = [shape_bbox(s)[3] for s in model["shapes"]]
    y_struct_top = max([0.0] + tops)
    y_cover_top = y_struct_top + d["cover_nm"]
    y_pml_top = y_cover_top + d["pml_top_nm"]
    y_sub_top = interfaces[-1]
    y_sub_bottom = y_sub_top - d["substrate_nm"]
    pml_bottom = d.get("bottom", "pec") == "pml" and d.get("pml_bottom_nm", 0.0) > 0
    y_pml_bottom = y_sub_bottom - (d["pml_bottom_nm"] if pml_bottom else 0.0)
    slabs = []
    if pml_bottom:
        slabs.append((y_pml_bottom, y_sub_bottom, model["substrate"], "pml"))
    slabs.append((y_sub_bottom, y_sub_top, model["substrate"], "substrate"))
    for i in range(len(thick) - 1, -1, -1):
        slabs.append((interfaces[i + 1], interfaces[i], model["layers"][i]["material"], "layer"))
    slabs.append((0.0, y_cover_top, model["cover"], "cover"))
    slabs.append((y_cover_top, y_pml_top, model["cover"], "pml"))
    return dict(interfaces=interfaces, y_struct_top=y_struct_top, y_cover_top=y_cover_top, y_pml_top=y_pml_top, y_sub_bottom=y_sub_bottom,
                y_pml_bottom=y_pml_bottom, y_min=y_pml_bottom, y_max=y_pml_top, pml_bottom=pml_bottom, slabs=slabs, period=P)


def material_index_map(model, X, Y):
    """Index into list(model['materials']) at the points (X, Y) in nm (x is wrapped periodically); NaN-free."""
    names = list(model["materials"])
    lay = layout(model)
    P = model["period_nm"]
    out = np.zeros(np.shape(X), dtype=int)
    for y0, y1, mat, _ in lay["slabs"]:
        out[(Y >= y0) & (Y <= y1)] = names.index(mat)
    Xw = np.mod(X, P)
    for s in model["shapes"]:
        hit = np.zeros(np.shape(X), dtype=bool)
        for off in (-P, 0.0, P):
            hit |= contains(s, Xw + off, Y)
        out[hit] = names.index(s["material"])
    return out


# --------------------------------------------------------------------------------------------------------- validation
def validate(model, lam_nm=None):
    """List of (level, text), level 'error' or 'warn'."""
    msgs = []
    names = list(model["materials"])
    P = model["period_nm"]
    d = model["domain"]
    if P <= 0:
        msgs.append(("error", "Die Periode muss größer als 0 sein."))
    for what, mat in [("Einfallsmedium", model["cover"]), ("Substrat", model["substrate"])] + [(f"Schicht {i + 1}", l["material"]) for i, l in enumerate(model["layers"])] + \
                     [(f"Form {i + 1}", s["material"]) for i, s in enumerate(model["shapes"])]:
        if mat not in names:
            msgs.append(("error", f"{what}: Material '{mat}' ist nicht definiert."))
    for i, l in enumerate(model["layers"]):
        if l["thickness_nm"] <= 0:
            msgs.append(("error", f"Schicht {i + 1}: Die Dicke muss größer als 0 sein."))
    for i, s in enumerate(model["shapes"]):
        try:
            p = shape_polygon(s)
        except Exception as exc:
            msgs.append(("error", f"Form {i + 1}: {exc}"))
            continue
        if s["type"] == "polygon" and len(p) < 3:
            msgs.append(("error", f"Form {i + 1} (Polygon): mindestens 3 Punkte."))
        for k, v in s.items():
            if k in ("width", "height", "radius", "rx", "ry", "width_bottom", "height") and isinstance(v, (int, float)) and v <= 0:
                msgs.append(("error", f"Form {i + 1}: '{k}' muss größer als 0 sein."))
        if s["type"] == "trapezoid" and s["width_top"] < 0:
            msgs.append(("error", f"Form {i + 1}: Breite oben darf nicht negativ sein."))
    if not msgs:
        lay = layout(model)
        for i, s in enumerate(model["shapes"]):
            x0, x1, y0, y1 = shape_bbox(s)
            if y1 > lay["y_cover_top"] + 1e-9 or y0 < lay["y_sub_bottom"] - 1e-9:
                msgs.append(("error", f"Form {i + 1} reicht in die PML oder aus dem Rechengebiet (y von {y0:.0f} bis {y1:.0f} nm, erlaubt "
                                      f"{lay['y_sub_bottom']:.0f} bis {lay['y_cover_top']:.0f} nm): Höhe oder Tiefe des Gebiets vergrößern."))
            if s["type"] in ("circle", "ellipse"):
                for yi in lay["interfaces"]:
                    if abs(y0 - yi) < 0.5 or abs(y1 - yi) < 0.5:
                        msgs.append(("warn", f"Form {i + 1} ({SHAPE_TYPES[s['type']][0]}) berührt die Grenzfläche bei y = {yi:.0f} nm fast oder genau tangential: "
                                             "im Netz entsteht ein extrem schmaler Keil mit verzerrten Dreiecken. Die Form 1–2 nm eintauchen lassen."))
                        break
            if x1 - x0 > P + 1e-9:
                msgs.append(("warn", f"Form {i + 1} ist breiter als die Periode ({x1 - x0:.0f} nm > {P:.0f} nm): überlappt mit ihren Nachbarn."))
        if lam_nm is not None:
            try:
                e = fm.eps_at(model["materials"][model["cover"]], lam_nm)
                if abs(e.imag) > 1e-9 or e.real < 1:
                    msgs.append(("error", "Das Einfallsmedium muss verlustfrei sein (ε reell, ≥ 1)."))
            except Exception as exc:
                msgs.append(("error", f"Einfallsmedium: {exc}"))
    return msgs


def resolution_limit(p):
    """Largest |k s| h of a PML cell of order p (hpfem PmlBox::resolution_limit): 3 for p >= 4, else 0.75 p."""
    return 3.0 if p >= 4 else 0.75 * max(int(p), 1)


def _incidence_ranges(model):
    sw, inc = model["sweep"], model["incidence"]
    lams = [sw["start"], sw["stop"]] if sw["mode"] == "wavelength" else [inc["wavelength_nm"]]
    thetas = [abs(sw["start"]), abs(sw["stop"])] if sw["mode"] == "theta" else [abs(inc["theta"])]
    phis = [sw["start"], sw["stop"]] if sw["mode"] == "phi" else [inc["phi"]]
    return sorted(lams), thetas, phis


def _cover_sub_index(model, lam):
    try:
        n_c = max(1.0, fm.nk(model["materials"][model["cover"]], lam)[0])
    except Exception:
        n_c = 1.0
    try:
        n_s, k_s = fm.nk(model["materials"][model["substrate"]], lam)
    except Exception:
        n_s, k_s = 1.5, 0.0
    return n_c, max(n_s, 1.0), k_s


def pml_angle(model, cap_deg=80.0):
    """Largest angle against the normal of the propagating orders of the cover over the whole sweep (the angle the PML is designed for)."""
    lams, thetas, phis = _incidence_ranges(model)
    P = model["period_nm"]
    psi = max(thetas)
    for lam in lams:
        n_c = _cover_sub_index(model, lam)[0]
        k = 2 * math.pi * n_c / lam
        for th in thetas:
            for ph in phis:
                kx0 = k * math.sin(math.radians(th)) * math.cos(math.radians(ph))
                beta = k * math.sin(math.radians(th)) * math.sin(math.radians(ph))
                for m in range(-8, 9):
                    kx = kx0 + 2 * math.pi * m / P
                    if kx * kx + beta * beta < k * k:
                        psi = max(psi, math.degrees(math.acos(math.sqrt(max(k * k - kx * kx - beta * beta, 0.0)) / k)))
    return min(psi, cap_deg)


def pml_plan(model, p, target, cap_deg=80.0, n_pml=6.0, ms=None):
    """Design numbers of the PML after the formulas of hpfem (pml.hpp): R0 = (target/2)^(2/cos psi), sigma_max = (m+1) ln(1/R0) / (2 k0 n_ref d),
    |s| = sqrt(1 + sigma^2), resolved if k0 n |s| h <= resolution_limit(p) with the cell size h of the PML mesh (lambda_min / (n_pml n) unless
    the sizes of the actual mesh are given in ms['pml_sizes_nm'])."""
    lay = layout(model)
    lams, _, _ = _incidence_ranges(model)
    psi = pml_angle(model, cap_deg)
    R0 = max((target / 2.0) ** (2.0 / math.cos(math.radians(psi))), 1e-300)
    E = 3.0 * math.log(1.0 / R0)                                  # (m + 1) ln(1/R0) with the profile order m = 2
    n_c, n_s, k_s = _cover_sub_index(model, lams[0])
    lossless_sub = k_s < 1e-9
    n_ref = min(n_c, n_s) if (lay["pml_bottom"] and lossless_sub) else n_c
    L = resolution_limit(p)
    d = model["domain"]
    sides = []
    for name, thick, n_med in (("oben", d["pml_top_nm"], n_c), ("unten", d["pml_bottom_nm"] if lay["pml_bottom"] else 0.0, n_s)):
        if thick <= 0:
            continue
        worst, h_have = None, None
        for lam in set(lams):
            k0 = 2 * math.pi / lam
            sigma = E / (2 * k0 * n_ref * thick)
            s_abs = math.sqrt(1 + sigma * sigma)
            h_need = L / (k0 * n_med * s_abs)
            if worst is None or h_need < worst[0]:
                worst = (h_need, sigma, s_abs, lam)
        h_have = (ms or {}).get("pml_sizes_nm", {}).get(model["cover"] if name == "oben" else model["substrate"])
        if h_have is None:
            h_have = lams[0] / (n_pml * n_med)
        sides.append(dict(side=name, thickness_nm=thick, sigma=worst[1], s=worst[2], h_needed_nm=worst[0], h_mesh_nm=h_have, ok=h_have <= worst[0] * 1.0001))
    return dict(psi_deg=psi, R0=R0, E=E, n_ref=n_ref, limit=L, sides=sides, ok=all(sd["ok"] for sd in sides))


def suggest_domain(model, lam_min_nm, lam_max_nm, order=4, pml_target=1e-3, cap_deg=80.0, n_pml=6.0):
    """Domain heights in nm from the wavelength range.

    Cover: about one wavelength. Substrate: a PEC wall is only safe if the field has decayed to e^-6 before it (six decay lengths 1/(k0 k)); that is
    used when it fits in 1600 nm, otherwise a PML below a substrate of about one wavelength. PML thickness: the smallest thickness at which the
    stretched field is resolved by `n_pml` cells per wavelength of the PML mesh at the polynomial order `order` (see pml_plan); the mesh of the PML
    is built with that many cells per wavelength."""
    d = dict(model["domain"])
    n_cover = _cover_sub_index(model, lam_max_nm)[0]
    d["cover_nm"] = float(round(max(400.0, 0.8 * lam_max_nm / n_cover), -1))
    sub = model["materials"].get(model["substrate"], {"type": "eps", "re": 1.0})
    n_sub, decay = 1.5, math.inf                                       # decay: longest decay length of the field in the substrate (nm)
    for lam in (lam_min_nm, lam_max_nm):
        try:
            n, k = fm.nk(sub, lam)
        except Exception:
            continue
        n_sub = max(n_sub if math.isfinite(decay) else 0.0, n)
        decay = min(decay, lam / (2 * math.pi * k)) if k > 1e-9 and not math.isfinite(decay) else (max(decay, lam / (2 * math.pi * k)) if k > 1e-9 else math.inf)
    if math.isfinite(decay) and 6.0 * decay <= 1600.0:
        d.update(bottom="pec", substrate_nm=float(round(max(300.0, 6.0 * decay), -1)), pml_bottom_nm=0.0)
    else:
        d.update(bottom="pml", substrate_nm=float(round(max(400.0, 0.8 * lam_max_nm / max(n_sub, 1.0)), -1)), pml_bottom_nm=1.0)
    # PML thickness from the resolution criterion
    trial = dict(model, domain=dict(d))
    psi = pml_angle(trial, cap_deg)
    R0 = max((pml_target / 2.0) ** (2.0 / math.cos(math.radians(psi))), 1e-300)
    E = 3.0 * math.log(1.0 / R0)
    n_c, n_s, _ = _cover_sub_index(trial, lam_min_nm)
    n_ref = min(n_c, n_s) if d["bottom"] == "pml" else n_c
    L = resolution_limit(order)
    need = {}
    for key, n_med in (("pml_top_nm", n_c), ("pml_bottom_nm", n_s)):
        worst = 0.0
        for lam in {lam_min_nm, lam_max_nm}:
            k0 = 2 * math.pi / lam
            h = lam_min_nm / (max(n_pml, 2 * math.pi / L * 1.2) * n_med)         # cell size of the PML mesh (nm)
            S = L / (k0 * n_med * h)
            sigma = math.sqrt(max(S * S - 1.0, 1e-6))
            worst = max(worst, E / (2 * k0 * n_ref * sigma))
        need[key] = min(max(worst, 1.5 * lam_max_nm / n_c if key == "pml_top_nm" else 1.5 * lam_max_nm / n_s), 10 * lam_max_nm)
    d["pml_top_nm"] = float(math.ceil(need["pml_top_nm"] / 10.0) * 10.0)
    d["pml_bottom_nm"] = float(math.ceil(need["pml_bottom_nm"] / 10.0) * 10.0) if d["bottom"] == "pml" else 0.0
    return d


# -------------------------------------------------------------------------------------------------------------- preview
PALETTE = ["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc", "#fdd0a2", "#c7e9c0", "#dadaeb", "#fcbba1", "#c6dbef", "#d9d9d9"]
FIXED = {"air": "#f7fbff", "vacuum": "#f7fbff", "Luft": "#f7fbff", "Si": "#7f8fa6", "Ag": "#cfd8dc", "Au": "#f4d03f", "Al": "#b0bec5",
         "SiO2": "#d6eaf8", "Glas": "#d6eaf8", "TiO2": "#f5cba7", "water": "#aed6f1", "Wasser": "#aed6f1", "GaAs": "#a9a9a9"}


def material_color(model, name):
    spec = model["materials"].get(name, {})
    key = spec.get("name") if spec.get("type") == "library" else name
    if key in FIXED:
        return FIXED[key]
    if name in FIXED:
        return FIXED[name]
    return PALETTE[list(model["materials"]).index(name) % len(PALETTE)] if name in model["materials"] else "#cccccc"


def preview_figure(model, show_neighbours=True, figsize=(7.2, 6.0)):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    P = model["period_nm"]
    lay = layout(model)
    fig, ax = plt.subplots(figsize=figsize)
    offsets = (-P, 0.0, P) if show_neighbours else (0.0,)
    for off in offsets:
        alpha = 1.0 if off == 0.0 else 0.35
        for y0, y1, mat, kind in lay["slabs"]:
            ax.add_patch(Rectangle((off, y0), P, y1 - y0, facecolor=material_color(model, mat), edgecolor="none", alpha=alpha,
                                   hatch="///" if kind == "pml" else None, zorder=1))
        for s in model["shapes"]:
            p = shape_polygon(s) + np.array([off, 0.0])
            ax.add_patch(Polygon(p, closed=True, facecolor=material_color(model, s["material"]), edgecolor="k", lw=1.0 if off == 0 else 0.5,
                                 alpha=alpha, zorder=3))
    for y in lay["interfaces"]:
        ax.axhline(y, color="k", lw=0.6, ls="--", zorder=2)
    for x in (0, P):
        ax.axvline(x, color="crimson", lw=1.0, ls=":", zorder=4)
    ax.text(P / 2, lay["y_pml_top"], "PML", ha="center", va="top", fontsize=8, color="0.3", zorder=5)
    if lay["pml_bottom"]:
        ax.text(P / 2, lay["y_pml_bottom"], "PML", ha="center", va="bottom", fontsize=8, color="0.3", zorder=5)
    else:
        ax.plot([0, P], [lay["y_min"]] * 2, color="k", lw=2.0, zorder=5)
        ax.text(P / 2, lay["y_min"], "PEC-Wand", ha="center", va="bottom", fontsize=8, color="0.3", zorder=5)
    handles = [Rectangle((0, 0), 1, 1, facecolor=material_color(model, n), edgecolor="k") for n in model["materials"]]
    ax.legend(handles, list(model["materials"]), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="Materialien")
    ax.set_xlim(-P if show_neighbours else 0, 2 * P if show_neighbours else P)
    ax.set_ylim(lay["y_min"], lay["y_max"])
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm), rot gepunktet: Zellrand (periodisch)")
    ax.set_ylabel("y (nm)")
    ax.set_title(model.get("name", ""), fontsize=10)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------------------------ mesh
def _size_of(spec, lam_min, lam_max, ms):
    """Largest element edge in nm for a material: lambda / (N Re n), and for lossy materials the field decay length / skin_cells."""
    hs = []
    for lam in (lam_min, lam_max):
        try:
            e = fm.eps_at(spec, lam)
        except Exception:
            continue
        n = np.sqrt(e + 0j)
        h = lam / (ms["cells_per_wavelength"] * max(n.real, 1.0))
        if n.imag > 0.05:
            h = min(h, lam / (2 * math.pi * n.imag) / ms["skin_cells"])
        hs.append(h)
    return min(hs) if hs else 50.0


def _offsets_needed(s, P):
    x0, x1, _, _ = shape_bbox(s)
    return [off for off in (-P, 0.0, P) if x1 + off > 1e-9 and x0 + off < P - 1e-9]


def _make_shape(occ, s, off):
    t = s["type"]
    if t == "rect":
        return occ.addRectangle(s["x_center"] - s["width"] / 2 + off, s["y_bottom"], 0, s["width"], s["height"])
    if t == "circle":
        return occ.addDisk(s["x_center"] + off, s["y_center"], 0, s["radius"], s["radius"])
    if t == "ellipse":
        if s["rx"] >= s["ry"]:
            return occ.addDisk(s["x_center"] + off, s["y_center"], 0, s["rx"], s["ry"])
        return occ.addDisk(s["x_center"] + off, s["y_center"], 0, s["ry"], s["rx"], -1, [], [0, 1, 0])
    pts = shape_polygon(s) + np.array([off, 0.0])
    ids = [occ.addPoint(float(x), float(y), 0) for x, y in pts]
    lines = [occ.addLine(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def _snap_nodes(gmsh, ys, P, tol=1e-6):
    """Sets node coordinates that lie within `tol` nm of an interface (y in `ys`) or of a cell edge (x = 0, P) exactly on it. Gmsh computes the
    intersections of curved shapes with the interface lines with rounding noise (about 6e-14 nm); the solver rejects a cell whose vertices lie
    on both sides of a stack interface even by that much."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    c = np.array(coords).reshape(-1, 3)
    ys = np.asarray(ys, dtype=float)
    new = c.copy()
    j = np.abs(c[:, 1][:, None] - ys[None, :]).argmin(axis=1)
    near = np.abs(c[:, 1] - ys[j]) < tol
    new[near, 1] = ys[j][near]
    new[np.abs(c[:, 0]) < tol, 0] = 0.0
    new[np.abs(c[:, 0] - P) < tol, 0] = P
    changed = np.where((new != c).any(axis=1))[0]
    for k in changed:
        gmsh.model.mesh.setNode(int(tags[k]), [float(new[k, 0]), float(new[k, 1]), float(new[k, 2])], [])
    return len(changed)


def reader_check(msh_path, ys=()):
    """Checks the written MSH 4.1 file with the rules of the hpfem Gmsh reader: every line element (boundary facet) must be an edge of a
    triangle (otherwise hpfem raises 'facet (a,b) is not part of the mesh'), every node must belong to a triangle, no cell may be untagged.
    Returns a list of problem texts (empty = ok)."""
    text = Path(msh_path).read_text()
    nodes_sec = text.split("$Nodes\n")[1].split("$EndNodes")[0].strip().split("\n")
    nb, i, node_tags, node_y = int(nodes_sec[0].split()[0]), 1, [], {}
    for _ in range(nb):
        n = int(nodes_sec[i].split()[3])
        tags_ = [int(nodes_sec[i + 1 + k]) for k in range(n)]
        node_tags += tags_
        for k, tg in enumerate(tags_):
            node_y[tg] = float(nodes_sec[i + 1 + n + k].split()[1])
        i += 1 + 2 * n
    el = text.split("$Elements\n")[1].split("$EndElements")[0].strip().split("\n")
    nb, i, tris, lines, types = int(el[0].split()[0]), 1, [], [], set()
    for _ in range(nb):
        dim, ent, typ, n = [int(v) for v in el[i].split()]
        i += 1
        types.add(typ)
        for k in range(n):
            a = [int(v) for v in el[i + k].split()][1:]
            if typ in (2, 9):
                tris.append(a[:3])
            elif typ in (1, 8):
                lines.append(a[:2])
        i += n
    problems = []
    edges = {tuple(sorted((t[a], t[b]))) for t in tris for a, b in ((0, 1), (1, 2), (2, 0))}
    bad = [l for l in lines if tuple(sorted(l)) not in edges]
    if bad:
        problems.append(f"{len(bad)} Randelemente sind keine Dreieckskanten (hpfem: 'facet … is not part of the mesh')")
    corner = {v for t in tris for v in t}
    if types & {1, 2} and types & {8, 9}:
        problems.append("Elemente erster und zweiter Ordnung gemischt")
    if types & {1, 2}:                                   # first order: every node is a vertex
        orphan = len(set(node_tags) - corner)
        if orphan:
            problems.append(f"{orphan} Knoten gehören zu keinem Dreieck")
    for y_target in ys:                                   # the solver rejects cells with vertices on both sides of a stack interface
        vals = np.array(list(node_y.values()))
        y_i = float(vals[np.abs(vals - y_target).argmin()])   # as the worker does: the interface is the nearest vertex coordinate
        if abs(y_i - y_target) > 1e-6:
            continue
        cells = [k for k, t in enumerate(tris) if min(node_y[v] for v in t) < y_i < max(node_y[v] for v in t)]
        if cells:
            problems.append(f"{len(cells)} Zellen liegen auf beiden Seiten der Grenzfläche y = {y_target:g} nm (z. B. Zelle {cells[0]})")
    if not tris:
        problems.append("keine Dreiecke")
    if '$PhysicalNames' not in text:
        problems.append("keine Physical Names")
    return problems


def build_mesh(model, ms, out_dir, log=print):
    """Mesh the model with Gmsh and write mesh.msh (MSH 4.1 ASCII) and mesh_plot.npz into out_dir.

    ms: cells_per_wavelength, skin_cells, interface_factor, curved (bool), lam_min_nm, lam_max_nm, optional pml_cells_per_wavelength (finer cells
    in the PML slabs). Returns a dict of statistics."""
    try:
        import gmsh
    except Exception as exc:
        raise RuntimeError("Das Python-Modul 'gmsh' fehlt (pip install gmsh).") from exc
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errs = [m for lvl, m in validate(model, ms["lam_min_nm"]) if lvl == "error"]
    if errs:
        raise ValueError("Modell fehlerhaft: " + " | ".join(errs))
    names = list(model["materials"])
    P = model["period_nm"]
    lay = layout(model)
    curved = bool(ms.get("curved", True)) and any(s["type"] in ("circle", "ellipse") for s in model["shapes"])
    try:
        gmsh.initialize(interruptible=False)          # Streamlit runs scripts in a worker thread: no signal handler allowed there
    except TypeError:                                  # gmsh < 4.11 has no such argument
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cell")
        occ = gmsh.model.occ
        slab_tags = [occ.addRectangle(0, y0, 0, P, y1 - y0) for y0, y1, _, _ in lay["slabs"]]
        cell = occ.addRectangle(0, lay["y_min"], 0, P, lay["y_max"] - lay["y_min"])
        pieces = []                                               # (dimtag, shape index)
        for idx, s in enumerate(model["shapes"]):
            for off in _offsets_needed(s, P):
                tag = _make_shape(occ, s, off)
                out, _ = occ.intersect([(2, tag)], [(2, cell)], removeObject=True, removeTool=False)
                pieces += [(dt, idx) for dt in out if dt[0] == 2]
        occ.remove([(2, cell)], recursive=True)          # with recursive=False its four edges stay behind as free curves
        objs = [(2, t) for t in slab_tags]
        tools = [dt for dt, _ in pieces]
        out, outmap = occ.fragment(objs, tools)
        occ.synchronize()
        sources = {}                                               # surface -> list of input numbers
        for k, produced in enumerate(outmap):
            for dim, tag in produced:
                if dim == 2:
                    sources.setdefault(tag, []).append(k)
        surf_mat, surf_kind = {}, {}
        for tag, srcs in sources.items():
            shape_srcs = [pieces[k - len(objs)][1] for k in srcs if k >= len(objs)]
            if shape_srcs:
                surf_mat[tag] = model["shapes"][max(shape_srcs)]["material"]
                surf_kind[tag] = "shape"
            else:
                slab_i = min(srcs)
                surf_mat[tag] = lay["slabs"][slab_i][2]
                surf_kind[tag] = lay["slabs"][slab_i][3]
        e = 1e-6 * max(P, 1.0)
        ymin, ymax = lay["y_min"], lay["y_max"]
        used_curves = {c[1] for c in gmsh.model.getBoundary([(2, t) for t in surf_mat], combined=False, oriented=False, recursive=False)}
        sides = {
            TAG_LEFT: gmsh.model.getEntitiesInBoundingBox(-e, ymin - e, -e, e, ymax + e, e, 1),
            TAG_RIGHT: gmsh.model.getEntitiesInBoundingBox(P - e, ymin - e, -e, P + e, ymax + e, e, 1),
            TAG_BOTTOM: gmsh.model.getEntitiesInBoundingBox(-e, ymin - e, -e, P + e, ymin + e, e, 1),
            TAG_TOP: gmsh.model.getEntitiesInBoundingBox(-e, ymax - e, -e, P + e, ymax + e, e, 1),
        }
        sides = {k: [c for c in v if c[1] in used_curves] for k, v in sides.items()}
        for tag, nm in ((TAG_LEFT, "left"), (TAG_RIGHT, "right"), (TAG_BOTTOM, "bottom"), (TAG_TOP, "top")):
            if not sides[tag]:
                raise RuntimeError(f"Gmsh: keine Randkurven '{nm}' gefunden (Geometrie nicht konform)")
            gmsh.model.addPhysicalGroup(1, [c[1] for c in sides[tag]], tag, nm)
        for name in names:
            surfs = [t for t, m in surf_mat.items() if m == name]
            if surfs:
                gmsh.model.addPhysicalGroup(2, surfs, 1 + names.index(name), name)
        gmsh.model.mesh.setPeriodic(1, [c[1] for c in sides[TAG_RIGHT]], [c[1] for c in sides[TAG_LEFT]], [1, 0, 0, P, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        # element sizes at the points: wavelength in the material, refined at interfaces
        size_of = {n: _size_of(model["materials"][n], ms["lam_min_nm"], ms["lam_max_nm"], ms) for n in names}
        n_pml = float(ms.get("pml_cells_per_wavelength", 0.0) or 0.0)
        pml_size = {}
        if n_pml > 0:                                             # PML cells: |k s| h must stay below the resolution limit, see pml_plan
            for n in names:
                try:
                    pml_size[n] = ms["lam_min_nm"] / (n_pml * max(fm.nk(model["materials"][n], ms["lam_min_nm"])[0], 1.0))
                except Exception:
                    pass
        point_size, point_mats = {}, {}
        for tag, mat in surf_mat.items():
            h_surf = min(size_of[mat], pml_size.get(mat, 1e9)) if surf_kind[tag] == "pml" else size_of[mat]
            for dim, p in gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False, recursive=True):
                point_size[p] = min(point_size.get(p, 1e9), h_surf)
                point_mats.setdefault(p, set()).add(mat)
        for p in point_size:
            if len(point_mats[p]) > 1:
                point_size[p] *= ms["interface_factor"]
        for p, h in point_size.items():
            gmsh.model.mesh.setSize([(0, p)], h)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(size_of.values()))
        gmsh.model.mesh.generate(2)
        if curved:
            gmsh.model.mesh.setOrder(2)
        snap_ys = sorted({round(float(v), 9) for slab in lay["slabs"] for v in slab[:2]} | {round(float(v), 9) for v in lay["interfaces"]})
        n_snapped = _snap_nodes(gmsh, snap_ys, round(float(P), 9))
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        msh = out_dir / "mesh.msh"
        gmsh.write(str(msh))
        # data for plotting and statistics
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(ntags)}
        tris, mats = [], []
        for tag, mat in surf_mat.items():
            types, _, enodes = gmsh.model.mesh.getElements(2, tag)
            for tp, nd in zip(types, enodes):
                if tp not in (2, 9):
                    continue
                per = 3 if tp == 2 else 6
                arr = np.array(nd, dtype=int).reshape(-1, per)[:, :3]
                tris += [[index[int(v)] for v in row] for row in arr]
                mats += [names.index(mat)] * len(arr)
        tris, mats = np.array(tris), np.array(mats)
        xy = coords[:, :2]
        # periodic check: the nodes on the left and right sides must be copies of each other
        left_y = np.sort(np.unique(np.round(xy[np.abs(xy[:, 0]) < 1e-6, 1], 6)))
        right_y = np.sort(np.unique(np.round(xy[np.abs(xy[:, 0] - P) < 1e-6, 1], 6)))
        periodic_ok = len(left_y) == len(right_y) and bool(np.allclose(left_y, right_y, atol=1e-5))
        a, b, c = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
        lens = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], axis=1)
        s = lens.sum(axis=1) / 2
        area = np.sqrt(np.maximum(s * (s - lens[:, 0]) * (s - lens[:, 1]) * (s - lens[:, 2]), 0))
        cosines = np.stack([(lens[:, 0] ** 2 + lens[:, 2] ** 2 - lens[:, 1] ** 2) / (2 * lens[:, 0] * lens[:, 2]),
                            (lens[:, 0] ** 2 + lens[:, 1] ** 2 - lens[:, 2] ** 2) / (2 * lens[:, 0] * lens[:, 1]),
                            (lens[:, 1] ** 2 + lens[:, 2] ** 2 - lens[:, 0] ** 2) / (2 * lens[:, 1] * lens[:, 2])], axis=1)
        min_angle = np.degrees(np.arccos(np.clip(cosines.max(axis=1), -1, 1)))
        np.savez(out_dir / "mesh_plot.npz", xy=xy, tri=tris, mat=mats, names=np.array(names), order=2 if curved else 1)
        problems = reader_check(msh, snap_ys)
        if problems:
            raise RuntimeError("Das Netz erfüllt die Anforderungen des hpfem-Lesers nicht: " + "; ".join(problems))
        stats = dict(cells=int(len(tris)), nodes=int(len(ntags)), order=2 if curved else 1, periodic_ok=periodic_ok,
                     edge_min_nm=float(lens.min()), edge_max_nm=float(lens.max()), edge_mean_nm=float(lens.mean()),
                     min_angle_deg=float(min_angle.min()), mean_min_angle_deg=float(min_angle.mean()),
                     cells_per_material={names[i]: int((mats == i).sum()) for i in sorted(set(mats))},
                     area_per_material_nm2={names[i]: float(area[mats == i].sum()) for i in sorted(set(mats))},
                     sizes_nm={n: round(float(h), 2) for n, h in size_of.items()}, snapped_nodes=int(n_snapped),
                     pml_sizes_nm={n: round(float(min(size_of[n], pml_size.get(n, 1e9))), 2) for n in {surf_mat[t] for t in surf_kind if surf_kind[t] == "pml"}},
                     file=str(msh))
        return stats
    finally:
        gmsh.finalize()


def mesh_figure(plot_npz, model=None, figsize=(7.2, 6.0), zoom=None):
    """Triangles coloured by material (from mesh_plot.npz); zoom = (x0, x1, y0, y1) in nm."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    d = np.load(plot_npz, allow_pickle=False)
    xy, tri, mat, names = d["xy"], d["tri"], d["mat"], [str(n) for n in d["names"]]
    fig, ax = plt.subplots(figsize=figsize)
    colors = [material_color(model, names[m]) if model else "#cccccc" for m in mat]
    ax.add_collection(PolyCollection(xy[tri], facecolors=colors, edgecolors="0.35", linewidths=0.25))
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------------------------------- presets
def _mat_lib(n):
    return {"type": "library", "name": n}


def presets():
    P = {}
    m = default_model()
    m.update(name="Si-Lamellengitter (Projektfall: 400 nm, Nut 200 nm, 148 nm tief)",
             materials={"Luft": _mat_lib("air"), "Si": _mat_lib("Si")}, cover="Luft", substrate="Si",
             shapes=[dict(type="rect", material="Luft", x_center=200.0, y_bottom=-148.0, width=200.0, height=148.0)])
    m["domain"].update(bottom="pec", substrate_nm=400.0)
    P[m["name"]] = m

    m = default_model()
    m.update(name="Ag-Lamellengitter (TM, 50°: Test der hp-Adaptivität)",
             materials={"Luft": _mat_lib("air"), "Ag": _mat_lib("Ag")}, cover="Luft", substrate="Ag",
             shapes=[dict(type="rect", material="Luft", x_center=200.0, y_bottom=-148.0, width=200.0, height=148.0)])
    m["incidence"].update(pol="TM", theta=50.0, wavelength_nm=405.0)
    P[m["name"]] = m

    m = default_model()
    m.update(name="Trapezsteg aus Si auf Si-Substrat (Periode 500 nm)", period_nm=500.0,
             materials={"Luft": _mat_lib("air"), "Si": _mat_lib("Si")}, cover="Luft", substrate="Si",
             shapes=[dict(type="trapezoid", material="Si", x_center=250.0, y_bottom=0.0, width_bottom=260.0, width_top=140.0, height=160.0)])
    m["incidence"].update(pol="TM", theta=30.0, wavelength_nm=600.0)
    m["domain"].update(bottom="pec", substrate_nm=400.0)
    P[m["name"]] = m

    m = default_model()
    m.update(name="TiO2-Stege auf Glas (Metaoberfläche)", period_nm=400.0,
             materials={"Luft": _mat_lib("air"), "SiO2": _mat_lib("SiO2"), "TiO2": _mat_lib("TiO2")}, cover="Luft", substrate="SiO2",
             shapes=[dict(type="rect", material="TiO2", x_center=200.0, y_bottom=0.0, width=180.0, height=300.0)])
    m["incidence"].update(pol="TE", theta=0.0, wavelength_nm=650.0)
    m["sweep"].update(mode="wavelength", start=550.0, stop=800.0, n=11)
    m["domain"].update(bottom="pml", substrate_nm=500.0, pml_bottom_nm=1000.0, pml_top_nm=1000.0, cover_nm=600.0)
    P[m["name"]] = m

    m = default_model()
    m.update(name="Ag-Nanodrähte (Kreise) auf Glas", period_nm=300.0,
             materials={"Luft": _mat_lib("air"), "SiO2": _mat_lib("SiO2"), "Ag": _mat_lib("Ag")}, cover="Luft", substrate="SiO2",
             shapes=[dict(type="circle", material="Ag", x_center=150.0, y_center=43.0, radius=45.0)])
    m["incidence"].update(pol="TM", theta=0.0, wavelength_nm=450.0)
    m["sweep"].update(mode="wavelength", start=350.0, stop=600.0, n=11)
    m["domain"].update(bottom="pml", substrate_nm=400.0, pml_bottom_nm=800.0, pml_top_nm=800.0, cover_nm=500.0)
    P[m["name"]] = m

    m = default_model()
    m.update(name="Dünnschicht SiO2 auf Si (ebener Stapel, Test gegen Fresnel)", period_nm=200.0,
             materials={"Luft": _mat_lib("air"), "SiO2": _mat_lib("SiO2"), "Si": _mat_lib("Si")}, cover="Luft", substrate="Si",
             layers=[dict(material="SiO2", thickness_nm=100.0)], shapes=[])
    m["incidence"].update(pol="TE", theta=40.0, wavelength_nm=550.0)
    m["domain"].update(bottom="pec", substrate_nm=400.0)
    P[m["name"]] = m

    m = default_model()
    m.update(name="Si-Ellipsen-Gitter mit Schicht (Periode 600 nm)", period_nm=600.0,
             materials={"Luft": _mat_lib("air"), "SiO2": _mat_lib("SiO2"), "Si": _mat_lib("Si")}, cover="Luft", substrate="SiO2",
             layers=[dict(material="SiO2", thickness_nm=50.0)],
             shapes=[dict(type="ellipse", material="Si", x_center=300.0, y_center=98.0, rx=200.0, ry=100.0)])
    m["incidence"].update(pol="TE", theta=0.0, wavelength_nm=800.0)
    m["domain"].update(bottom="pml", substrate_nm=500.0, pml_bottom_nm=1200.0, pml_top_nm=1200.0, cover_nm=700.0)
    P[m["name"]] = m
    for m in P.values():
        sw, inc = m["sweep"], m["incidence"]
        lams = [sw["start"], sw["stop"]] if sw["mode"] == "wavelength" else [inc["wavelength_nm"]]
        m["domain"] = suggest_domain(m, min(lams), max(lams))
    return P


def save_model(model, path):
    Path(path).write_text(json.dumps(model, indent=1, ensure_ascii=False), encoding="utf-8")


def load_model(text):
    m = json.loads(text)
    base = default_model()
    for k, v in base.items():
        m.setdefault(k, v)
    return m
