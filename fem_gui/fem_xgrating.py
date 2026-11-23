"""Doubly periodic cross grating ("Kreuzgitter") without hpfem and without Streamlit: model, validation, diffraction orders, figures.

Input mask only (hp-FEM M21): there is no computation yet; the solver interface is fixed with hp-FEM M21 S2 and announced in PhotonPower/hp-FEM#237.

Conventions (as in hpfem): the stack normal is z, the structure is periodic in x and y with the periods ``period_x_nm`` and ``period_y_nm``;
the unit cell is [0, px] x [0, py]. Layers run from top to bottom (``cover`` above the first, ``substrate`` below the last). A layer has a
background material and shapes (priority by order, later ones cover earlier ones) that lie in the xy plane:

* ``circle``:  ``cx, cy, radius``
* ``ellipse``: ``cx, cy, rx, ry`` (semi-axes along x and y)
* ``rect``:    ``cx, cy, wx, wy`` (full widths along x and y)

``kind`` is ``hole`` or ``pillar``; both fill the region with ``material`` (a hole usually with the cover material, a pillar with its own),
the kind only chooses the default material and the label. Lengths in nm. Incidence: wavelength in vacuum, polar angle theta from the normal z,
azimuth phi of the plane of incidence from x, polarisation ``s`` (E perpendicular to the plane of incidence) or ``p``. Time factor exp(-i omega t).

FourierGrating uses y as the stack normal; the conversion is done by the import in hpfem, not here.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np

import fem_geometry as fg
import fem_materials as fm

KIND = "xgrating"
SHAPES = {"circle": "Kreis", "ellipse": "Ellipse", "rect": "Rechteck"}
KINDS = {"hole": "Loch", "pillar": "Säule"}
POLS = {"s": "s (E senkrecht zur Einfallsebene)", "p": "p (E in der Einfallsebene)"}
DEFAULT_SHAPE = {
    "circle": dict(cx=250.0, cy=250.0, radius=125.0),
    "ellipse": dict(cx=250.0, cy=250.0, rx=150.0, ry=100.0),
    "rect": dict(cx=250.0, cy=250.0, wx=200.0, wy=200.0),
}
_SIZE_KEYS = ("radius", "rx", "ry", "wx", "wy")


# ===================================================================================================================== model
def new_shape(kind_of_shape, material, kind="hole", period=(500.0, 500.0)):
    s = {"type": kind_of_shape, "kind": kind, "material": material}
    s.update(copy.deepcopy(DEFAULT_SHAPE[kind_of_shape]))
    s["cx"], s["cy"] = period[0] / 2, period[1] / 2
    return s


def default_model():
    """The hole grating of the FourierGrating example: circular holes (r = 125 nm) in a 150 nm glass layer on glass, period 500 nm, 405 nm."""
    return {
        "kind": KIND, "name": "Lochgitter in Glas auf Glas (Periode 500 nm)",
        "period_x_nm": 500.0, "period_y_nm": 500.0,
        "materials": {"Luft": {"type": "index", "n": 1.0, "k": 0.0}, "Glas": {"type": "index", "n": 1.5, "k": 0.0}},
        "cover": "Luft", "substrate": "Glas",
        "layers": [{"thickness_nm": 150.0, "material": "Glas",
                    "shapes": [{"type": "circle", "kind": "hole", "material": "Luft", "cx": 250.0, "cy": 250.0, "radius": 125.0}]}],
        "incidence": {"theta": 0.0, "phi": 0.0, "pol": "s", "wavelength_nm": 405.0},
        "sweep": {"mode": "single", "start": 380.0, "stop": 500.0, "n": 13},
    }


def presets():
    base = default_model()
    out = {base["name"]: base}
    pil = copy.deepcopy(base)
    pil.update(name="Si-Säulen auf Glas (Periode 600 nm, schräger Einfall)", period_x_nm=600.0, period_y_nm=600.0)
    pil["materials"] = {"Luft": {"type": "index", "n": 1.0, "k": 0.0}, "Glas": {"type": "index", "n": 1.5, "k": 0.0},
                        "Si3N4": {"type": "index", "n": 2.0, "k": 0.0}}
    pil["layers"] = [{"thickness_nm": 200.0, "material": "Luft",
                      "shapes": [{"type": "ellipse", "kind": "pillar", "material": "Si3N4", "cx": 300.0, "cy": 300.0, "rx": 200.0, "ry": 150.0}]}]
    pil["incidence"] = {"theta": 30.0, "phi": 20.0, "pol": "p", "wavelength_nm": 633.0}
    out[pil["name"]] = pil
    flat = copy.deepcopy(base)
    flat.update(name="Ebene Schicht auf Glas (Probe gegen Fresnel)")
    flat["layers"] = [{"thickness_nm": 100.0, "material": "Glas", "shapes": []}]
    out[flat["name"]] = flat
    return out


def sweep_list(sw, inc):
    if sw["mode"] == "single":
        return [float(inc["wavelength_nm"])]
    return [float(round(v, 6)) for v in np.linspace(float(sw["start"]), float(sw["stop"]), max(int(sw["n"]), 1))]


def used_materials(m):
    return {m["cover"], m["substrate"]} | {la["material"] for la in m["layers"]} | {s["material"] for la in m["layers"] for s in la["shapes"]}


def rename_material(m, old, new):
    m["materials"] = {(new if k == old else k): v for k, v in m["materials"].items()}
    for key in ("cover", "substrate"):
        if m[key] == old:
            m[key] = new
    for la in m["layers"]:
        if la["material"] == old:
            la["material"] = new
        for s in la["shapes"]:
            if s["material"] == old:
                s["material"] = new


def load_model(text):
    """Reads a cross-grating model (JSON); foreign material specs without "type" are normalised, what cannot be read raises ``ValueError``."""
    m = json.loads(text)
    if not isinstance(m, dict):
        raise ValueError("Kein Modell (JSON-Objekt erwartet)")
    if m.get("kind") != KIND:
        raise ValueError("Kein Kreuzgitter-Modell (kind = xgrating)"
                         + (": FourierGrating-Dateien werden mit dem Import von hpfem umgesetzt (hp-FEM M21 S3)" if "period_um" in m or "dimension" in m else ""))
    base = default_model()
    mats = m.get("materials")
    if not isinstance(mats, dict) or not mats:
        raise ValueError("„materials“ muss ein nicht leeres Objekt {Name: Angabe} sein")
    for name, spec in list(mats.items()):
        try:
            mats[name] = fm.normalize_spec(spec)
        except ValueError as exc:
            raise ValueError(f"Material '{name}': {exc}") from exc
    for k, v in base.items():
        m.setdefault(k, copy.deepcopy(v))
    for k in ("incidence", "sweep"):
        for kk, vv in base[k].items():
            m[k].setdefault(kk, vv)
    for i, la in enumerate(m["layers"]):
        if not isinstance(la, dict) or "thickness_nm" not in la or "material" not in la:
            raise ValueError(f"Schicht {i + 1}: „thickness_nm“ und „material“ fehlen")
        la.setdefault("shapes", [])
        for j, s in enumerate(la["shapes"]):
            if s.get("type") not in SHAPES:
                raise ValueError(f"Schicht {i + 1}, Form {j + 1}: unbekannte Form '{s.get('type')}' (bekannt: {', '.join(SHAPES)})")
            s.setdefault("kind", "hole")
    errs = [t for lvl, t in validate(m) if lvl == "error" and "nicht definiert" in t]
    if errs:
        raise ValueError(errs[0])
    return m


# ===================================================================================================================== validation
def shape_extent(s):
    """Half-widths (hx, hy) of the bounding box of a shape in nm."""
    t = s["type"]
    if t == "circle":
        return s["radius"], s["radius"]
    if t == "ellipse":
        return s["rx"], s["ry"]
    return s["wx"] / 2, s["wy"] / 2


def validate(m, lam_nm=None):
    """List of (level, text), level 'error' or 'warn'."""
    msgs = []
    px, py = m["period_x_nm"], m["period_y_nm"]
    if px <= 0 or py <= 0:
        msgs.append(("error", "Die Perioden müssen größer als 0 sein."))
    names = m["materials"]
    refs = [("Einfallsmedium", m["cover"]), ("Substrat", m["substrate"])]
    refs += [(f"Schicht {i + 1}", la["material"]) for i, la in enumerate(m["layers"])]
    refs += [(f"Schicht {i + 1}, Form {j + 1}", s["material"]) for i, la in enumerate(m["layers"]) for j, s in enumerate(la["shapes"])]
    for what, mat in refs:
        if mat not in names:
            msgs.append(("error", f"{what}: Material '{mat}' ist nicht definiert."))
    for i, la in enumerate(m["layers"]):
        if la["thickness_nm"] <= 0:
            msgs.append(("error", f"Schicht {i + 1}: Die Dicke muss größer als 0 sein."))
        for j, s in enumerate(la["shapes"]):
            for k in _SIZE_KEYS:
                if k in s and s[k] <= 0:
                    msgs.append(("error", f"Schicht {i + 1}, Form {j + 1}: '{k}' muss größer als 0 sein."))
            if px > 0 and py > 0 and not any(k in s and s[k] <= 0 for k in _SIZE_KEYS):
                hx, hy = shape_extent(s)
                if 2 * hx > px + 1e-9 or 2 * hy > py + 1e-9:
                    msgs.append(("warn", f"Schicht {i + 1}, Form {j + 1} ist größer als die Zelle und überlappt ihre periodischen Nachbarn."))
    inc = m["incidence"]
    if not 0.0 <= inc["theta"] < 90.0:
        msgs.append(("error", "Der Polarwinkel θ muss in [0°, 90°) liegen."))
    if inc["pol"] not in POLS:
        msgs.append(("error", "Polarisation s oder p wählen."))
    if m["sweep"]["mode"] != "single":
        sw = m["sweep"]
        if sw["stop"] <= sw["start"] or sw["n"] < 2:
            msgs.append(("error", "Durchlauf: „bis“ muss größer als „von“ sein, mindestens 2 Punkte."))
    lams = [lam_nm] if lam_nm else sweep_list(m["sweep"], inc)
    if min(lams) <= 0:
        msgs.append(("error", "Die Wellenlänge muss größer als 0 sein."))
    elif not [x for x in msgs if x[0] == "error"]:
        for name in sorted(used_materials(m)):
            for lam in (min(lams), max(lams)):
                try:
                    fm.eps_at(m["materials"][name], lam)
                except Exception as exc:
                    msgs.append(("error", f"{name} bei {lam:g} nm: {exc}"))
                    break
    return msgs


# ===================================================================================================================== diffraction orders
def propagating_orders(m, lam_nm, nmax=12):
    """Orders (m, n) that propagate in the cover (reflection) and in the substrate (transmission) at ``lam_nm``, from the grating equation
    k_t = k_t,inc + 2 pi (m / px, n / py), |k_t| < k0 Re n: ``{"R": [(m, n), ...], "T": [...]}`` sorted by |(m, n)|."""
    inc = m["incidence"]
    th, ph = math.radians(inc["theta"]), math.radians(inc["phi"])
    n_c = max(fm.nk(m["materials"][m["cover"]], lam_nm)[0], 1e-9)
    n_s = fm.nk(m["materials"][m["substrate"]], lam_nm)[0]
    k0 = 2 * math.pi / lam_nm
    kx0, ky0 = k0 * n_c * math.sin(th) * math.cos(ph), k0 * n_c * math.sin(th) * math.sin(ph)
    out = {"R": [], "T": []}
    for key, n_med in (("R", n_c), ("T", n_s)):
        for a in range(-nmax, nmax + 1):
            for b in range(-nmax, nmax + 1):
                kx, ky = kx0 + 2 * math.pi * a / m["period_x_nm"], ky0 + 2 * math.pi * b / m["period_y_nm"]
                if kx * kx + ky * ky < (k0 * n_med) ** 2 * (1 - 1e-12):
                    out[key].append((a, b))
        out[key].sort(key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    return out


def orders_table(m, lam_nm):
    """Rows of the (empty) result table: one row per order (m, n) propagating in cover or substrate; R and T are filled by the solver later."""
    po = propagating_orders(m, lam_nm)
    both = sorted(set(po["R"]) | set(po["T"]), key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    return [{"m": a, "n": b, "reflektiert": (a, b) in po["R"], "transmittiert": (a, b) in po["T"], "R": None, "T": None} for a, b in both]


# ===================================================================================================================== figures
def fig_cell(m, layer=0, n_cells=1):
    """Top view of the unit cell of layer ``layer`` (shapes with their periodic images, n_cells x n_cells cells)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle

    px, py = m["period_x_nm"], m["period_y_nm"]
    la = m["layers"][layer]
    fig, ax = plt.subplots(figsize=(4.4, 4.4 * max(py, 1e-9) / max(px, 1e-9)))
    ax.add_patch(Rectangle((0, 0), px * n_cells, py * n_cells, facecolor=fg.material_color(m, la["material"]), edgecolor="none"))
    for s in la["shapes"]:
        col = fg.material_color(m, s["material"])
        hx, hy = shape_extent(s)
        for i in range(n_cells):
            for j in range(n_cells):
                cx, cy = s["cx"] + i * px, s["cy"] + j * py
                if s["type"] == "rect":
                    ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, facecolor=col, edgecolor="k", lw=0.6))
                else:
                    ax.add_patch(Ellipse((cx, cy), 2 * hx, 2 * hy, facecolor=col, edgecolor="k", lw=0.6))
    for i in range(n_cells + 1):
        ax.axvline(i * px, color="k", lw=0.8, ls="--" if i % n_cells else "-")
    for j in range(n_cells + 1):
        ax.axhline(j * py, color="k", lw=0.8, ls="--" if j % n_cells else "-")
    ax.set_xlim(-0.05 * px, px * n_cells * 1.05)
    ax.set_ylim(-0.05 * py, py * n_cells * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"Schicht {layer + 1}: Hintergrund {la['material']}, Zelle {px:g} × {py:g} nm", fontsize=9)
    handles = [Rectangle((0, 0), 1, 1, facecolor=fg.material_color(m, n), edgecolor="k") for n in m["materials"]]
    ax.legend(handles, list(m["materials"]), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="Materialien")
    fig.tight_layout()
    return fig


def fig_stack(m):
    """Side view of the layer stack (top to bottom) with cover and substrate."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ts = [la["thickness_nm"] for la in m["layers"]]
    ref = max(sum(ts), 100.0) * 0.5
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    y = 0.0
    ax.add_patch(Rectangle((0, y), 1, ref * 0.6, facecolor=fg.material_color(m, m["cover"]), edgecolor="k"))
    ax.text(0.5, y + ref * 0.3, f"Einfallsmedium: {m['cover']}", ha="center", va="center", fontsize=8)
    for i, la in enumerate(m["layers"]):
        y -= la["thickness_nm"]
        ax.add_patch(Rectangle((0, y), 1, la["thickness_nm"], facecolor=fg.material_color(m, la["material"]), edgecolor="k"))
        ax.text(0.5, y + la["thickness_nm"] / 2, f"Schicht {i + 1}: {la['material']}, {la['thickness_nm']:g} nm" + (f", {len(la['shapes'])} Form(en)" if la["shapes"] else ""),
                ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((0, y - ref * 0.6), 1, ref * 0.6, facecolor=fg.material_color(m, m["substrate"]), edgecolor="k"))
    ax.text(0.5, y - ref * 0.3, f"Substrat: {m['substrate']}", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(y - ref * 0.6, ref * 0.6)
    ax.set_xticks([])
    ax.set_ylabel("z (nm)")
    ax.set_title("Schichtstapel (oben: Einfall)", fontsize=9)
    fig.tight_layout()
    return fig
