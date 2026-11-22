"""Heat and mechanics (2D) in the FEM model builder, hp-FEM 0.5, without hpfem and without Streamlit: models, presets, validation, analytic checks,
result handling and figures. Solver and mesh generators: hpfem (``square_with_disc``, ``rectangle``, ``ThermoOptical2D`` / ``Thermal2D``,
``Elasticity2D``); worker: fem_thermo_worker.py.

Three cases:

* ``wire``: a disc (nanowire, invariant along z) in a matrix, with a PML for the optical problem (hpfem.square_with_disc, the disc has tag 2, the matrix
  tag 0). Chain: plane wave -> absorbed power -> temperature (heat sink at the outer wall of the computational domain, T = T_ref) -> optional feedback
  eps(T) = eps + (d eps / dT)(T - T_ref) (``ThermoOptical2D``) -> optional thermal stress in plane strain (outer wall clamped or free body).
* ``block``: rectangle [0, W] x [0, H] in plane strain, boundary conditions per side (free, clamped, symmetry, traction), temperature change and shrinkage.
* ``axi``: meridian half-plane of a cylinder (x = r in [0, R], y = z in [0, H], axis x = 0), same options; stresses (rr, zz, theta theta, rz).

Units in the model: nm (wire), um (block, axi), MV/m, K, W/(m K), GPa, 1e-6/K, MPa. The solver gets SI. Conventions: time factor exp(-i omega t), Im eps > 0 loss.

Library conventions that matter (checked against the closed forms in ``analytic_checks``): the 2D plane-strain solver applies temperature and shrinkage as an
in-plane eigenstrain (eps*_zz = 0), the thermal strain is alpha (T - T_ref) with alpha per material tag; 2D results are per metre of the invariant direction.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

CASES = {"wire": "Nanodraht: Optik → Wärme → Spannung", "block": "Block (ebene Dehnung)", "axi": "Zylinder (rotationssymmetrisch)"}
BC_KINDS = {"free": "frei", "clamped": "eingespannt (u = 0)", "symmetry": "Symmetrie (u_n = 0)", "traction": "Zug (+) / Druck (−) σ"}
SIDES_BLOCK = {"left": "links (x = 0)", "right": "rechts (x = W)", "bottom": "unten (y = 0)", "top": "oben (y = H)"}
SIDES_AXI = {"left": "Achse (r = 0)", "right": "Mantel (r = R)", "bottom": "unten (z = 0)", "top": "oben (z = H)"}
SIDE_TAGS = {"left": 1, "right": 2, "bottom": 3, "top": 4}                 # hpfem.box_tag.X_MIN, X_MAX, Y_MIN, Y_MAX
SIDE_NORMAL = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}
WALL_BC = {"clamped": "eingespannt (u = 0 am Rand)", "free": "frei (Starrkörperbewegung entfernt)"}
FIELDS = {
    "T": ("Temperatur (K)", "inferno"), "Eabs": ("|E| / E₀ (Feldüberhöhung)", "viridis"), "umag": ("|u| (nm)", "viridis"), "ux": ("u_x bzw. u_r (nm)", "RdBu_r"),
    "uy": ("u_y bzw. u_z (nm)", "RdBu_r"), "svm": ("Von-Mises-Spannung (MPa)", "magma"), "sxx": ("σ_xx bzw. σ_rr (MPa)", "RdBu_r"),
    "syy": ("σ_yy bzw. σ_zz (MPa)", "RdBu_r"), "sxy": ("σ_xy bzw. σ_rz (MPa)", "RdBu_r"), "szz": ("σ_zz (eben) bzw. σ_θθ (axial) (MPa)", "RdBu_r"),
}

# Rough room-temperature handbook values (kappa W/(m K), E GPa, nu, alpha 1e-6/K) as starting values only: thin films, crystal orientation and
# temperature change them considerably. water / air / vacuum have no elastic stiffness (a fluid): E is a stiff placeholder and the app warns.
PROPS = {
    "Au": (315.0, 79.0, 0.42, 14.2), "Ag": (429.0, 83.0, 0.37, 18.9), "Al": (237.0, 70.0, 0.35, 23.1), "Ti": (21.9, 116.0, 0.32, 8.6),
    "Cr": (93.9, 279.0, 0.21, 4.9), "Si": (148.0, 160.0, 0.22, 2.6), "SiO2": (1.4, 73.0, 0.17, 0.5), "TiO2": (8.5, 230.0, 0.27, 8.8),
    "Al2O3": (30.0, 370.0, 0.22, 8.1), "GaAs": (55.0, 85.0, 0.31, 5.7), "MAPbI3": (0.5, 20.0, 0.30, 40.0),
    "water": (0.6, 1.0, 0.45, 0.0), "air": (0.026, 0.001, 0.30, 0.0), "vacuum": (0.0, 0.001, 0.30, 0.0),
}
FLUIDS = {"water", "air", "vacuum"}
EPS0, C0 = 8.8541878128e-12, 299792458.0


def default_props(spec):
    k, e, nu, al = PROPS.get(spec.get("name", "") if spec.get("type") == "library" else "", (1.0, 70.0, 0.3, 10.0))
    return {"kappa": k, "E_GPa": e, "nu": nu, "alpha": al, "deps_re": 0.0, "deps_im": 0.0}


def used_materials(m):
    if m["case"] == "wire":
        return {m["background"], m["inclusion"]}
    return {m["background"]}


# ===================================================================================================================== models
def _spec_index(n, k=0.0):
    return {"type": "index", "n": n, "k": k}


def default_model():
    """Gold nanowire r = 50 nm in glass at 520 nm (the plasmon resonance region), 1 MV/m, heat sink at 8 r."""
    mats = {"Gold": {"type": "library", "name": "Au"}, "Glas": {"type": "library", "name": "SiO2"}}
    return {
        "name": "Goldnanodraht in Glas: Erwärmung", "case": "wire", "materials": mats, "background": "Glas", "inclusion": "Gold",
        "props": {"Gold": default_props(mats["Gold"]), "Glas": default_props(mats["Glas"])},
        "optics": {"wavelength_nm": 520.0, "E0_MVm": 1.0},
        "wire": {"radius_nm": 50.0, "half_factor": 4, "outer_factor": 8, "cells_per_radius": 2},
        "block": {"width_um": 4.0, "height_um": 2.0, "nx": 8, "ny": 4},
        "thermal": {"T_ref": 300.0, "selfconsistent": False, "relaxation": 1.0, "tolerance": 1e-3, "max_iter": 20},
        "mech": {"on": False, "wall": "clamped", "bc": {s: {"kind": "free", "value": 0.0} for s in SIDE_TAGS}, "dT_a": 0.0, "dT_b": 0.0, "shrink_pct": 0.0},
        "order": 3, "grid": 120,
    }


def _block(name, bc, dT=(0.0, 0.0), shrink=0.0, case="block", w=4.0, h=2.0, nx=8, ny=4, material=("Material", {"type": "eps", "re": 2.25, "im": 0.0}, None), order=3):
    mat_name, spec, props = material
    m = default_model()
    m.update(name=name, case=case, materials={mat_name: spec}, background=mat_name, inclusion=mat_name, props={mat_name: props or default_props(spec)})
    m["block"] = {"width_um": w, "height_um": h, "nx": nx, "ny": ny}
    m["mech"] = {"on": True, "wall": "clamped", "bc": {s: dict(bc.get(s, {"kind": "free", "value": 0.0})) for s in SIDE_TAGS}, "dT_a": dT[0], "dT_b": dT[1], "shrink_pct": shrink}
    m["order"] = order
    return m


def presets():
    base = default_model()
    out = {base["name"]: base}
    stress = copy.deepcopy(base)
    stress.update(name="Goldnanodraht in Glas: Erwärmung und Wärmespannung")
    stress["mech"]["on"] = True
    out[stress["name"]] = stress
    sc = copy.deepcopy(base)
    sc.update(name="Goldnanodraht: selbstkonsistent mit dε/dT (Beispielwert)")
    sc["thermal"]["selfconsistent"] = True
    sc["props"]["Gold"].update(deps_re=-0.01, deps_im=0.0)
    out[sc["name"]] = sc
    steel = ("Stahl-ähnlich", {"type": "eps", "re": 1.0, "im": 0.0}, {"kappa": 50.0, "E_GPa": 200.0, "nu": 0.3, "alpha": 12.0, "deps_re": 0.0, "deps_im": 0.0})
    sym = {"left": {"kind": "symmetry", "value": 0.0}, "bottom": {"kind": "symmetry", "value": 0.0}}
    out["Block unter Zug (Prüfung gegen die geschlossene Lösung)"] = _block(
        "Block unter Zug (Prüfung gegen die geschlossene Lösung)", {**sym, "right": {"kind": "traction", "value": 100.0}}, material=steel)
    out["Block eingespannt, Erwärmung um 50 K"] = _block(
        "Block eingespannt, Erwärmung um 50 K", {"left": {"kind": "clamped", "value": 0.0}}, dT=(50.0, 50.0), material=steel)
    out["Bimetall-artiger Temperaturverlauf im Block"] = _block(
        "Bimetall-artiger Temperaturverlauf im Block", {"left": {"kind": "clamped", "value": 0.0}}, dT=(0.0, 80.0), material=steel)
    out["Zylinder unter Axialdruck (Prüfung gegen die geschlossene Lösung)"] = _block(
        "Zylinder unter Axialdruck (Prüfung gegen die geschlossene Lösung)",
        {"left": {"kind": "symmetry", "value": 0.0}, "bottom": {"kind": "symmetry", "value": 0.0}, "top": {"kind": "traction", "value": -100.0}},
        case="axi", w=1.0, h=2.0, nx=4, ny=8, material=steel)
    out["Zylinder, unten eingespannt, Schrumpfung 1 %"] = _block(
        "Zylinder, unten eingespannt, Schrumpfung 1 %", {"left": {"kind": "symmetry", "value": 0.0}, "bottom": {"kind": "clamped", "value": 0.0}},
        case="axi", w=1.0, h=2.0, nx=4, ny=8, shrink=1.0, material=("Harz", {"type": "eps", "re": 2.25, "im": 0.0},
                                                                        {"kappa": 0.2, "E_GPa": 4.5, "nu": 0.35, "alpha": 0.0, "deps_re": 0.0, "deps_im": 0.0}))
    return out


def normalise(m):
    """Fill keys that older or hand-made models miss."""
    d = default_model()
    for k, v in d.items():
        if k not in m:
            m[k] = copy.deepcopy(v)
    for k in ("optics", "wire", "block", "thermal"):
        for kk, vv in d[k].items():
            m[k].setdefault(kk, vv)
    for kk, vv in d["mech"].items():
        m["mech"].setdefault(kk, copy.deepcopy(vv))
    for s in SIDE_TAGS:
        m["mech"]["bc"].setdefault(s, {"kind": "free", "value": 0.0})
    for n, spec in m["materials"].items():
        m["props"].setdefault(n, default_props(spec))
    return m


def sides(m):
    return SIDES_AXI if m["case"] == "axi" else SIDES_BLOCK


def intensity_W_m2(m):
    """Intensity of the incident wave 1/2 n eps0 c E0^2 in the (lossless) matrix; n from eps of the matrix at the wavelength."""
    import fem_materials as fm

    try:
        n = math.sqrt(max(float(fm.eps_at(m["materials"][m["background"]], m["optics"]["wavelength_nm"]).real), 0.0))
    except Exception:
        return float("nan")
    return 0.5 * n * EPS0 * C0 * (m["optics"]["E0_MVm"] * 1e6) ** 2


def validate(m):
    """[(level, text)] with level 'error' or 'warn'."""
    import fem_materials as fm

    out = []
    names = set(m["materials"])
    for key in ("background", "inclusion"):
        if m[key] not in names:
            out.append(("error", f"Material „{m[key]}“ fehlt."))
    if out:
        return out
    case = m["case"]
    if case == "wire":
        w = m["wire"]
        if w["radius_nm"] <= 0:
            out.append(("error", "Der Radius muss positiv sein."))
        if not (isinstance(w["half_factor"], int) and isinstance(w["outer_factor"], int)) or w["half_factor"] < 2 or w["outer_factor"] <= w["half_factor"]:
            out.append(("error", "Gebiet: ganzzahlige Faktoren mit 2 ≤ Innenbreite/Radius < Außenbreite/Radius (der Rest ist die PML-Schicht)."))
        elif w["outer_factor"] - w["half_factor"] < 2:
            out.append(("warn", "Die PML-Schicht ist dünner als zwei Radien: Reflexionen wahrscheinlich."))
        if m["optics"]["wavelength_nm"] <= 0 or m["optics"]["E0_MVm"] <= 0:
            out.append(("error", "Wellenlänge und Feldamplitude müssen positiv sein."))
        try:
            bg = fm.eps_at(m["materials"][m["background"]], m["optics"]["wavelength_nm"])
            if abs(bg.imag) > 1e-9:
                out.append(("error", "Das Matrixmaterial muss verlustfrei sein (Im ε = 0): die einfallende Welle ist eine ebene Welle im Hintergrund."))
            if fm.is_crystal(m["materials"][m["background"]]) or fm.is_crystal(m["materials"][m["inclusion"]]):
                out.append(("error", "Kristalle (anisotrop) gibt es in diesem Modus nicht."))
        except Exception as exc:
            out.append(("error", f"Material bei λ nicht auswertbar: {exc}"))
        if m["thermal"]["selfconsistent"] and not (0 < m["thermal"]["relaxation"] <= 1):
            out.append(("error", "Relaxation zwischen 0 (ausgeschlossen) und 1."))
        if m["thermal"]["selfconsistent"] and not any(abs(m["props"][n]["deps_re"]) + abs(m["props"][n]["deps_im"]) > 0 for n in used_materials(m)):
            out.append(("warn", "Selbstkonsistent ohne dε/dT: die Schleife bricht nach einem Durchlauf ab (Ergebnis wie ohne Rückkopplung)."))
        if m["props"][m["background"]]["kappa"] <= 0 or m["props"][m["inclusion"]]["kappa"] <= 0:
            out.append(("error", "Die Wärmeleitfähigkeit muss positiv sein."))
    else:
        b = m["block"]
        if b["width_um"] <= 0 or b["height_um"] <= 0 or b["nx"] < 1 or b["ny"] < 1:
            out.append(("error", "Abmessungen und Zellzahlen müssen positiv sein."))
    mech = m["mech"]
    if case != "wire" or mech["on"]:
        names_used = [m["background"]] if case != "wire" else [m["background"], m["inclusion"]]
        for n in names_used:
            p = m["props"][n]
            if p["E_GPa"] <= 0 or not (-0.99 < p["nu"] < 0.5):
                out.append(("error", f"{n}: E > 0 und −1 < ν < 0,5 nötig."))
            if m["materials"][n].get("type") == "library" and m["materials"][n].get("name") in FLUIDS:
                out.append(("warn", f"{n} ist ein Fluid: es hat keine Schubsteifigkeit, die elastischen Werte sind nur ein Platzhalter."))
        if case != "wire":
            bc = mech["bc"]
            kinds = [bc[s]["kind"] for s in SIDE_TAGS]
            fixed = any(k in ("clamped", "symmetry") for k in kinds)
            loaded = any(bc[s]["kind"] == "traction" and bc[s]["value"] != 0 for s in SIDE_TAGS)
            if not fixed and loaded:
                out.append(("error", "Alle Seiten frei oder belastet: die Last muss im Gleichgewicht sein, sonst weist der Löser sie ab. Mindestens eine Seite festhalten oder eine Symmetrie wählen."))
            if case == "axi" and bc["left"]["kind"] in ("clamped", "traction"):
                out.append(("warn", "Die Achse hat immer u_r = 0 (die Bibliothek setzt es); „eingespannt“ ändert dort u_z, eine Last auf der Achse gibt es nicht."))
            if not fixed and not loaded:
                out.append(("warn", "Alles frei und ohne Last: nur Temperatur und Schrumpfung, der Körper dehnt sich frei (Starrkörperbewegung wird entfernt)."))
    return out


def analytic_checks(m, summary=None):
    """Closed-form reference values for the simple block / cylinder setups: [(label, expected, key)] where key looks up the value in the results'
    ``probes``. Only when the setup is exactly the textbook one (uniaxial traction or free expansion with the symmetry planes), else []."""
    if m["case"] == "wire" or not m["mech"]["on"]:
        return []
    bc, p = m["mech"]["bc"], m["props"][m["background"]]
    E, nu, al = p["E_GPa"] * 1e9, p["nu"], p["alpha"] * 1e-6
    b = m["block"]
    L1, L2 = b["width_um"] * 1e-6, b["height_um"] * 1e-6
    k = {s: bc[s]["kind"] for s in SIDE_TAGS}
    dTa, dTb = m["mech"]["dT_a"], m["mech"]["dT_b"]
    shrink = m["mech"]["shrink_pct"] / 100.0
    if dTa != dTb:
        return []
    e_th = al * dTa - shrink                                                   # eigenstrain along the axes (in-plane, hoop for axi)
    out = []
    if m["case"] == "block":
        if k["left"] == "symmetry" and k["bottom"] == "symmetry" and k["top"] == "free" and k["right"] in ("traction", "free"):
            s = bc["right"]["value"] * 1e6 if k["right"] == "traction" else 0.0
            out.append(("u_x(W, H/2), m", (1 - nu**2) * s * L1 / E + e_th * L1, "ux_end"))
            out.append(("u_y(W/2, H), m", -nu * (1 + nu) * s * L2 / E + e_th * L2, "uy_top"))
    else:
        if k["left"] in ("symmetry", "free") and k["bottom"] == "symmetry" and k["right"] == "free" and k["top"] in ("traction", "free"):
            s = bc["top"]["value"] * 1e6 if k["top"] == "traction" else 0.0
            out.append(("u_z(0, H), m", s * L2 / E + e_th * L2, "uy_top"))
            out.append(("u_r(R, H/2), m", -nu * s * L1 / E + e_th * L1, "ux_end"))
    return out


def heat_line_estimate(m, power_w_per_m):
    """Plausibility check of the wire temperature: the temperature rise of a thin line source (isothermal wire, power P per metre) in a matrix of
    conductivity kappa with a square heat sink of half-width ``a``: dT = P / (2 pi kappa) ln(R_eff / r) with R_eff = 1.0787 a (equivalent radius
    of a square). Only meaningful without feedback and when the wire conducts much better than the matrix; the wire's own gradient is neglected."""
    w = m["wire"]
    kappa = m["props"][m["background"]]["kappa"]
    r, a = w["radius_nm"], w["outer_factor"] * w["radius_nm"]
    if kappa <= 0 or power_w_per_m <= 0:
        return float("nan")
    return power_w_per_m / (2 * math.pi * kappa) * math.log(1.0787 * a / r)


def rename_material(m, old, new):
    m["materials"] = {(new if k == old else k): v for k, v in m["materials"].items()}
    m["props"] = {(new if k == old else k): v for k, v in m["props"].items()}
    for key in ("background", "inclusion"):
        if m[key] == old:
            m[key] = new


# ===================================================================================================================== results
def summary_table(res):
    rows = [(k, v) for k, v in res.get("scalars", {}).items()]
    return pd.DataFrame(rows, columns=["Größe", "Wert"])


def history_dataframe(res):
    h = res.get("history") or []
    return pd.DataFrame({"Iteration": list(range(1, len(h) + 1)), "größte Temperaturänderung (K)": h})


def field_keys(res_fields):
    return [k for k in FIELDS if k in res_fields]


# ===================================================================================================================== figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_preview(m):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    if m["case"] == "wire":
        w = m["wire"]
        r = w["radius_nm"]
        h, o = w["half_factor"] * r, w["outer_factor"] * r
        ax.add_patch(plt.Rectangle((-o, -o), 2 * o, 2 * o, fc="#e8e8e8", ec="k", lw=1.0))
        ax.add_patch(plt.Rectangle((-h, -h), 2 * h, 2 * h, fc="#f7f7f7", ec="#888", lw=0.8, ls="--"))
        ax.add_patch(plt.Circle((0, 0), r, fc="#e0a800", ec="k"))
        ax.text(0, 0, m["inclusion"], ha="center", va="center", fontsize=8)
        ax.text(-o * 0.97, o * 0.9, "PML (außen)", fontsize=8, color="#555")
        ax.text(-h * 0.95, h * 0.85, m["background"], fontsize=8)
        ax.annotate("", xy=(-o * 0.6, -o * 0.5), xytext=(-o * 0.95, -o * 0.5), arrowprops=dict(arrowstyle="->", color="#2060c0"))
        ax.text(-o * 0.95, -o * 0.42, f"Welle, E ∥ y, λ = {m['optics']['wavelength_nm']:g} nm", fontsize=8, color="#2060c0")
        ax.set(xlim=(-o * 1.05, o * 1.05), ylim=(-o * 1.05, o * 1.05), xlabel="x (nm)", ylabel="y (nm)", title="Rechengebiet: Wärmesenke T = T_ref am äußeren Rand")
    else:
        b = m["block"]
        W, H = b["width_um"], b["height_um"]
        axi = m["case"] == "axi"
        ax.add_patch(plt.Rectangle((0, 0), W, H, fc="#e8e8e8", ec="k"))
        names = sides(m)
        mid = {"left": (0, H / 2), "right": (W, H / 2), "bottom": (W / 2, 0), "top": (W / 2, H)}
        for s, (x, y) in mid.items():
            c = m["mech"]["bc"][s]
            lab = {"free": "", "clamped": "fest", "symmetry": "Symm.", "traction": f"σ = {c['value']:g} MPa"}[c["kind"]]
            col = {"free": "#aaa", "clamped": "#c03030", "symmetry": "#2060c0", "traction": "#208040"}[c["kind"]]
            ax.plot(*zip(*{"left": [(0, 0), (0, H)], "right": [(W, 0), (W, H)], "bottom": [(0, 0), (W, 0)], "top": [(0, H), (W, H)]}[s]), color=col, lw=3)
            dx = {"left": -0.02, "right": 0.02}.get(s, 0) * W
            dy = {"bottom": -0.06, "top": 0.04}.get(s, 0) * H
            ax.text(x + dx, y + dy, lab, fontsize=8, ha="right" if s == "left" else ("left" if s == "right" else "center"), va="center", color=col)
        ax.set(xlim=(-0.2 * W, 1.25 * W), ylim=(-0.15 * H, 1.15 * H), xlabel=("r (µm)" if axi else "x (µm)"), ylabel=("z (µm)" if axi else "y (µm)"),
               title=" · ".join(f"{names[s].split(' ')[0]}: {BC_KINDS[m['mech']['bc'][s]['kind']].split(' ')[0]}" for s in SIDE_TAGS)[:90])
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def fig_field(npz, key, unit_lab=None, extent_scale=1.0, mesh_overlay=False, deformed=None):
    """Raster map of a grid field (NaN outside the mesh); lengths in the axes labels follow ``npz['unit']``."""
    plt = _plt()
    x, y, f = npz["x"], npz["y"], npz[key]
    title, cmap = FIELDS[key]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sym = key in ("ux", "uy", "sxx", "syy", "sxy", "szz")
    vmax = float(np.nanmax(np.abs(f))) if np.isfinite(f).any() else 1.0
    kw = dict(vmin=-vmax, vmax=vmax) if sym and vmax > 0 else {}
    im = ax.pcolormesh(x, y, f, cmap=cmap, shading="auto", **kw)
    fig.colorbar(im, ax=ax, label=title)
    unit = str(npz["unit"])
    ax.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})", title=title)
    ax.set_aspect("equal")
    if "circle_r" in npz.files and float(npz["circle_r"]) > 0:
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(float(npz["circle_r"]) * np.cos(t), float(npz["circle_r"]) * np.sin(t), "w-", lw=0.8, alpha=0.8)
    fig.tight_layout()
    return fig


def fig_deformed(npz, scale):
    """Mesh and its deformation scaled by ``scale`` (original in grey)."""
    plt = _plt()
    import matplotlib.tri as mtri

    v, c, d = npz["mesh_vertices"], npz["mesh_cells"], npz["mesh_disp"]
    unit = str(npz["unit"])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.triplot(mtri.Triangulation(v[:, 0], v[:, 1], c), color="#bbbbbb", lw=0.4)
    vd = v + scale * d
    ax.triplot(mtri.Triangulation(vd[:, 0], vd[:, 1], c), color="#c03030", lw=0.5)
    ax.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})", title=f"Verformtes Netz (Verschiebung × {scale:g})")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def fig_history(history):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy(range(1, len(history) + 1), np.maximum(history, 1e-300), "o-")
    ax.set(xlabel="Iteration", ylabel="größte Temperaturänderung (K)", title="Konvergenz der thermo-optischen Schleife")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_line(npz, key, axis="x", frac=0.5):
    """Cut through a grid field along x (at the height fraction) or y."""
    plt = _plt()
    x, y, f = npz["x"], npz["y"], npz[key]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    if axis == "x":
        j = int(round(frac * (len(y) - 1)))
        ax.plot(x, f[j, :])
        ax.set_xlabel(f"x ({npz['unit']}) bei y = {y[j]:.4g}")
    else:
        i = int(round(frac * (len(x) - 1)))
        ax.plot(y, f[:, i])
        ax.set_xlabel(f"y ({npz['unit']}) bei x = {x[i]:.4g}")
    ax.set_ylabel(FIELDS[key][0])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
