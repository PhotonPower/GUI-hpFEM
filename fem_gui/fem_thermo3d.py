"""Heat and mechanics in 3D for the FEM model builder, hp-FEM 0.5, without hpfem and without Streamlit: models, presets, validation, closed-form checks,
estimates, result handling and figures. Solvers: hpfem ``Scattering3D`` / ``ThermoOptical3D`` / ``Thermal3D`` / ``Elasticity3D`` (worker: fem_thermo3d_worker.py).
The 2D module fem_thermo.py supplies the material data (``PROPS``, ``default_props``), the boundary-condition kinds and the intensity formula.

Two cases:

* ``particle``: a sphere (tag 2) in a matrix (tag 0) in a box with PML (``hpfem.box_with_ball``). Chain: plane wave (E along x, along +z) -> absorbed power
  -> temperature (heat sink T = T_ref on the outer faces of the box) -> optional feedback eps(T) = eps + (d eps/dT)(T - T_ref) (``ThermoOptical3D``) -> optional
  thermal stress (``Elasticity3D``; outer box faces clamped or free). Optionally the quarter model x > 0, y > 0 (mirror symmetry, as in fem_threed).
* ``block``: box [0, W] x [0, D] x [0, H], boundary conditions per face (free, clamped, symmetry, traction), temperature change and shrinkage linear along z, or
  a volumetric heat source with heat sinks on chosen faces (``Thermal3D``, the other faces adiabatic) whose temperature drives the thermal expansion.

Units in the model: nm (particle), um (block), MV/m, K, W/(m K), GPa, 1e-6/K, MPa, mW/um^3 (heat source, 1 mW/um^3 = 1e15 W/m^3). The solver gets SI.
Conventions: time factor exp(-i omega t), Im eps > 0 loss. In 3D the thermal strain alpha (T - T_ref) acts in all three directions.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

import fem_thermo as ft

CASES = {"particle": "Kugel: Optik → Wärme → Spannung", "block": "Quader: Wärmeleitung und Mechanik"}
SIDES = {"xmin": "x = 0", "xmax": "x = W", "ymin": "y = 0", "ymax": "y = D", "zmin": "z = 0", "zmax": "z = H"}
SIDE_TAGS = {"xmin": 1, "xmax": 2, "ymin": 3, "ymax": 4, "zmin": 5, "zmax": 6}      # hpfem.box_tag.X_MIN ... Z_MAX
SIDE_NORMAL = {"xmin": (-1.0, 0.0, 0.0), "xmax": (1.0, 0.0, 0.0), "ymin": (0.0, -1.0, 0.0), "ymax": (0.0, 1.0, 0.0), "zmin": (0.0, 0.0, -1.0), "zmax": (0.0, 0.0, 1.0)}
BC_KINDS = ft.BC_KINDS
WALL_BC = {"clamped": "eingespannt (u = 0 an den Außenflächen)", "free": "frei (Starrkörperbewegung entfernt)"}
FIELDS = {
    "T": ("Temperatur (K)", "inferno"), "Eabs": ("|E| / E₀ (Feldüberhöhung)", "viridis"), "umag": ("|u| (nm)", "viridis"), "ux": ("u_x (nm)", "RdBu_r"),
    "uy": ("u_y (nm)", "RdBu_r"), "uz": ("u_z (nm)", "RdBu_r"), "svm": ("Von-Mises-Spannung (MPa)", "magma"), "sxx": ("σ_xx (MPa)", "RdBu_r"),
    "syy": ("σ_yy (MPa)", "RdBu_r"), "szz": ("σ_zz (MPa)", "RdBu_r"), "sxy": ("σ_xy (MPa)", "RdBu_r"), "syz": ("σ_yz (MPa)", "RdBu_r"), "sxz": ("σ_xz (MPa)", "RdBu_r"),
}
SYMMETRIC_FIELDS = ("ux", "uy", "uz", "sxx", "syy", "szz", "sxy", "syz", "sxz")
EQUIV_CUBE = 0.6606785                    # capacity of a cube of edge L is that of a sphere of radius EQUIV_CUBE * L (used for the plausibility check)
PML_THICK_MIN = 1.5                       # radii; thinner PML layers are warned about


# ===================================================================================================================== estimates
def mech_estimate(h1_dofs):
    """Rough single-thread SparseLU cost of Elasticity3D from measurements on a box (p = 2 and 3, 2 187 to 14 739 unknowns, polymer or steel data):
    seconds ~ 2 (N / 2187)^1.8, peak MB ~ 45 + 54 (N / 2187)^1.2 with N = 3 x the H1 unknowns. A guide, not a bound (other meshes factor differently)."""
    n = 3.0 * float(h1_dofs)
    ratio = max(n / 2187.0, 1e-9)
    return dict(dofs=int(n), seconds=2.0 * ratio**1.8, bytes=(45.0 + 54.0 * ratio**1.2) * 1024.0 * 1024.0)


def available_gb():
    """Free main memory in GB (MemAvailable), None if unknown."""
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return None


# ===================================================================================================================== models
def _side_dict(default=None):
    return {s: dict(default or {"kind": "free", "value": 0.0}) for s in SIDE_TAGS}


def default_model():
    """Gold sphere r = 50 nm in glass at 520 nm, 1 MV/m, heat sink at 5 r, quarter model, coarse mesh (n = 1, p = 2): the plasmon region, a few seconds of solving."""
    mats = {"Gold": {"type": "library", "name": "Au"}, "Glas": {"type": "library", "name": "SiO2"}}
    return {
        "name": "Goldkugel in Glas: Erwärmung", "dim": 3, "case": "particle", "materials": mats, "background": "Glas", "inclusion": "Gold",
        "props": {"Gold": ft.default_props(mats["Gold"]), "Glas": ft.default_props(mats["Glas"])},
        "optics": {"wavelength_nm": 520.0, "E0_MVm": 1.0},
        "particle": {"radius_nm": 50.0, "half_factor": 3, "outer_factor": 5, "n": 1, "quarter": True},
        "block": {"W_um": 4.0, "D_um": 2.0, "H_um": 1.0, "nx": 4, "ny": 2, "nz": 2},
        "thermal": {"T_ref": 300.0, "selfconsistent": False, "relaxation": 1.0, "tolerance": 1e-3, "max_iter": 20, "heat": False, "q_mWum3": 1.0,
                    "sinks": {"xmin": True, "xmax": True, "ymin": False, "ymax": False, "zmin": False, "zmax": False}},
        "mech": {"on": False, "wall": "clamped", "bc": _side_dict(), "dT_a": 0.0, "dT_b": 0.0, "shrink_a_pct": 0.0, "shrink_b_pct": 0.0},
        "order": 2, "grid": 36, "max_gb": 8.0,
    }


def _block(name, bc, dT=(0.0, 0.0), shrink=(0.0, 0.0), heat=False, sinks=("xmin", "xmax"), q=1.0, size=(4.0, 2.0, 1.0), cells=(4, 2, 2),
           material=("Material", {"type": "eps", "re": 2.25, "im": 0.0}, None), order=2):
    mat_name, spec, props = material
    m = default_model()
    m.update(name=name, case="block", materials={mat_name: spec}, background=mat_name, inclusion=mat_name, props={mat_name: props or ft.default_props(spec)})
    m["block"] = dict(W_um=size[0], D_um=size[1], H_um=size[2], nx=cells[0], ny=cells[1], nz=cells[2])
    m["thermal"].update(heat=heat, q_mWum3=q, sinks={s: s in sinks for s in SIDE_TAGS})
    m["mech"] = {"on": True, "wall": "clamped", "bc": {s: dict(bc.get(s, {"kind": "free", "value": 0.0})) for s in SIDE_TAGS}, "dT_a": dT[0], "dT_b": dT[1],
                 "shrink_a_pct": shrink[0], "shrink_b_pct": shrink[1]}
    m["order"] = order
    return m


_STEEL = ("Stahl-ähnlich", {"type": "eps", "re": 1.0, "im": 0.0}, {"kappa": 50.0, "E_GPa": 200.0, "nu": 0.3, "alpha": 12.0, "deps_re": 0.0, "deps_im": 0.0})
_RESIN = ("Harz", {"type": "eps", "re": 2.25, "im": 0.0}, {"kappa": 0.2, "E_GPa": 4.5, "nu": 0.35, "alpha": 0.0, "deps_re": 0.0, "deps_im": 0.0})
_SYM3 = {"xmin": {"kind": "symmetry", "value": 0.0}, "ymin": {"kind": "symmetry", "value": 0.0}, "zmin": {"kind": "symmetry", "value": 0.0}}


def presets():
    base = default_model()
    out = {base["name"]: base}
    stress = copy.deepcopy(base)
    stress.update(name="Goldkugel in Glas: Erwärmung und Wärmespannung")
    stress["mech"]["on"] = True
    out[stress["name"]] = stress
    sc = copy.deepcopy(base)
    sc.update(name="Goldkugel: selbstkonsistent mit dε/dT (Beispielwert)")
    sc["thermal"]["selfconsistent"] = True
    sc["props"]["Gold"].update(deps_re=-0.01, deps_im=0.0)
    out[sc["name"]] = sc
    tens = "Quader unter Zug (Prüfung gegen die geschlossene Lösung)"
    out[tens] = _block(tens, {**_SYM3, "xmax": {"kind": "traction", "value": 100.0}}, material=_STEEL)
    heat = "Quader: Wärmequelle zwischen zwei Senken (Prüfung gegen die geschlossene Lösung)"
    out[heat] = _block(heat, _SYM3, heat=True, q=1.0, material=_STEEL)
    hot = "Quader eingespannt, Erwärmung um 50 K"
    out[hot] = _block(hot, {"xmin": {"kind": "clamped", "value": 0.0}}, dT=(50.0, 50.0), material=_STEEL)
    shr = "Schrumpfblock auf dem Substrat (Viertelmodell)"
    out[shr] = _block(shr, {"xmin": {"kind": "symmetry", "value": 0.0}, "ymin": {"kind": "symmetry", "value": 0.0}, "zmin": {"kind": "clamped", "value": 0.0}},
                      shrink=(2.0, 4.0), size=(20.0, 20.0, 2.0), cells=(4, 4, 2), material=_RESIN)
    return out


def normalise(m):
    """Fill keys that older or hand-made models miss."""
    d = default_model()
    for k, v in d.items():
        if k not in m:
            m[k] = copy.deepcopy(v)
    for k in ("optics", "particle", "block", "thermal"):
        for kk, vv in d[k].items():
            m[k].setdefault(kk, copy.deepcopy(vv))
    for kk, vv in d["mech"].items():
        m["mech"].setdefault(kk, copy.deepcopy(vv))
    for s in SIDE_TAGS:
        m["mech"]["bc"].setdefault(s, {"kind": "free", "value": 0.0})
        m["thermal"]["sinks"].setdefault(s, False)
    for n, spec in m["materials"].items():
        m["props"].setdefault(n, ft.default_props(spec))
    return m


def used_materials(m):
    return {m["background"], m["inclusion"]} if m["case"] == "particle" else {m["background"]}


def block_size_m(m):
    b = m["block"]
    return b["W_um"] * 1e-6, b["D_um"] * 1e-6, b["H_um"] * 1e-6


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
    th, mech = m["thermal"], m["mech"]
    if int(m["order"]) < 1:
        out.append(("error", "Die Polynomordnung muss mindestens 1 sein."))
    if case == "particle":
        p = m["particle"]
        if p["radius_nm"] <= 0:
            out.append(("error", "Der Radius muss positiv sein."))
        if not (isinstance(p["half_factor"], int) and isinstance(p["outer_factor"], int)) or p["half_factor"] < 2 or p["outer_factor"] <= p["half_factor"]:
            out.append(("error", "Gebiet: ganzzahlige Faktoren mit 2 ≤ Innenbreite/Radius < Außenbreite/Radius (der Rest ist die PML-Schicht)."))
        elif p["outer_factor"] - p["half_factor"] < PML_THICK_MIN:
            out.append(("warn", "Die PML-Schicht ist dünner als anderthalb Radien: Reflexionen wahrscheinlich."))
        if int(p["n"]) < 1:
            out.append(("error", "Zellen je Radius: mindestens 1."))
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
        if th["selfconsistent"] and not (0 < th["relaxation"] <= 1):
            out.append(("error", "Relaxation zwischen 0 (ausgeschlossen) und 1."))
        if th["selfconsistent"] and not any(abs(m["props"][n]["deps_re"]) + abs(m["props"][n]["deps_im"]) > 0 for n in used_materials(m)):
            out.append(("warn", "Selbstkonsistent ohne dε/dT: die Schleife bricht nach einem Durchlauf ab (Ergebnis wie ohne Rückkopplung)."))
        if m["props"][m["background"]]["kappa"] <= 0 or m["props"][m["inclusion"]]["kappa"] <= 0:
            out.append(("error", "Die Wärmeleitfähigkeit muss positiv sein."))
        if th["selfconsistent"]:
            out.append(("warn", f"Selbstkonsistent: jede der bis zu {th['max_iter']} Iterationen löst die Optik neu (je Iteration etwa so lange wie ohne Rückkopplung)."))
    else:
        b = m["block"]
        if min(b["W_um"], b["D_um"], b["H_um"]) <= 0 or min(b["nx"], b["ny"], b["nz"]) < 1:
            out.append(("error", "Abmessungen und Zellzahlen müssen positiv sein."))
        if th["heat"]:
            if th["q_mWum3"] == 0:
                out.append(("warn", "Wärmequelle q = 0: die Temperatur bleibt überall T_ref."))
            if not any(th["sinks"].values()):
                out.append(("error", "Wärmequelle ohne Wärmesenke: die stationäre Gleichung hat dann keine Lösung. Mindestens eine Seite als Senke wählen."))
            if m["props"][m["background"]]["kappa"] <= 0:
                out.append(("error", "Die Wärmeleitfähigkeit muss positiv sein."))
            if mech["dT_a"] != 0 or mech["dT_b"] != 0:
                out.append(("warn", "Mit Wärmequelle gilt das berechnete Temperaturfeld; ΔT (linear) wird ignoriert."))
    if case == "block" or mech["on"]:
        names_used = [m["background"]] if case == "block" else [m["background"], m["inclusion"]]
        for n in names_used:
            p = m["props"][n]
            if p["E_GPa"] <= 0 or not (-0.99 < p["nu"] < 0.5):
                out.append(("error", f"{n}: E > 0 und −1 < ν < 0,5 nötig."))
            if m["materials"][n].get("type") == "library" and m["materials"][n].get("name") in ft.FLUIDS:
                out.append(("warn", f"{n} ist ein Fluid: es hat keine Schubsteifigkeit, die elastischen Werte sind nur ein Platzhalter."))
        if case == "block":
            bc = mech["bc"]
            kinds = [bc[s]["kind"] for s in SIDE_TAGS]
            fixed = any(k in ("clamped", "symmetry") for k in kinds)
            loaded = any(bc[s]["kind"] == "traction" and bc[s]["value"] != 0 for s in SIDE_TAGS)
            if not fixed and loaded:
                out.append(("error", "Alle Seiten frei oder belastet: die Last muss im Gleichgewicht sein, sonst weist der Löser sie ab. Mindestens eine Seite festhalten oder eine Symmetrie wählen."))
            if not fixed and not loaded:
                out.append(("warn", "Alles frei und ohne Last: nur Temperatur und Schrumpfung, der Körper dehnt sich frei (Starrkörperbewegung wird entfernt)."))
    return out


# ===================================================================================================================== closed-form checks
def analytic_checks(m):
    """[(label, expected, key)] for the textbook block setups; ``key`` looks up the value in the results' ``probes``. Uniaxial traction on x = W (or none) with the
    symmetry planes x = 0, y = 0, z = 0 and free other faces, uniform temperature change and shrinkage: all strains are uniform, so the closed form is exact
    in 3D. [] otherwise."""
    if m["case"] != "block" or not m["mech"]["on"]:
        return []
    bc, p, th, mech = m["mech"]["bc"], m["props"][m["background"]], m["thermal"], m["mech"]
    k = {s: bc[s]["kind"] for s in SIDE_TAGS}
    if th["heat"] or mech["dT_a"] != mech["dT_b"] or mech["shrink_a_pct"] != mech["shrink_b_pct"]:
        return []
    if not (k["xmin"] == "symmetry" and k["ymin"] == "symmetry" and k["zmin"] == "symmetry" and k["ymax"] == "free" and k["zmax"] == "free"
            and k["xmax"] in ("traction", "free")):
        return []
    E, nu, al = p["E_GPa"] * 1e9, p["nu"], p["alpha"] * 1e-6
    W, D, H = block_size_m(m)
    s = bc["xmax"]["value"] * 1e6 if k["xmax"] == "traction" else 0.0
    e_th = al * mech["dT_a"] - mech["shrink_a_pct"] / 100.0
    return [("u_x(W, D/2, H/2), m", (s / E + e_th) * W, "ux_end"), ("u_y(W/2, D, H/2), m", (-nu * s / E + e_th) * D, "uy_end"),
            ("u_z(W/2, D/2, H), m", (-nu * s / E + e_th) * H, "uz_end")]


def heat_checks(m):
    """Closed forms for the block with a uniform source q between the sink faces x = 0 / x = W (other faces adiabatic, so the problem is one-dimensional):
    both sinks T_max = T_ref + q W^2 / (8 kappa) at x = W/2; only x = 0: T_max = T_ref + q W^2 / (2 kappa) at x = W. [(label, expected, key)] or []."""
    th = m["thermal"]
    if m["case"] != "block" or not th["heat"]:
        return []
    sinks = {s for s, v in th["sinks"].items() if v}
    kappa = m["props"][m["background"]]["kappa"]
    q = th["q_mWum3"] * 1e15
    W = block_size_m(m)[0]
    if kappa <= 0:
        return []
    if sinks == {"xmin", "xmax"}:
        return [("T(W/2, D/2, H/2), K", th["T_ref"] + q * W * W / (8 * kappa), "T_mid")]
    if sinks == {"xmin"}:
        return [("T(W, D/2, H/2), K", th["T_ref"] + q * W * W / (2 * kappa), "T_end")]
    return []


def particle_heat_estimate(m, power_full_w):
    """Plausibility value of the sphere temperature rise: a small isothermal sphere (radius R, power P) in a matrix (kappa) with the heat sink on a surface that
    acts like a sphere of radius R_s: dT = P / (4 pi kappa) (1/R - 1/R_s). The cube of half-width a (the outer box) has the capacity radius 0.6607 * 2 a. Valid for a
    well-conducting sphere without feedback; the cube-as-sphere substitution makes it an estimate (a few percent)."""
    p = m["particle"]
    kappa = m["props"][m["background"]]["kappa"]
    r, a = p["radius_nm"] * 1e-9, p["outer_factor"] * p["radius_nm"] * 1e-9
    if kappa <= 0 or power_full_w <= 0:
        return float("nan")
    return power_full_w / (4 * math.pi * kappa) * (1.0 / r - 1.0 / (2 * EQUIV_CUBE * a))


def rename_material(m, old, new):
    ft.rename_material(m, old, new)


# ===================================================================================================================== job
def job(m, max_gb=None):
    j = copy.deepcopy(m)
    j["max_gb"] = float(max_gb if max_gb is not None else m.get("max_gb", 8.0))
    return j


# ===================================================================================================================== results
def summary_table(res):
    return pd.DataFrame([(k, v) for k, v in res.get("scalars", {}).items()], columns=["Größe", "Wert"])


def field_keys(files):
    return [k for k in FIELDS if k in files]


def _unit_label(npz):
    u = str(npz["unit"])
    return "µm" if u == "um" else u


# ===================================================================================================================== figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_preview(m):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    if m["case"] == "particle":
        p = m["particle"]
        r = p["radius_nm"]
        h, o = p["half_factor"] * r, p["outer_factor"] * r
        ax.add_patch(plt.Rectangle((-o, -o), 2 * o, 2 * o, fc="#e8e8e8", ec="k", lw=1.0))
        ax.add_patch(plt.Rectangle((-h, -h), 2 * h, 2 * h, fc="#f7f7f7", ec="#888", lw=0.8, ls="--"))
        ax.add_patch(plt.Circle((0, 0), r, fc="#e0a800", ec="k"))
        ax.text(0, 0, m["inclusion"], ha="center", va="center", fontsize=8)
        ax.text(-o * 0.97, o * 0.9, "PML (außen)", fontsize=8, color="#555")
        ax.text(-h * 0.95, h * 0.85, m["background"], fontsize=8)
        ax.annotate("", xy=(-o * 0.5, -o * 0.55), xytext=(-o * 0.5, -o * 0.95), arrowprops=dict(arrowstyle="->", color="#2060c0"))
        ax.text(-o * 0.45, -o * 0.8, f"Welle in +z, E ∥ x, λ = {m['optics']['wavelength_nm']:g} nm", fontsize=8, color="#2060c0")
        if p["quarter"]:
            ax.axvline(0, color="#c03030", lw=1.2)
            ax.text(o * 0.02, o * 0.9, "Viertelmodell: x > 0, y > 0 (Schnitt y = 0)", fontsize=7, color="#c03030")
        ax.set(xlim=(-o * 1.05, o * 1.05), ylim=(-o * 1.05, o * 1.05), xlabel="x (nm)", ylabel="z (nm)", title="Schnitt y = 0; Wärmesenke T = T_ref an den Außenflächen")
    else:
        b = m["block"]
        W, H = b["W_um"], b["H_um"]
        ax.add_patch(plt.Rectangle((0, 0), W, H, fc="#e8e8e8", ec="k"))
        col = {"free": "#aaa", "clamped": "#c03030", "symmetry": "#2060c0", "traction": "#208040"}
        pos = {"xmin": [(0, 0), (0, H)], "xmax": [(W, 0), (W, H)], "zmin": [(0, 0), (W, 0)], "zmax": [(0, H), (W, H)]}
        mid = {"xmin": (0, H / 2), "xmax": (W, H / 2), "zmin": (W / 2, 0), "zmax": (W / 2, H)}
        for s, pts in pos.items():
            c = m["mech"]["bc"][s]
            lab = {"free": "", "clamped": "fest", "symmetry": "Symm.", "traction": f"σ = {c['value']:g} MPa"}[c["kind"]]
            ax.plot(*zip(*pts), color=col[c["kind"]], lw=3)
            x, y = mid[s]
            dx = {"xmin": -0.02, "xmax": 0.02}.get(s, 0) * W
            dy = {"zmin": -0.06, "zmax": 0.04}.get(s, 0) * H
            ax.text(x + dx, y + dy, lab + (" · Senke" if m["thermal"]["heat"] and m["thermal"]["sinks"].get(s) else ""), fontsize=8,
                    ha="right" if s == "xmin" else ("left" if s == "xmax" else "center"), va="center", color=col[c["kind"]])
        ylab = ", ".join(f"y-{'min' if s == 'ymin' else 'max'}: {BC_KINDS[m['mech']['bc'][s]['kind']].split(' ')[0]}" for s in ("ymin", "ymax"))
        ax.set(xlim=(-0.25 * W, 1.35 * W), ylim=(-0.25 * H, 1.25 * H), xlabel="x (µm)", ylabel="z (µm)", title=f"Ansicht x–z · {ylab}"[:90])
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def _axes(npz):
    return {"x": npz["x"], "y": npz["y"], "z": npz["z"]}


def slice_of(npz, key, axis, frac):
    """(array in plot order, horizontal axis name, vertical axis name, coordinate of the cut): cut perpendicular to ``axis`` at the grid plane nearest to the
    fraction ``frac`` of the extent. x-cut -> (y, z), y-cut -> (x, z), z-cut -> (x, y)."""
    ax_ = _axes(npz)
    c = ax_[axis]
    i = int(round(frac * (len(c) - 1)))
    f = npz[key]
    if axis == "x":
        return f[i, :, :].T, "y", "z", float(c[i])
    if axis == "y":
        return f[:, i, :].T, "x", "z", float(c[i])
    return f[:, :, i].T, "x", "y", float(c[i])


def fig_slice(npz, key, axis="y", frac=0.5):
    """Colour map of a 3D grid field on the plane perpendicular to ``axis`` (NaN outside the mesh is left empty); the sphere's outline for the particle case."""
    plt = _plt()
    f, hn, vn, c0 = slice_of(npz, key, axis, frac)
    ax_ = _axes(npz)
    title, cmap = FIELDS[key]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    full = npz[key]
    sym = key in SYMMETRIC_FIELDS
    vmax = float(np.nanmax(np.abs(full))) if np.isfinite(full).any() else 1.0
    kw = dict(vmin=-vmax, vmax=vmax) if sym and vmax > 0 else {}
    im = ax.pcolormesh(ax_[hn], ax_[vn], f, cmap=cmap, shading="auto", **kw)
    fig.colorbar(im, ax=ax, label=title)
    unit = _unit_label(npz)
    ax.set(xlabel=f"{hn} ({unit})", ylabel=f"{vn} ({unit})", title=f"{title} · Schnitt {axis} = {c0:.4g} {unit}")
    ax.set_aspect("equal")
    r = float(npz["sphere_r"]) if "sphere_r" in npz.files else 0.0
    if r > 0 and abs(c0) < r:
        rr = math.sqrt(r * r - c0 * c0)
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rr * np.cos(t), rr * np.sin(t), "w-", lw=0.8, alpha=0.85)        # a plane cut of a sphere about the origin is a circle about the origin
    fig.tight_layout()
    return fig


def fig_line(npz, key, axis="x", fa=0.5, fb=0.5):
    """Values along ``axis`` through the grid point nearest to the fractions (fa, fb) of the two other axes (in the order x, y, z without ``axis``)."""
    plt = _plt()
    ax_ = _axes(npz)
    others = [a for a in "xyz" if a != axis]
    ia = int(round(fa * (len(ax_[others[0]]) - 1)))
    ib = int(round(fb * (len(ax_[others[1]]) - 1)))
    f = npz[key]
    idx = {others[0]: ia, others[1]: ib}
    sl = tuple(slice(None) if a == axis else idx[a] for a in "xyz")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ax_[axis], f[sl])
    unit = _unit_label(npz)
    ax.set_xlabel(f"{axis} ({unit}) bei {others[0]} = {ax_[others[0]][ia]:.4g}, {others[1]} = {ax_[others[1]][ib]:.4g}")
    ax.set_ylabel(FIELDS[key][0])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_deformed(npz, axis, frac, scale):
    """Sample grid of the cut plane before (grey) and after (red) the displacement, in-plane components times ``scale``."""
    plt = _plt()
    ax_ = _axes(npz)
    ux, hn, vn, c0 = slice_of(npz, "u" + {"x": "y", "y": "x", "z": "x"}[axis], axis, frac)
    uv, _, _, _ = slice_of(npz, "u" + {"x": "z", "y": "z", "z": "y"}[axis], axis, frac)
    H, V = np.meshgrid(ax_[hn], ax_[vn])
    # the stored displacements are in nm, the axes in the unit of the grid
    conv = 1.0 / (1.0 if str(npz["unit"]) == "nm" else 1e3)
    Hd, Vd = H + scale * ux * conv, V + scale * uv * conv
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    step = max(1, len(ax_[hn]) // 24)
    stepv = max(1, len(ax_[vn]) // 24)
    for arrs, col, lw in (((H, V), "#bbbbbb", 0.5), ((Hd, Vd), "#c03030", 0.6)):
        a, b = arrs
        for j in range(0, a.shape[0], stepv):
            ax.plot(a[j, :], b[j, :], color=col, lw=lw)
        for i in range(0, a.shape[1], step):
            ax.plot(a[:, i], b[:, i], color=col, lw=lw)
    unit = _unit_label(npz)
    ax.set(xlabel=f"{hn} ({unit})", ylabel=f"{vn} ({unit})", title=f"Verformtes Raster, Schnitt {axis} = {c0:.4g} {unit} (Verschiebung × {scale:g})")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def fig_history(history):
    return ft.fig_history(history)
