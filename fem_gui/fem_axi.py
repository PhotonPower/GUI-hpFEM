"""Bodies of revolution (2.5D) for the FEM model builder: resonators and emitters on the meridian plane (x = r, y = z).

The structure is rotationally symmetric about the z axis; every azimuthal order m is a 2D problem on the meridian half-plane r >= 0
(hpfem.AxisymmetricResonance / AxisymmetricScattering, ADR-0010 of hp-FEM). This module needs no hpfem: model, presets, checks, layout,
preview, the Gmsh meridian mesh and the figures. The solver side is fem_axi_worker.py.

Model (JSON dict, lengths in nm):
    kind = "axi", name, materials (as in fem_materials), background (surrounding lossless medium), substrate (material of the half-space
    z < 0 or None), layers (planar layers {material, z_bottom, height} that run radially through the PML to the outer wall, like the
    substrate: Bragg mirrors, membranes, slab waveguides; a later layer covers earlier ones and the substrate), parts (drawn in order, a later
    part covers earlier ones, the layers and the substrate):
        cylinder   r_inner, radius, z_bottom, height            (disc, pillar layer, ring with r_inner > 0)
        cone       r_bottom, r_top, z_bottom, height             (frustum on the axis)
        sphere     z_center, radius                              (on the axis)
        ellipsoid  z_center, r_semi, z_semi                      (spheroid on the axis)
        torus      r_center, z_center, radius                    (ring with circular cross-section, r_center > radius)
        polygon    points [[r, z], ...]                          (any cross-section, r >= 0)
    domain = {margin_nm, pml_nm}: distance from the structure to the PML and its thickness (PML on the outer side, at the top and bottom).
    resonance = {wavelength_nm, m, num_modes}
    emitter = {z_nm, orientation ("transverse": dipole perpendicular to the axis, orders m = +-1; "axial": along the axis, m = 0),
               sigma_nm (Gaussian smearing), sweep = {mode "fixed" | "resonance", start, stop, n, linewidths}, modal (Riesz expansion)}
    scattering = {theta_deg (angle of incidence to the +z axis, the wave runs along (sin theta, 0, cos theta)), pol ("S": E along y,
                  "P": E in the x-z plane), sweep = {start, stop, n}, max_order (largest |m|), tol (stop when the pair +-m carries less)}

Mesh (MSH 4.1): surfaces tag 1 + index of the material; the cells of the small box around the emitter 101 + index (same material, the closed
surface around them measures the emitted power); curves: the axis r = 0 tag 90, the outer wall behind the PML tag 91 (PEC). Mesh lines at the
measurement planes z_plane_top / z_plane_bottom (r < r_in) and at r = r_plane between them: the closed box of the scattered power and the far
field.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

import fem_geometry as fg
import fem_materials as fm

TAG_AXIS, TAG_WALL, TAG_SOURCE = 90, 91, 100

PART_TYPES = {
    "cylinder": ("Zylinder / Scheibe / Ring", [("r_inner", "Innenradius (nm)"), ("radius", "Außenradius (nm)"), ("z_bottom", "Unterkante z (nm)"),
                                                ("height", "Höhe (nm)")]),
    "cone": ("Kegelstumpf", [("r_bottom", "Radius unten (nm)"), ("r_top", "Radius oben (nm)"), ("z_bottom", "Unterkante z (nm)"), ("height", "Höhe (nm)")]),
    "sphere": ("Kugel", [("z_center", "Mitte z (nm)"), ("radius", "Radius (nm)")]),
    "ellipsoid": ("Rotationsellipsoid", [("z_center", "Mitte z (nm)"), ("r_semi", "Halbachse r (nm)"), ("z_semi", "Halbachse z (nm)")]),
    "torus": ("Torus (Ring, runder Querschnitt)", [("r_center", "Ringradius (nm)"), ("z_center", "Mitte z (nm)"), ("radius", "Querschnittsradius (nm)")]),
    "polygon": ("Polygon (Querschnitt r, z)", []),
}
DEFAULT_PART = {
    "cylinder": dict(r_inner=0.0, radius=500.0, z_bottom=0.0, height=200.0),
    "cone": dict(r_bottom=600.0, r_top=400.0, z_bottom=0.0, height=500.0),
    "sphere": dict(z_center=300.0, radius=300.0),
    "ellipsoid": dict(z_center=300.0, r_semi=400.0, z_semi=250.0),
    "torus": dict(r_center=1500.0, z_center=300.0, radius=250.0),
    "polygon": dict(points=[[0.0, 0.0], [500.0, 0.0], [300.0, 400.0], [0.0, 400.0]]),
}
ORIENTATIONS = {"transverse": "senkrecht zur Achse (m = ±1)", "axial": "entlang der Achse (m = 0)"}


# ------------------------------------------------------------------------------------------------------------------ model
def new_part(kind, material):
    p = {"type": kind, "material": material}
    p.update(copy.deepcopy(DEFAULT_PART[kind]))
    return p


def _idx(n=0.0):
    return {"type": "index", "n": float(n), "k": 0.0}


def default_model():
    return {
        "kind": "axi", "name": "Neuer Rotationskörper",
        "materials": {"Luft": {"type": "library", "name": "air"}, "Glas": _idx(1.5)},
        "background": "Luft", "substrate": None, "layers": [],
        "parts": [dict(type="sphere", material="Glas", z_center=0.0, radius=500.0)],
        "domain": {"margin_nm": 600.0, "pml_nm": 1000.0},
        "resonance": {"wavelength_nm": 900.0, "m": 1, "num_modes": 6},
        "emitter": {"z_nm": 520.0, "orientation": "axial", "sigma_nm": 10.0,
                    "sweep": {"mode": "fixed", "start": 800.0, "stop": 1000.0, "n": 11, "linewidths": 1.5}, "modal": False},
        "scattering": default_scattering(800.0, 1200.0),
    }


def default_scattering(start, stop, n=21):
    return {"theta_deg": 0.0, "pol": "S", "sweep": {"start": float(start), "stop": float(stop), "n": int(n)}, "max_order": 8, "tol": 1e-5}


def dbr_pillar(top_pairs, bottom_pairs, radius, lam, hi="GaAs", lo="AlAs", n_hi=3.53, n_lo=2.95, cavity_lambdas=1.0):
    """Parts of a micropillar: bottom DBR, a cavity of cavity_lambdas wavelengths (in the high-index material), top DBR; quarter-wave layers.
    Returns (parts, z of the cavity centre, height)."""
    q_hi, q_lo = lam / (4 * n_hi), lam / (4 * n_lo)
    cav = cavity_lambdas * lam / n_hi
    stack = [(hi, q_hi), (lo, q_lo)] * int(bottom_pairs) + [(hi, cav)] + [(lo, q_lo), (hi, q_hi)] * int(top_pairs)
    bounds = np.round(np.concatenate([[0.0], np.cumsum([t for _, t in stack])]), 4)   # rounded interfaces: adjacent layers share them exactly
    parts = [dict(type="cylinder", material=mat, r_inner=0.0, radius=float(radius), z_bottom=float(bounds[i]), height=float(bounds[i + 1] - bounds[i]))
             for i, (mat, _) in enumerate(stack)]
    k = 2 * int(bottom_pairs)
    centre = round(0.5 * (bounds[k] + bounds[k + 1]), 4)
    return parts, centre, float(bounds[-1])


def presets():
    P = {}
    lam = 940.0
    parts, centre, height = dbr_pillar(6, 10, 750.0, lam)
    P["Mikrosäule GaAs/AlAs mit Quantenpunkt (schnell: r = 0,75 µm, 6/10 Paare)"] = {
        "kind": "axi", "name": "Mikrosäule 6/10, r = 0,75 µm",
        "materials": {"Luft": {"type": "library", "name": "air"}, "GaAs": _idx(3.53), "AlAs": _idx(2.95)},
        "background": "Luft", "substrate": "GaAs", "parts": parts,
        "domain": {"margin_nm": 0.75 * lam, "pml_nm": 1.5 * lam},
        "resonance": {"wavelength_nm": lam, "m": 1, "num_modes": 4},
        "emitter": {"z_nm": round(centre, 4), "orientation": "transverse", "sigma_nm": 20.0,
                    "sweep": {"mode": "resonance", "start": 920.0, "stop": 940.0, "n": 9, "linewidths": 1.5}},
    }
    parts, centre, height = dbr_pillar(10, 16, 1000.0, lam)
    P["Mikrosäule GaAs/AlAs mit Quantenpunkt (voll: r = 1 µm, 10/16 Paare)"] = {
        "kind": "axi", "name": "Mikrosäule 10/16, r = 1 µm",
        "materials": {"Luft": {"type": "library", "name": "air"}, "GaAs": _idx(3.53), "AlAs": _idx(2.95)},
        "background": "Luft", "substrate": "GaAs", "parts": parts,
        "domain": {"margin_nm": 0.75 * lam, "pml_nm": 1.5 * lam},
        "resonance": {"wavelength_nm": lam, "m": 1, "num_modes": 4},
        "emitter": {"z_nm": round(centre, 4), "orientation": "transverse", "sigma_nm": 20.0,
                    "sweep": {"mode": "resonance", "start": 920.0, "stop": 940.0, "n": 25, "linewidths": 1.5}},
    }
    P["Dielektrische Kugel n = 2,5 (Mie-Resonanzen, Emitter außen)"] = {
        "kind": "axi", "name": "Kugel n = 2,5, r = 300 nm",
        "materials": {"Luft": {"type": "library", "name": "air"}, "Kugel": _idx(2.5)},
        "background": "Luft", "substrate": None,
        "parts": [dict(type="sphere", material="Kugel", z_center=0.0, radius=300.0)],
        "domain": {"margin_nm": 500.0, "pml_nm": 900.0},
        "resonance": {"wavelength_nm": 1000.0, "m": 1, "num_modes": 6},
        "emitter": {"z_nm": 340.0, "orientation": "axial", "sigma_nm": 10.0,
                    "sweep": {"mode": "fixed", "start": 700.0, "stop": 1300.0, "n": 25, "linewidths": 1.5}},
    }
    P["Gold-Nanokugel mit Emitter (Plasmon, Purcell und Quenching)"] = {
        "kind": "axi", "name": "Au-Kugel r = 40 nm, Emitter 10 nm über der Oberfläche",
        "materials": {"Luft": {"type": "library", "name": "air"}, "Au": {"type": "library", "name": "Au"}},
        "background": "Luft", "substrate": None,
        "parts": [dict(type="sphere", material="Au", z_center=0.0, radius=40.0)],
        "domain": {"margin_nm": 150.0, "pml_nm": 300.0},
        "resonance": {"wavelength_nm": 520.0, "m": 0, "num_modes": 4},
        "emitter": {"z_nm": 50.0, "orientation": "axial", "sigma_nm": 2.0,
                    "sweep": {"mode": "fixed", "start": 450.0, "stop": 700.0, "n": 26, "linewidths": 1.5}},
    }
    P["Gold-Nanokugel in Wasser: Streuung und Absorption (Mie-Vergleich)"] = {
        "kind": "axi", "name": "Au-Kugel r = 40 nm in Wasser",
        "materials": {"Wasser": {"type": "library", "name": "water"}, "Au": {"type": "library", "name": "Au"}},
        "background": "Wasser", "substrate": None,
        "parts": [dict(type="sphere", material="Au", z_center=0.0, radius=40.0)],
        "domain": {"margin_nm": 200.0, "pml_nm": 400.0},
        "resonance": {"wavelength_nm": 530.0, "m": 1, "num_modes": 4},
        "emitter": {"z_nm": 50.0, "orientation": "axial", "sigma_nm": 2.0,
                    "sweep": {"mode": "fixed", "start": 450.0, "stop": 700.0, "n": 26, "linewidths": 1.5}, "modal": False},
        "scattering": dict(default_scattering(450.0, 700.0, 26), max_order=4),
    }
    P["Silizium-Nanokugel: Mie-Resonanzen (magnetischer Dipol, schräger Einfall)"] = {
        "kind": "axi", "name": "Si-Kugel r = 75 nm, Einfall 45°",
        "materials": {"Luft": {"type": "library", "name": "air"}, "Si": {"type": "library", "name": "Si"}},
        "background": "Luft", "substrate": None,
        "parts": [dict(type="sphere", material="Si", z_center=0.0, radius=75.0)],
        "domain": {"margin_nm": 300.0, "pml_nm": 500.0},
        "resonance": {"wavelength_nm": 600.0, "m": 1, "num_modes": 4},
        "emitter": {"z_nm": 100.0, "orientation": "axial", "sigma_nm": 5.0,
                    "sweep": {"mode": "fixed", "start": 450.0, "stop": 800.0, "n": 15, "linewidths": 1.5}, "modal": False},
        "scattering": dict(default_scattering(450.0, 800.0, 36), theta_deg=45.0, pol="P", max_order=8),
    }
    P["GaAs-Membran 200 nm mit Quantenpunkt (Schicht bis in die PML, Sommerfeld-Test)"] = {
        "kind": "axi", "name": "GaAs-Membran 200 nm in Luft, Emitter in der Mitte",
        "materials": {"Luft": {"type": "library", "name": "air"}, "GaAs": _idx(3.53)},
        "background": "Luft", "substrate": None, "layers": [dict(material="GaAs", z_bottom=0.0, height=200.0)],
        "parts": [],
        "domain": {"margin_nm": 0.75 * lam, "pml_nm": 1.5 * lam},
        "resonance": {"wavelength_nm": lam, "m": 1, "num_modes": 4},
        "emitter": {"z_nm": 100.0, "orientation": "transverse", "sigma_nm": 10.0,
                    "sweep": {"mode": "fixed", "start": 900.0, "stop": 1000.0, "n": 5, "linewidths": 1.5}, "modal": False},
    }
    parts, centre, height = dbr_pillar(6, 0, 750.0, lam)
    mirror, _, _ = dbr_pillar(0, 10, 1.0, lam)                            # 10 pairs and a cavity layer; the pairs become planar layers
    lay_mirror = [dict(material=q["material"], z_bottom=q["z_bottom"], height=q["height"]) for q in mirror[:-1]]
    pillar = [dict(q, z_bottom=round(q["z_bottom"] + mirror[-1]["z_bottom"], 4)) for q in parts]
    P["Mikrosäule auf planarem unterem Spiegel (Schichten bis in die PML)"] = {
        "kind": "axi", "name": "Säule (Kavität + 6 Paare) auf planarem DBR (10 Paare)",
        "materials": {"Luft": {"type": "library", "name": "air"}, "GaAs": _idx(3.53), "AlAs": _idx(2.95)},
        "background": "Luft", "substrate": "GaAs", "layers": lay_mirror, "parts": pillar,
        "domain": {"margin_nm": 0.75 * lam, "pml_nm": 1.5 * lam},
        "resonance": {"wavelength_nm": lam, "m": 1, "num_modes": 4},
        "emitter": {"z_nm": round(centre + mirror[-1]["z_bottom"], 4), "orientation": "transverse", "sigma_nm": 20.0,
                    "sweep": {"mode": "resonance", "start": 920.0, "stop": 940.0, "n": 9, "linewidths": 1.5}, "modal": False},
    }
    P["Mikroscheibe n = 2 auf Glas (Flüstergalerie-Moden, m = 12)"] = {
        "kind": "axi", "name": "Mikroscheibe r = 1,5 µm, 250 nm dick",
        "materials": {"Luft": {"type": "library", "name": "air"}, "Scheibe": _idx(2.0), "Glas": _idx(1.45)},
        "background": "Luft", "substrate": None,
        "parts": [dict(type="cylinder", material="Glas", r_inner=0.0, radius=900.0, z_bottom=-600.0, height=600.0),
                  dict(type="cylinder", material="Scheibe", r_inner=0.0, radius=1500.0, z_bottom=0.0, height=250.0)],
        "domain": {"margin_nm": 600.0, "pml_nm": 1000.0},
        "resonance": {"wavelength_nm": 975.0, "m": 12, "num_modes": 6},
        "emitter": {"z_nm": 125.0, "orientation": "axial", "sigma_nm": 20.0,
                    "sweep": {"mode": "fixed", "start": 900.0, "stop": 1050.0, "n": 11, "linewidths": 1.5}},
    }
    return {k: complete(v) for k, v in P.items()}


def complete(m):
    """Adds the keys of newer versions (scattering, modal) to a model."""
    if not m.get("scattering"):
        lam = float(m.get("resonance", {}).get("wavelength_nm", 900.0))
        m["scattering"] = default_scattering(round(0.7 * lam), round(1.4 * lam))
    base = default_model()
    for k, v in base.items():
        m.setdefault(k, copy.deepcopy(v))
    m["emitter"].setdefault("modal", False)
    m.setdefault("layers", [])
    return m


def load_model(text):
    m = json.loads(text)
    if m.get("kind") != "axi":
        raise ValueError("Kein Modell eines Rotationskörpers (kind = axi)")
    base = default_model()
    complete(m)
    for k in ("domain", "resonance", "emitter"):
        for kk, vv in base[k].items():
            m[k].setdefault(kk, copy.deepcopy(vv))
    return m


# ---------------------------------------------------------------------------------------------------------------- geometry
def part_polygon(p, n=96):
    """Cross-section of a part in the meridian half-plane as a closed polygon (r, z) in nm."""
    t = p["type"]
    if t == "cylinder":
        r0, r1, z0, h = p["r_inner"], p["radius"], p["z_bottom"], p["height"]
        return np.array([[r0, z0], [r1, z0], [r1, z0 + h], [r0, z0 + h]], float)
    if t == "cone":
        z0, h = p["z_bottom"], p["height"]
        return np.array([[0.0, z0], [p["r_bottom"], z0], [p["r_top"], z0 + h], [0.0, z0 + h]], float)
    if t in ("sphere", "ellipsoid"):
        a = p["radius"] if t == "sphere" else p["r_semi"]
        b = p["radius"] if t == "sphere" else p["z_semi"]
        s = np.linspace(-np.pi / 2, np.pi / 2, n)
        return np.column_stack([a * np.cos(s), p["z_center"] + b * np.sin(s)])
    if t == "torus":
        s = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([p["r_center"] + p["radius"] * np.cos(s), p["z_center"] + p["radius"] * np.sin(s)])
    return np.asarray(p["points"], float)


def part_bbox(p):
    q = part_polygon(p)
    return float(q[:, 0].min()), float(q[:, 0].max()), float(q[:, 1].min()), float(q[:, 1].max())


def _inside(poly, r, z):
    """Point in polygon (even-odd) for arrays r, z."""
    r, z = np.asarray(r, float), np.asarray(z, float)
    inside = np.zeros(r.shape, bool)
    x, y = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((y[i] > z) != (y[j] > z)) & (r < (x[j] - x[i]) * (z - y[i]) / (y[j] - y[i] + 1e-300) + x[i])
        inside ^= cond
        j = i
    return inside


def materials_at(model, r, z):
    """Material names at the points (r, z) [nm] (arrays): the last part containing a point, else the substrate (z < 0), else the background."""
    r, z = np.atleast_1d(np.asarray(r, float)), np.atleast_1d(np.asarray(z, float))
    out = np.where((z < 0) & bool(model.get("substrate")), model.get("substrate") or "", model["background"]).astype(object)
    for ly in model.get("layers", []):
        out[(z >= ly["z_bottom"]) & (z < ly["z_bottom"] + ly["height"])] = ly["material"]
    for p in model["parts"]:
        out[_inside(part_polygon(p, 192), r, z)] = p["material"]
    return out


def base_material(model, z):
    """Material of the planar background at height z (substrate, layers, surrounding medium) without the parts."""
    m = model["substrate"] if (model.get("substrate") and z < 0) else model["background"]
    for ly in model.get("layers", []):
        if ly["z_bottom"] <= z < ly["z_bottom"] + ly["height"]:
            m = ly["material"]
    return m


def material_at(model, r, z):
    """Name of the material at the point (r, z) [nm]."""
    return str(materials_at(model, [r], [z])[0])


def layout(model):
    """Vertical and radial layout in nm: structure extent, the inner box (PML starts outside it), the outer domain, the collection planes."""
    d = model["domain"]
    margin, pml = float(d["margin_nm"]), float(d["pml_nm"])
    zs_ = [(b[2], b[3]) for b in (part_bbox(p) for p in model["parts"])]
    zs_ += [(float(ly["z_bottom"]), float(ly["z_bottom"]) + float(ly["height"])) for ly in model.get("layers", [])]
    if model.get("substrate"):
        zs_.append((0.0, 0.0))
    r_max = max([part_bbox(p)[1] for p in model["parts"]], default=0.0)
    z_lo_s = min([a for a, _ in zs_], default=0.0)
    z_hi_s = max([b for _, b in zs_], default=0.0)
    em = model.get("emitter", {})
    if em.get("z_nm") is not None:
        z_lo_s, z_hi_s = min(z_lo_s, float(em["z_nm"])), max(z_hi_s, float(em["z_nm"]))
    r_in = r_max + margin
    z_lo, z_hi = z_lo_s - margin, z_hi_s + margin
    return dict(r_struct=r_max, z_struct_bottom=z_lo_s, z_struct_top=z_hi_s, r_in=r_in, r_out=r_in + pml, z_lo=z_lo, z_hi=z_hi,
                z_bot=z_lo - pml, z_top=z_hi + pml, pml=pml, margin=margin,
                z_plane_top=z_hi_s + 0.5 * margin, z_plane_bottom=z_lo_s - 0.5 * margin, r_plane=r_max + 0.5 * margin)


def source_box(model):
    """Half-size a (nm) of the box [0, a] x [z - a, z + a] around the emitter, from the smearing width."""
    em = model["emitter"]
    return max(4.0 * float(em["sigma_nm"]), 5.0)


def wavelength_range(model, task="resonance"):
    """(lam_min, lam_max) in nm over which the model is meshed and checked."""
    lam_r = float(model["resonance"]["wavelength_nm"])
    if task == "resonance":
        return lam_r * 0.95, lam_r * 1.05
    if task == "scattering":
        sw = model.get("scattering", {}).get("sweep") or {"start": lam_r, "stop": lam_r}
        return float(min(sw["start"], sw["stop"])), float(max(sw["start"], sw["stop"]))
    sw = model["emitter"]["sweep"]
    if sw["mode"] == "resonance":
        return lam_r * 0.95, lam_r * 1.05
    return float(min(sw["start"], sw["stop"])), float(max(sw["start"], sw["stop"]))


def sweep_values(sw):
    n = max(int(sw["n"]), 1)
    return [float(round(v, 6)) for v in np.linspace(sw["start"], sw["stop"], n)]


def validate(model, task="resonance"):
    """[(level, text)] of the model."""
    out = []
    names = set(model["materials"])
    for key in ("background",):
        if model[key] not in names:
            out.append(("error", f"Umgebungsmaterial „{model[key]}“ fehlt in der Materialliste."))
    if model.get("substrate") and model["substrate"] not in names:
        out.append(("error", f"Substrat „{model['substrate']}“ fehlt in der Materialliste."))
    lam_lo, lam_hi = wavelength_range(model, task)
    try:
        e = fm.eps_at(model["materials"][model["background"]], lam_lo)
        if abs(e.imag) > 1e-9:
            out.append(("error", "Das Umgebungsmaterial muss verlustfrei sein (PML und Abstrahlung)."))
    except Exception as exc:
        out.append(("error", f"Umgebungsmaterial: {exc}"))
    if not model["parts"] and not model.get("layers"):
        out.append(("warning", "Keine Struktur: nur Umgebung (und Substrat)."))
    for i, ly in enumerate(model.get("layers", [])):
        if ly["material"] not in names:
            out.append(("error", f"Schicht {i + 1}: Material „{ly['material']}“ fehlt."))
        if ly["height"] <= 0:
            out.append(("error", f"Schicht {i + 1}: Dicke > 0 nötig."))
        if model.get("substrate") and ly["z_bottom"] < -1e-9:
            out.append(("warning", f"Schicht {i + 1} reicht unter z = 0 ins Substrat und ersetzt es dort."))
    for i, p in enumerate(model["parts"]):
        if p["material"] not in names:
            out.append(("error", f"Teil {i + 1}: Material „{p['material']}“ fehlt."))
        q = part_polygon(p)
        if q[:, 0].min() < -1e-9:
            out.append(("error", f"Teil {i + 1}: negative Radien (r muss ≥ 0 sein)."))
        t = p["type"]
        if t == "cylinder" and (p["radius"] <= p["r_inner"] or p["height"] <= 0):
            out.append(("error", f"Teil {i + 1}: Außenradius > Innenradius und Höhe > 0 nötig."))
        if t == "cone" and (p["height"] <= 0 or p["r_bottom"] < 0 or p["r_top"] < 0 or max(p["r_bottom"], p["r_top"]) <= 0):
            out.append(("error", f"Teil {i + 1}: Höhe > 0 und Radien ≥ 0 nötig."))
        if t == "torus" and p["r_center"] <= p["radius"]:
            out.append(("error", f"Teil {i + 1}: der Ringradius muss größer als der Querschnittsradius sein (sonst „Kugel“ oder „Polygon“)."))
        if t in ("sphere", "ellipsoid") and min(p.get("radius", 1), p.get("r_semi", 1), p.get("z_semi", 1)) <= 0:
            out.append(("error", f"Teil {i + 1}: Radien > 0 nötig."))
    for lam in (lam_lo, lam_hi):
        for n_, spec in model["materials"].items():
            try:
                fm.eps_at(spec, lam)
            except Exception as exc:
                out.append(("error", f"{n_} bei {lam:g} nm: {exc}"))
    d = model["domain"]
    if d["pml_nm"] < 0.5 * lam_hi:
        out.append(("warning", f"PML dünner als eine halbe Wellenlänge ({d['pml_nm']:g} nm < {0.5 * lam_hi:g} nm): Reflexionen möglich."))
    if d["margin_nm"] < 0.25 * lam_hi:
        out.append(("warning", "Abstand Struktur–PML unter λ/4: Nahfelder reichen in die PML."))
    r = model["resonance"]
    if int(r["m"]) < 0:
        out.append(("error", "Die Azimutordnung m muss ≥ 0 sein (−m hat dieselben Moden)."))
    if task == "scattering":
        sc = model.get("scattering") or {}
        if model.get("substrate") or model.get("layers"):
            out.append(("error", "Streuung ebener Wellen geht nur in homogener Umgebung: Substrat und radial unendliche Schichten entfernen (die "
                                 "einfallende Welle muss eine Lösung im Umgebungsmedium sein; geschichtete Hintergründe bietet der zylindersymmetrische "
                                 "Löser nicht)."))
        if not 0.0 <= float(sc.get("theta_deg", 0.0)) <= 180.0:
            out.append(("error", "Einfallswinkel zwischen 0° und 180° (gegen die +z-Achse)."))
        if int(sc.get("max_order", 8)) < 1:
            out.append(("error", "Höchste Azimutordnung mindestens 1."))
        for p in model["parts"]:
            if part_bbox(p)[1] > 0 and material_at(model, part_bbox(p)[1] + 0.25 * float(model["domain"]["margin_nm"]), 0.5 * (part_bbox(p)[2] + part_bbox(p)[3])) != model["background"]:
                out.append(("warning", "Die Messfläche der Streuleistung liegt nicht vollständig in der Umgebung."))
                break
    if task == "emitter":
        em = model["emitter"]
        a = source_box(model)
        z = float(em["z_nm"])
        R, Z = np.meshgrid(np.linspace(0.0, a, 5), np.linspace(z - a, z + a, 41))
        mats = set(materials_at(model, R.ravel(), Z.ravel()))
        if len(mats) > 1:
            out.append(("error", f"Der Kasten um den Emitter (r ≤ {a:g} nm, z = {z:g} ± {a:g} nm, 4σ) liegt in mehreren Materialien "
                                 f"({', '.join(sorted(mats))}): Emitter weiter von der Grenzfläche weg oder σ kleiner."))
        else:
            try:
                ed = fm.eps_at(model["materials"][mats.pop()], lam_lo)
                if abs(ed.imag) > 1e-6:
                    out.append(("error", "Der Emitter sitzt in einem verlustbehafteten Material: die Purcell-Referenz (Larmor-Leistung) ist dort nicht definiert."))
            except Exception:
                pass
        if em["sweep"]["mode"] == "fixed" and int(em["sweep"]["n"]) < 1:
            out.append(("error", "Mindestens ein Punkt im Spektrum."))
    return out


# ----------------------------------------------------------------------------------------------------------------- figures
def preview_figure(model, figsize=(7.2, 6.0), show_domain=True):
    """Cross-section of the body of revolution, mirrored to -r (dashed axis), with domain, PML, emitter and collection planes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Poly
    from matplotlib.patches import Rectangle

    lay = layout(model)
    fig, ax = plt.subplots(figsize=figsize)
    R, z0, z1 = lay["r_out"], lay["z_bot"], lay["z_top"]
    ax.add_patch(Rectangle((-R, z0), 2 * R, z1 - z0, color=fg.material_color(model, model["background"]), alpha=0.35, lw=0))
    if model.get("substrate"):
        ax.add_patch(Rectangle((-R, z0), 2 * R, -z0, color=fg.material_color(model, model["substrate"]), alpha=0.9, lw=0))
    for ly in model.get("layers", []):
        ax.add_patch(Rectangle((-R, ly["z_bottom"]), 2 * R, ly["height"], facecolor=fg.material_color(model, ly["material"]), edgecolor="0.3", lw=0.3))
    for p in model["parts"]:
        q = part_polygon(p)
        col = fg.material_color(model, p["material"])
        for s in (1, -1):
            ax.add_patch(Poly(np.column_stack([s * q[:, 0], q[:, 1]]), closed=True, facecolor=col, edgecolor="0.2", lw=0.4))
    if show_domain:
        for s in (1, -1):
            ax.add_patch(Rectangle((s * lay["r_in"] if s > 0 else -lay["r_out"], z0), lay["pml"], z1 - z0, facecolor="none", hatch="///",
                                   edgecolor="0.6", lw=0))
        ax.add_patch(Rectangle((-lay["r_in"], lay["z_hi"]), 2 * lay["r_in"], lay["pml"], facecolor="none", hatch="///", edgecolor="0.6", lw=0))
        ax.add_patch(Rectangle((-lay["r_in"], z0), 2 * lay["r_in"], lay["pml"], facecolor="none", hatch="///", edgecolor="0.6", lw=0))
        ax.plot([-lay["r_in"], lay["r_in"], lay["r_in"], -lay["r_in"], -lay["r_in"]], [lay["z_lo"], lay["z_lo"], lay["z_hi"], lay["z_hi"], lay["z_lo"]],
                color="0.4", lw=0.8, ls=":")
        for zp in (lay["z_plane_top"], lay["z_plane_bottom"]):
            ax.plot([-lay["r_in"], lay["r_in"]], [zp, zp], color="C2", lw=0.8, ls="--")
    em = model.get("emitter")
    if em and em.get("z_nm") is not None:
        mk = "↕" if em["orientation"] == "axial" else "↔"
        ax.plot([0], [em["z_nm"]], marker="*", ms=13, color="C3", zorder=5)
        ax.annotate(f"Emitter {mk}", (0, em["z_nm"]), xytext=(8, 8), textcoords="offset points", color="C3", fontsize=8)
    ax.axvline(0, color="k", lw=0.8, ls="-.")
    handles = [Rectangle((0, 0), 1, 1, color=fg.material_color(model, n)) for n in model["materials"]]
    ax.legend(handles, list(model["materials"]), fontsize=7, loc="upper right")
    ax.set_xlim(-R, R)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal")
    ax.set_xlabel("r (nm)  (gespiegelt: Schnitt durch die Achse)")
    ax.set_ylabel("z (nm)")
    ax.set_title("Querschnitt; schraffiert: PML, grün gestrichelt: Messebenen", fontsize=9)
    fig.tight_layout()
    return fig


def mesh_figure(plot_npz, model, zoom=None, figsize=(7.2, 6.0)):
    fig = fg.mesh_figure(plot_npz, model, figsize=figsize, zoom=zoom)
    ax = fig.axes[0]
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("z (nm)")
    return fig


def fig_modes(modes, target_nm=None):
    """Quality factor over the resonance wavelength of the modes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ok = [m for m in modes if m["Q"] > 0]
    ax.semilogy([m["lam_nm"] for m in ok], [m["Q"] for m in ok], "o", ms=8)
    for m in ok:
        ax.annotate(str(m["k"]), (m["lam_nm"], m["Q"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    if target_nm:
        ax.axvline(target_nm, color="0.5", ls="--", lw=0.8, label="Ziel")
        ax.legend(fontsize=8)
    ax.set_xlabel("Resonanzwellenlänge λ_res (nm)")
    ax.set_ylabel("Güte Q")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_purcell(points, resonance=None):
    """Purcell factor and the fractions of the emitted power through the upper and lower plane over the wavelength."""
    import matplotlib.pyplot as plt

    ok = [p for p in points if "error" not in p]
    fig, axs = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    if not ok:
        axs[0].text(0.5, 0.5, "keine Punkte", ha="center")
        return fig
    lam = [p["lam_nm"] for p in ok]
    axs[0].plot(lam, [p["purcell"] for p in ok], "o-", color="C0", label="Purcell-Faktor F_P = P / P_bulk")
    if any(p.get("purcell_radiative") is not None for p in ok):
        axs[0].plot(lam, [p.get("purcell_radiative") for p in ok], "s--", color="C4", ms=4, label="abgestrahlt (oben + unten) / P_bulk")
    axs[0].set_ylabel("F_P")
    axs[1].plot(lam, [p["beta_top"] for p in ok], "o-", color="C1", label="β oben (durch die obere Ebene)")
    axs[1].plot(lam, [p["beta_bottom"] for p in ok], "s-", color="C2", label="nach unten (durch die untere Ebene)")
    axs[1].plot(lam, [max(1.0 - p["beta_top"] - p["beta_bottom"], 0.0) for p in ok], "^--", color="0.5", label="Rest: seitlich (+ absorbiert)")
    axs[1].set_ylabel("Anteil der emittierten Leistung")
    axs[1].set_xlabel("Wellenlänge (nm)")
    if resonance:
        for a in axs:
            a.axvline(resonance["lam_nm"], color="C3", lw=0.8, ls="--", label=f"Resonanz (Q = {resonance['Q']:.0f})" if a is axs[0] else None)
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_cross_sections(points, geometric_nm2=None, theta=0.0, pol="S"):
    """Scattering, absorption and extinction cross-sections over the wavelength (and the Mie series where the worker gave it)."""
    import matplotlib.pyplot as plt

    ok = [p for p in points if "error" not in p]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    if not ok:
        ax.text(0.5, 0.5, "keine Punkte", ha="center")
        return fig
    lam = [p["lam_nm"] for p in ok]
    scale = 1e12                                                        # m^2 -> um^2
    for key, lab, col, mk in (("sigma_ext", "Extinktion", "C0", "o-"), ("sigma_sca", "Streuung", "C1", "s-"), ("sigma_abs", "Absorption", "C3", "^-")):
        ax.plot(lam, [p[key] * scale for p in ok], mk, ms=4, color=col, label=f"σ_{lab[:3].lower()} {lab}")
        if all(p.get("mie") for p in ok):
            ax.plot(lam, [p["mie"][key] * scale for p in ok], "--", color=col, lw=1.0, alpha=0.8, label=f"{lab} (Mie)")
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("Querschnitt (µm²)")
    if geometric_nm2:
        sec = ax.secondary_yaxis("right", functions=(lambda v: v / (geometric_nm2 * 1e-6), lambda q: q * geometric_nm2 * 1e-6))
        sec.set_ylabel("Effizienz Q = σ / (π R²)")
    ax.set_title(f"Einfall θ = {theta:g}° gegen die Achse, {pol}-polarisiert", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def fig_pattern(pt):
    """Differential scattering cross-section dσ/dΩ in the plane of incidence (x-z) and perpendicular to it (y-z), polar plot."""
    import matplotlib.pyplot as plt

    th = np.asarray(pt["pattern_theta"])
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for key, lab, col in (("xz", "Einfallsebene (x-z)", "C0"), ("yz", "senkrecht dazu (y-z)", "C1")):
        a, b = np.asarray(pt[f"pattern_{key}_0"]), np.asarray(pt[f"pattern_{key}_pi"])
        ang = np.concatenate([th, 2 * np.pi - th[::-1]])
        val = np.concatenate([a, b[::-1]]) * 1e12
        ax.plot(ang, val, color=col, label=lab)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ti = np.radians(pt.get("theta_deg", 0.0))
    ax.annotate("", xy=(ti, ax.get_rmax() * 0.9), xytext=(ti + np.pi, ax.get_rmax() * 0.9),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))
    ax.set_title(f"dσ/dΩ (µm²/sr) bei λ = {pt['lam_nm']:.1f} nm; 0° = +z, Pfeil: Einfallsrichtung", fontsize=9)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.1, -0.12))
    fig.tight_layout()
    return fig


SCATTER_QUANTITIES = {"|E| gesamt": ("tot", "abs"), "|E| Streufeld": ("sca", "abs"), "|E_x| gesamt": ("tot", 0), "|E_y| gesamt": ("tot", 1),
                      "|E_z| gesamt": ("tot", 2), "Re E_x gesamt": ("tot", "r0"), "Re E_y gesamt": ("tot", "r1"), "Re E_z gesamt": ("tot", "r2")}


def scatter_field(d, which):
    """Cartesian field in the plane of incidence: (points (2n, 2) as x = +-r, z; values (2n, 3)) of the scattered or the total field."""
    pts = d["points_nm"]
    E_r, E_l = d["E_right"].astype(complex), d["E_left"].astype(complex)
    x = np.concatenate([pts[:, 0], -pts[:, 0]])
    z = np.concatenate([pts[:, 1], pts[:, 1]])
    E = np.concatenate([E_r, E_l])
    if which == "tot":
        th = np.radians(float(d["theta_deg"]))
        e = np.array([0.0, 1.0, 0.0]) if str(d["pol"]) == "S" else np.array([np.cos(th), 0.0, -np.sin(th)])
        k = float(d["k_bg"]) * 1e-9                                      # per nm
        E = E + e[None, :] * np.exp(1j * k * (x * np.sin(th) + z * np.cos(th)))[:, None]
    return np.column_stack([x, z]), E


def fig_scatter_field(d, model, key, log=False, zoom=None, figsize=(7.4, 6.4)):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.tri import Triangulation

    which, comp = SCATTER_QUANTITIES[key]
    xz, E = scatter_field(d, which)
    if comp == "abs":
        f, signed = np.sqrt((np.abs(E) ** 2).sum(axis=1)), False
    elif isinstance(comp, int):
        f, signed = np.abs(E[:, comp]), False
    else:
        f, signed = E[:, int(comp[1])].real, True
    n = len(d["points_nm"])
    tri = np.concatenate([d["simplices"], d["simplices"] + n])
    top = float(np.nanmax(np.abs(f))) or 1.0
    if signed:
        norm, cm = Normalize(-top, top), "RdBu_r"
    elif log:
        f = np.maximum(f, top * 1e-3)
        norm, cm = LogNorm(top * 1e-3, top), "inferno"
    else:
        norm, cm = Normalize(0, top), "inferno"
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.tripcolor(Triangulation(xz[:, 0], xz[:, 1], tri), f, shading="gouraud", cmap=cm, norm=norm)
    fig.colorbar(im, ax=ax, fraction=0.046, label=key + " (|E₀| = 1)")
    for p in model["parts"]:
        q = part_polygon(p)
        for s_ in (1, -1):
            ax.plot(np.append(s_ * q[:, 0], s_ * q[0, 0]), np.append(q[:, 1], q[0, 1]), color="w", lw=0.6, alpha=0.8)
    th = np.radians(float(d["theta_deg"]))
    R = 0.8 * (zoom[0] if zoom else xz[:, 0].max())
    ax.annotate("", xy=(0.25 * R * np.sin(th) - 0.6 * R * np.sin(th), 0.25 * R * np.cos(th) - 0.6 * R * np.cos(th)),
                xytext=(-0.6 * R * np.sin(th) - 0.35 * R * np.sin(th), -0.6 * R * np.cos(th) - 0.35 * R * np.cos(th)),
                arrowprops=dict(arrowstyle="->", color="w", lw=1.2))
    ax.set_aspect("equal")
    if zoom:
        ax.set_xlim(-zoom[0], zoom[0])
        ax.set_ylim(zoom[1], zoom[2])
    ax.set_xlabel("x (nm)  (Einfallsebene y = 0)")
    ax.set_ylabel("z (nm)")
    ax.set_title(f"{key}, λ = {float(d['lam_nm']):.1f} nm, θ = {float(d['theta_deg']):g}°, {str(d['pol'])}", fontsize=9)
    fig.tight_layout()
    return fig


def fig_modal(points, chosen=None):
    """Purcell spectrum of the direct solves against the modal expansion (Riesz projection): total, the share of the chosen mode, background."""
    import matplotlib.pyplot as plt

    ok = [p for p in points if "error" not in p and p.get("modal")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if not ok:
        ax.text(0.5, 0.5, "keine Modenzerlegung", ha="center")
        return fig
    lam = [p["lam_nm"] for p in ok]
    ax.plot(lam, [p["purcell"] for p in ok], "o", color="k", ms=5, label="direkt (je Wellenlänge gelöst)")
    ax.plot(lam, [p["modal"]["total"] for p in ok], "-", color="C0", label="Summe der Moden + Hintergrund (Riesz)")
    ax.plot(lam, [p["modal"]["mode"] for p in ok], "--", color="C3", label="Anteil der Resonanz" + (f" (Q = {chosen['Q']:.0f})" if chosen else ""))
    ax.plot(lam, [p["modal"]["background"] for p in ok], ":", color="0.4", label="Hintergrund (übrige Pole, Kontinuum)")
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("Purcell-Faktor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


AXI_QUANTITIES = {
    "|E| (Betrag)": "Eabs", "|E|²": "E2", "|E_r|": "c0", "|E_φ|": "c1", "|E_z|": "c2", "Re E_r": "r0", "Re E_φ": "r1", "Re E_z": "r2",
}


def fig_axi_field(d, model, key, log=False, cmap=None, mirror=True, zoom=None, figsize=(7.4, 6.4), title=""):
    """Field on the subdivided meridian mesh (from fem_axi_worker), mirrored to -r: the cut through the axis at azimuth 0 / pi. For order m the
    mirrored half carries the factor e^{i m pi} = (-1)^m (signed components only; magnitudes are mirror-symmetric)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.tri import Triangulation

    pts, simp, E = d["points_nm"], d["simplices"].astype(int), d["E"].astype(complex)
    code = AXI_QUANTITIES[key]
    if code == "Eabs":
        f = np.sqrt((np.abs(E) ** 2).sum(axis=1))
    elif code == "E2":
        f = (np.abs(E) ** 2).sum(axis=1)
    else:
        c = E[:, int(code[1])]
        f = np.abs(c) if code[0] == "c" else c.real
    signed = code[0] == "r"
    m = int(d.get("m", 0))
    x, y, tri, vals = pts[:, 0], pts[:, 1], simp, f
    if mirror:
        sign = (-1.0) ** m if signed else 1.0
        if signed and code == "r0":
            sign = -sign                                   # E_r points away from the axis on both sides
        x = np.concatenate([x, -pts[:, 0]])
        y = np.concatenate([y, pts[:, 1]])
        tri = np.concatenate([simp, simp + len(pts)])
        vals = np.concatenate([f, sign * f])
    top = float(np.nanmax(np.abs(vals))) or 1.0
    if signed:
        norm, cm = Normalize(-top, top), cmap or "RdBu_r"
    elif log:
        lo = top * 1e-4
        vals = np.maximum(vals, lo)
        norm, cm = LogNorm(lo, top), cmap or "inferno"
    else:
        norm, cm = Normalize(0, top), cmap or "inferno"
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.tripcolor(Triangulation(x, y, tri), vals, shading="gouraud", cmap=cm, norm=norm)
    fig.colorbar(im, ax=ax, fraction=0.046, label=key + " (normiert)")
    for p in model["parts"]:
        q = part_polygon(p)
        for s in ((1, -1) if mirror else (1,)):
            ax.plot(np.append(s * q[:, 0], s * q[0, 0]), np.append(q[:, 1], q[0, 1]), color="w", lw=0.6, alpha=0.8)
    if model.get("substrate"):
        ax.axhline(0.0, color="w", lw=0.6, ls="--", alpha=0.8)
    for ly in model.get("layers", []):
        for zz in (ly["z_bottom"], ly["z_bottom"] + ly["height"]):
            ax.axhline(zz, color="w", lw=0.4, ls=":", alpha=0.7)
    ax.axvline(0, color="w", lw=0.5, ls="-.", alpha=0.6)
    ax.set_aspect("equal")
    if zoom:
        ax.set_xlim(-zoom[0] if mirror else 0, zoom[0])
        ax.set_ylim(zoom[1], zoom[2])
    else:
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def load_field(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------------------------- mesh
def _make_part(occ, p):
    t = p["type"]
    if t == "cylinder":
        return occ.addRectangle(p["r_inner"], p["z_bottom"], 0, p["radius"] - p["r_inner"], p["height"])
    if t == "sphere":
        return occ.addDisk(0, p["z_center"], 0, p["radius"], p["radius"])
    if t == "ellipsoid":
        if p["r_semi"] >= p["z_semi"]:
            return occ.addDisk(0, p["z_center"], 0, p["r_semi"], p["z_semi"])
        return occ.addDisk(0, p["z_center"], 0, p["z_semi"], p["r_semi"], -1, [], [0, 1, 0])
    if t == "torus":
        return occ.addDisk(p["r_center"], p["z_center"], 0, p["radius"], p["radius"])
    q = part_polygon(p)
    ids = [occ.addPoint(float(r), float(z), 0) for r, z in q]
    lines = [occ.addLine(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def build_mesh(model, ms, out_dir, task="resonance"):
    """Meridian mesh with Gmsh (MSH 4.1 ASCII): out_dir/mesh.msh and mesh_plot.npz. ms: cells_per_wavelength, skin_cells, interface_factor,
    curved, pml_cells_per_wavelength, lam_min_nm, lam_max_nm. Returns statistics."""
    import gmsh

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errs = [t for lvl, t in validate(model, task) if lvl == "error"]
    if errs:
        raise ValueError("Modell fehlerhaft: " + " | ".join(errs))
    names = list(model["materials"])
    lay = layout(model)
    curved = bool(ms.get("curved", True)) and any(p["type"] in ("sphere", "ellipsoid", "torus") for p in model["parts"])
    em = model["emitter"]
    with_source = task == "emitter"
    a = source_box(model)
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-4)          # nm: coordinates closer than this are the same point
        gmsh.model.add("meridian")
        occ = gmsh.model.occ
        rs = [0.0, lay["r_in"], lay["r_out"]]
        zset = {lay["z_bot"], lay["z_lo"], lay["z_hi"], lay["z_top"]}
        if model.get("substrate"):
            zset |= {0.0}
        for ly in model.get("layers", []):
            zset |= {float(ly["z_bottom"]), float(ly["z_bottom"]) + float(ly["height"])}
        zs = sorted(z_ for z_ in zset if lay["z_bot"] <= z_ <= lay["z_top"])
        base = []                                                    # (tag, material, kind): planar slabs, radially through the PML
        for i in range(len(rs) - 1):
            for j in range(len(zs) - 1):
                tag = occ.addRectangle(rs[i], zs[j], 0, rs[i + 1] - rs[i], zs[j + 1] - zs[j])
                zc = 0.5 * (zs[j] + zs[j + 1])
                mat = base_material(model, zc)
                kind = "pml" if (i == 1 or zs[j + 1] <= lay["z_lo"] + 1e-9 or zs[j] >= lay["z_hi"] - 1e-9) else "inner"
                base.append((tag, mat, kind))
        domain_box = occ.addRectangle(0, lay["z_bot"], 0, lay["r_out"], lay["z_top"] - lay["z_bot"])
        tools = []                                                   # (dimtag, role, part index)
        for idx, p in enumerate(model["parts"]):
            tag = _make_part(occ, p)
            out, _ = occ.intersect([(2, tag)], [(2, domain_box)], removeObject=True, removeTool=False)
            tools += [(dt, "part", idx) for dt in out if dt[0] == 2]
        if with_source:
            sb = occ.addRectangle(0, float(em["z_nm"]) - a, 0, a, 2 * a)
            tools.append(((2, sb), "source", -1))
        lines = []
        for zp in (lay["z_plane_top"], lay["z_plane_bottom"]):
            if lay["z_lo"] < zp < lay["z_hi"]:
                lines.append(occ.addLine(occ.addPoint(0, zp, 0), occ.addPoint(lay["r_in"], zp, 0)))
        if 0 < lay["r_plane"] < lay["r_in"]:                         # side of the closed measurement box
            lines.append(occ.addLine(occ.addPoint(lay["r_plane"], lay["z_plane_bottom"], 0), occ.addPoint(lay["r_plane"], lay["z_plane_top"], 0)))
        occ.remove([(2, domain_box)], recursive=True)
        objs = [(2, t) for t, _, _ in base]
        tool_dimtags = [dt for dt, _, _ in tools] + [(1, l) for l in lines]
        out, outmap = occ.fragment(objs, tool_dimtags)
        occ.synchronize()
        sources = {}
        for k, produced in enumerate(outmap):
            for dim, tag in produced:
                if dim == 2:
                    sources.setdefault(tag, []).append(k)
        surf_tag, surf_mat, surf_kind = {}, {}, {}
        for tag, srcs in sources.items():
            part_srcs = [tools[k - len(objs)][2] for k in srcs if len(objs) <= k < len(objs) + len(tools) and tools[k - len(objs)][1] == "part"]
            in_source = any(len(objs) <= k < len(objs) + len(tools) and tools[k - len(objs)][1] == "source" for k in srcs)
            base_src = [k for k in srcs if k < len(objs)]
            kind = base[base_src[0]][2] if base_src else "inner"
            mat = model["parts"][max(part_srcs)]["material"] if part_srcs else (base[base_src[0]][1] if base_src else model["background"])
            surf_mat[tag], surf_kind[tag] = mat, kind
            surf_tag[tag] = 1 + names.index(mat) + (TAG_SOURCE if in_source else 0)
        e = 1e-6 * max(lay["r_out"], 1.0)
        axis, wall = [], []
        for dim, tag in gmsh.model.getEntities(1):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(dim, tag)
            if abs(x0) < e and abs(x1) < e:
                axis.append(tag)
            elif abs(x0 - lay["r_out"]) < e and abs(x1 - lay["r_out"]) < e:
                wall.append(tag)
            elif (abs(y0 - lay["z_bot"]) < e and abs(y1 - lay["z_bot"]) < e) or (abs(y0 - lay["z_top"]) < e and abs(y1 - lay["z_top"]) < e):
                wall.append(tag)
        gmsh.model.addPhysicalGroup(1, axis, TAG_AXIS, "axis")
        gmsh.model.addPhysicalGroup(1, wall, TAG_WALL, "wall")
        for t in sorted(set(surf_tag.values())):
            surfs = [s for s, v in surf_tag.items() if v == t]
            nm = names[(t - 1) % TAG_SOURCE] + (" (Emitterkasten)" if t > TAG_SOURCE else "")
            gmsh.model.addPhysicalGroup(2, surfs, t, nm)
        size_of = {n: fg._size_of(model["materials"][n], ms["lam_min_nm"], ms["lam_max_nm"], ms) for n in names}
        n_pml = float(ms.get("pml_cells_per_wavelength", 0.0) or 0.0)
        pml_size = {}
        if n_pml > 0:
            for n in names:
                try:
                    pml_size[n] = ms["lam_min_nm"] / (n_pml * max(fm.nk(model["materials"][n], ms["lam_min_nm"])[0], 1.0))
                except Exception:
                    pass
        point_size, point_mats = {}, {}
        for tag, mat in surf_mat.items():
            h = min(size_of[mat], pml_size.get(mat, 1e9)) if surf_kind[tag] == "pml" else size_of[mat]
            if surf_tag[tag] > TAG_SOURCE:
                h = min(h, max(2.0 * float(em["sigma_nm"]), 1.0))
            for dim, p in gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False, recursive=True):
                point_size[p] = min(point_size.get(p, 1e9), h)
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
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        msh = out_dir / "mesh.msh"
        gmsh.write(str(msh))
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
        a_, b_, c_ = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
        lens = np.stack([np.linalg.norm(b_ - a_, axis=1), np.linalg.norm(c_ - b_, axis=1), np.linalg.norm(a_ - c_, axis=1)], axis=1)
        np.savez(out_dir / "mesh_plot.npz", xy=xy, tri=tris, mat=mats, names=np.array(names), order=2 if curved else 1)
        problems = fg.reader_check(msh)
        if problems:
            raise RuntimeError("Das Netz erfüllt die Anforderungen des hpfem-Lesers nicht: " + "; ".join(problems))
        n_source = int(sum(1 for t in surf_tag.values() if t > TAG_SOURCE))
        if with_source and n_source == 0:
            raise RuntimeError("Gmsh: kein Emitterkasten im Netz")
        return dict(cells=int(len(tris)), nodes=int(len(ntags)), order=2 if curved else 1, edge_min_nm=float(lens.min()), edge_max_nm=float(lens.max()),
                    cells_per_material={names[i]: int((mats == i).sum()) for i in sorted(set(mats))},
                    sizes_nm={n: round(float(h), 2) for n, h in size_of.items()}, file=str(msh), task=task, source_box_nm=a if with_source else None)
    finally:
        gmsh.finalize()


def est_dofs(cells, p):
    return int(1.4 * 0.9 * (p + 1) * (p + 2) * cells)


def m_hint(model):
    """Text hint for the azimuthal order."""
    return ("m = 0: Moden ohne Azimutabhängigkeit (TE₀/TM₀, axialer Dipol); m = 1: Grundmode von Säulen (HE₁₁), seitlicher Dipol auf der Achse; "
            "große m: Flüstergalerie-Moden von Scheiben und Ringen (m ≈ 2π n_eff R / λ).")


def whispering_m(model, lam_nm):
    """Rough azimuthal order of a whispering-gallery mode at lam_nm: 2 pi n R / lam of the outermost part."""
    if not model["parts"]:
        return None
    p = max(model["parts"], key=lambda q: part_bbox(q)[1])
    R = part_bbox(p)[1]
    try:
        n = fm.nk(model["materials"][p["material"]], lam_nm)[0]
    except Exception:
        return None
    return int(round(2 * math.pi * 0.85 * n * R / lam_nm))
