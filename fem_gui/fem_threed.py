"""3D structures in the FEM model builder (hp-FEM 0.5): models, job, checks and figures. Pure Python (no hpfem needed); worker: fem_threed_worker.py,
user interface: fem_threed_app.py.

Three tasks with the 3D solvers of the library: ``scatter`` (Scattering3D: plane wave on a sphere or cube in a PML box, cross sections, Mie series as the
reference for the sphere), ``resonance`` (Resonance3D: complex eigenfrequencies, Q) and ``bands`` (BandStructure3D: simple-cubic photonic crystal).
Meshes come from the generators of the library (box_with_ball: curved sphere; box: cube aligned with the grid), no Gmsh needed.

Extensions (hp-FEM 0.5): (1) a planar layer stack under the object (``stack``: LayerStack3D as the background of the scattered-field formulation, oblique
incidence, s / p, R / T / A of the bare stack), (2) task ``ports``: waveguides and couplers between modal ports (fem_ports3d.py), (3) hp-adaptive refinement of
the scattering problem (``adaptive``: AdaptiveMesh3D, residual estimator, Doerfler marking, hp decision by prediction), (4) Gmsh meshes: own geometry (sphere,
cylinder, block; fem_threed_gmsh.py) or an uploaded .msh file with a tag to material table (``mesh_kind``).
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import fem_materials as fm
import fem_ports3d as fp3
import fem_run as fr
import fem_threed_gmsh as gm

TASKS = {"scatter": "Streuung (Wirkungsquerschnitte, optional auf Schichtstapel, hp-adaptiv)", "resonance": "Resonanzen (Quasi-Normalmoden, Q-Faktor)",
         "bands": "Bandstruktur (kubisches photonisches Kristallgitter)", "ports": "Wellenleiter und Ports (S-Parameter)"}
MESH_KINDS = {"builtin": "Bibliotheksnetz (Kugel, Würfel)", "gmsh": "Gmsh-Geometrie (Kugel, Zylinder, Quader)", "file": "Eigene Gmsh-Datei (.msh)"}
POLS = {"p": "p (TM)", "s": "s (TE)"}
UNIT_SCALE = {"nm": 1e-9, "µm": 1e-6, "mm": 1e-3, "m": 1.0}
AIR, GLASS = "Luft", "Glas"
SHAPES = {"sphere": "Kugel (gekrümmte Elemente)", "cube": "Würfel (an das Gitter angepasst)"}
SYM = {False: "ganzes Modell", True: "Viertelmodell (x > 0, y > 0, Spiegelsymmetrie)"}
GB = 1e9
OBJ = "Objekt"
OBJ_TAG = gm.TAG_OBJ


def default_model():
    return {"name": "Kugel in Luft", "task": "scatter", "shape": "sphere", "radius_nm": 100.0,
            "materials": {OBJ: {"type": "index", "n": 2.0, "k": 0.0}, AIR: {"type": "index", "n": 1.0, "k": 0.0}, GLASS: {"type": "index", "n": 1.5, "k": 0.0}},
            "object": OBJ,
            "half_width": 2.0, "outer": 3.0, "n": 2, "order": 2, "quarter": True, "fields": True, "max_gb": 16.0,
            "sweep": {"start": 314.159, "stop": 314.159, "count": 1},
            "resonance": {"lam0_nm": 700.0, "num_modes": 3, "krylov": 0},
            "bands": {"radius": 0.25, "eps": 12.0, "eps_bg": 1.0, "bands": 6, "n_seg": 2, "cells": 4, "krylov": 0},
            "mesh_kind": "builtin",
            "gmsh": {"kind": "sphere", "height_nm": 200.0, "lx_nm": 200.0, "ly_nm": 200.0, "lz_nm": 100.0, "hw_nm": 250.0, "pml_nm": 200.0, "clear_nm": 150.0,
                     "h_obj_nm": 45.0, "h_far_nm": 140.0, "grade_nm": 250.0, "order": 2},
            "stack": {"enabled": False, "layers": [], "substrate": GLASS, "incidence": AIR, "gap_nm": 50.0, "depth_nm": 150.0, "angle_deg": 0.0, "pol": "p",
                      "azimuth_deg": 0.0},
            "file": {"name": "", "scale": "nm", "tag_material": {}, "object_tags": [], "pml_nm": 200.0, "tags": {}},
            "adaptive": {"enabled": False, "theta": 0.5, "max_steps": 4, "max_dofs": 15000, "tol": 0.0},
            "p3": fp3.default_model()}


def ensure(m):
    """Fills keys that models of older versions lack (loaded files); returns m."""
    d = default_model()
    for k, v in d.items():
        if k not in m:
            m[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and k in ("gmsh", "stack", "file", "adaptive"):
            for kk, vv in v.items():
                m[k].setdefault(kk, copy.deepcopy(vv))
    for name, spec in ((AIR, {"type": "index", "n": 1.0, "k": 0.0}), (GLASS, {"type": "index", "n": 1.5, "k": 0.0})):
        m["materials"].setdefault(name, spec)
    return m


def presets():
    p = {}
    m = default_model()
    m["name"] = "Dielektrische Kugel n = 2, ka = 2 (Mie-Referenz)"
    p[m["name"]] = m
    m = default_model()
    m.update(name="Kugel n = 2, Wellenlängendurchlauf", sweep={"start": 250.0, "stop": 500.0, "count": 4}, order=2, n=1)
    p[m["name"]] = m
    m = default_model()
    m.update(name="Hochindexkugel (ε = 12): Resonanzen", task="resonance", n=1, order=2, quarter=False,
             materials={OBJ: {"type": "eps", "re": 12.0, "im": 0.0}}, resonance={"lam0_nm": 700.0, "num_modes": 3, "krylov": 0})
    p[m["name"]] = m
    m = default_model()
    m.update(name="Würfel n = 2 (Streuquerschnitt, ohne Referenz)", shape="cube", n=2, order=2)
    p[m["name"]] = m
    m = default_model()
    m.update(name="Kubischer Kristall: Würfel ε = 12 in Luft", task="bands", shape="cube",
             bands={"radius": 0.25, "eps": 12.0, "eps_bg": 1.0, "bands": 6, "n_seg": 2, "cells": 4, "krylov": 0}, order=2)
    p[m["name"]] = m
    # --- hp-adaptive
    m = default_model()
    m.update(name="Kugel n = 2: hp-adaptiv (Viertelmodell, klein)", n=1, order=2, adaptive={"enabled": True, "theta": 0.5, "max_steps": 4, "max_dofs": 12000, "tol": 0.0})
    m["half_width"], m["outer"] = 2.0, 3.0
    p[m["name"]] = m
    # --- Gmsh geometries and layer stacks (sizes in nm; deliberately coarse so that a run takes seconds to a few minutes)
    m = default_model()
    m.update(name="Kugel n = 2, Gmsh-Netz (Mie-Referenz, ca. 1,5 min)", mesh_kind="gmsh", order=2, quarter=False, sweep={"start": 628.3, "stop": 628.3, "count": 1})
    m["gmsh"].update(hw_nm=200.0, pml_nm=300.0, clear_nm=100.0, h_obj_nm=60.0, h_far_nm=150.0, grade_nm=200.0)
    p[m["name"]] = m
    m = default_model()
    m.update(name="Silizium-Zylinder auf Glas, schräger Einfall (p, 30°; grob, p = 1)", mesh_kind="gmsh", order=1, quarter=False, radius_nm=100.0,
             sweep={"start": 700.0, "stop": 700.0, "count": 1}, materials={OBJ: {"type": "library", "name": "Si"}, AIR: {"type": "index", "n": 1.0, "k": 0.0},
                                                                         GLASS: {"type": "index", "n": 1.5, "k": 0.0}})
    m["gmsh"].update(kind="cylinder", height_nm=120.0, hw_nm=250.0, pml_nm=300.0, clear_nm=150.0, h_obj_nm=45.0, h_far_nm=240.0, grade_nm=250.0)
    m["stack"].update(enabled=True, layers=[], substrate=GLASS, gap_nm=0.0, depth_nm=250.0, angle_deg=30.0, pol="p")
    p[m["name"]] = m
    m = default_model()
    m.update(name="Gold-Kugel auf SiO2-Schicht über Glas (Stapel, grob)", mesh_kind="gmsh", order=1, quarter=False, radius_nm=40.0,
             sweep={"start": 560.0, "stop": 560.0, "count": 1},
             materials={OBJ: {"type": "library", "name": "Au"}, AIR: {"type": "index", "n": 1.0, "k": 0.0}, GLASS: {"type": "index", "n": 1.5, "k": 0.0},
                        "SiO2-Schicht": {"type": "index", "n": 1.45, "k": 0.0}})
    m["gmsh"].update(kind="sphere", hw_nm=200.0, pml_nm=300.0, clear_nm=120.0, h_obj_nm=20.0, h_far_nm=180.0, grade_nm=150.0)
    m["stack"].update(enabled=True, layers=[{"material": "SiO2-Schicht", "thickness_nm": 100.0}], substrate=GLASS, gap_nm=20.0, depth_nm=150.0, angle_deg=0.0, pol="p")
    p[m["name"]] = m
    # --- waveguides
    for name, pm in fp3.presets().items():
        m = default_model()
        m.update(name=name, task="ports", p3=pm)
        p[name] = m
    return p


def wavelengths(m):
    s = m["sweep"]
    return [float(v) for v in np.linspace(s["start"], s["stop"], int(s["count"]))] if int(s["count"]) > 1 else [float(s["start"])]


def lam_check(m):
    """Wavelengths at which the object material is evaluated."""
    if m["task"] == "resonance":
        return [float(m["resonance"]["lam0_nm"])]
    return wavelengths(m) if m["task"] == "scatter" else []


def object_material(m):
    """Name of the material of the object: the chosen one; for a mesh file the material of its first object tag."""
    if m["mesh_kind"] == "file" and m["task"] != "bands":
        f = m["file"]
        for tag in f["object_tags"]:
            nm = f["tag_material"].get(str(tag))
            if nm in m["materials"]:
                return nm
    return m["object"]


def object_cell_tags(m):
    """Cell tags of the object(s) for the mesh view (orange): the object tag; the cores of a waveguide; the chosen tags of a mesh file."""
    if m["task"] == "ports":
        return (fp3.TAG_A, fp3.TAG_B)
    if m["task"] != "bands" and m["mesh_kind"] == "file":
        return tuple(int(x) for x in m["file"]["object_tags"])
    return (OBJ_TAG,)


def eps_list(m):
    spec = m["materials"][object_material(m)]
    return [fm.eps_at(spec, lam) for lam in lam_check(m)]


def stack_regions(m):
    """Material names from the top: incidence medium, the layers, the substrate."""
    st = m["stack"]
    return [st["incidence"]] + [layer["material"] for layer in st["layers"]] + [st["substrate"]]


def region_eps(m, name, lams):
    return [[float(e.real), float(e.imag)] for e in (fm.eps_at(m["materials"][name], lam) for lam in lams)]


def lam_min_nm(m):
    ls = lam_check(m)
    return min(ls) if ls else 500.0


def index_max(m):
    """Largest |n| of the object and of the stack regions at the shortest wavelength (for the mesh size suggestions)."""
    lam = lam_min_nm(m)
    names = [object_material(m)] + (stack_regions(m) if m["stack"]["enabled"] else [])
    vals = []
    for nm in names:
        if nm in m["materials"]:
            vals.append(abs(np.sqrt(fm.eps_at(m["materials"][nm], lam))))
    return max(vals + [1.0])


def suggest_sizes(m):
    """Mesh sizes [nm] for the Gmsh geometry: fine near the object (about lambda / (3 n) and a third of the radius), coarse far away (lambda / 2.5), PML half a wavelength."""
    lam = lam_min_nm(m)
    lam_max = max(lam_check(m) or [lam])
    r = float(m["radius_nm"])
    n_obj = abs(np.sqrt(fm.eps_at(m["materials"][object_material(m)], lam)))
    h_obj = max(min(r / 2.5, lam / (3.0 * max(n_obj, 1.0))), 1.0)
    h_far = lam / 2.5
    return dict(h_obj_nm=round(h_obj, 1), h_far_nm=round(h_far, 1), grade_nm=round(max(1.5 * h_far, 2 * r), 1), pml_nm=round(0.5 * lam_max, 1), hw_nm=round(max(1.5 * r, 0.3 * lam), 1))


def gmsh_params(m):
    """The parameter dict of fem_threed_gmsh.scene (nm) for the model."""
    g, st = m["gmsh"], m["stack"]
    d = dict(kind=g["kind"], r=float(m["radius_nm"]), h=float(g["height_nm"]), lx=float(g["lx_nm"]), ly=float(g["ly_nm"]), lz=float(g["lz_nm"]), hw=float(g["hw_nm"]),
             pml=float(g["pml_nm"]), clear=float(g["clear_nm"]), h_obj=float(g["h_obj_nm"]), h_far=float(g["h_far_nm"]), grade=float(g["grade_nm"]), order=int(g["order"]),
             stack=bool(st["enabled"]) and m["task"] == "scatter")
    if d["stack"]:
        lam = lam_min_nm(m)
        d.update(layers=[float(x["thickness_nm"]) for x in st["layers"]], depth=float(st["depth_nm"]), gap=float(st["gap_nm"]))
        d["region_n"] = [float(abs(np.sqrt(fm.eps_at(m["materials"][nm], lam)))) for nm in stack_regions(m)]
    else:
        d["region_n"] = [1.0]
    return d


def object_dims(m):
    """(half extent in x, y, height) of the Gmsh object [nm]."""
    return gm.object_extent(gmsh_params(m))


def _frac_int(x, tol=1e-9):
    return abs(x - round(x)) < tol


def validate_gmsh(m):
    out = []
    g, st = m["gmsh"], m["stack"]
    r = float(m["radius_nm"])
    if g["kind"] in ("sphere", "cylinder") and r <= 0:
        out.append(("error", "Radius > 0."))
    if g["kind"] == "cylinder" and g["height_nm"] <= 0:
        out.append(("error", "Höhe > 0."))
    if g["kind"] == "block" and min(g["lx_nm"], g["ly_nm"], g["lz_nm"]) <= 0:
        out.append(("error", "Kantenlängen > 0."))
    if g["h_obj_nm"] <= 0 or g["h_far_nm"] <= 0 or g["pml_nm"] <= 0:
        out.append(("error", "Maschenweiten und PML-Dicke > 0."))
    if g["h_obj_nm"] > g["h_far_nm"]:
        out.append(("warning", "Die Maschenweite am Objekt ist größer als die ferne: es gibt dann keine Verfeinerung zum Objekt."))
    ox, oy, oh = gm.object_extent({"kind": g["kind"], "r": r, "h": g["height_nm"], "lx": g["lx_nm"], "ly": g["ly_nm"], "lz": g["lz_nm"]})
    if g["hw_nm"] < 1.05 * max(ox, oy):
        out.append(("error", f"Der Innenraum (halbe Breite {g['hw_nm']:g} nm) muss größer sein als das Objekt (halbe Breite {max(ox, oy):g} nm)."))
    elif g["hw_nm"] < max(ox, oy) + 0.5 * g["h_far_nm"]:
        out.append(("warning", f"Der Abstand zwischen Objekt ({max(ox, oy):g} nm) und PML-Beginn ({g['hw_nm']:g} nm) ist kleiner als eine halbe ferne Zelle: das Nahfeld wird vom PML gestört."))
    if g["clear_nm"] <= 0:
        out.append(("error", "Abstand über dem Objekt bis zum PML > 0."))
    lam = lam_min_nm(m)
    if g["h_far_nm"] > lam / 1.8:
        out.append(("warning", f"Ferne Maschenweite {g['h_far_nm']:g} nm bei λ = {lam:g} nm: weniger als etwa 1,8 Zellen je Wellenlänge, das Ergebnis ist kaum aufgelöst (Vorschlag: λ/2,5)."))
    nmax = index_max(m)
    if g["h_obj_nm"] * nmax > lam / 1.8:
        out.append(("warning", f"Maschenweite am Objekt {g['h_obj_nm']:g} nm bei n ≈ {nmax:.2f}: unter 1,8 Zellen je Wellenlänge im Material."))
    if g["pml_nm"] < 0.3 * max(lam_check(m) or [lam]):
        out.append(("warning", "Das PML ist dünner als etwa 0,3 λ: die Reflexion am Rand kann das Ergebnis stören."))
    if m["task"] == "scatter" and st["enabled"]:
        names = stack_regions(m)
        for nm in names:
            if nm not in m["materials"]:
                out.append(("error", f"Material „{nm}“ des Schichtstapels fehlt."))
        if not any(e[0] == "error" for e in out):
            for lam_i in lam_check(m):
                if abs(fm.eps_at(m["materials"][st["incidence"]], lam_i).imag) > 1e-9:
                    out.append(("error", "Das Einfallsmedium muss verlustfrei sein."))
                    break
            if any(abs(fm.eps_at(m["materials"][nm], lam_min_nm(m)).imag) > 1e-9 for nm in names[1:2] if g["kind"] != "sphere") and st["gap_nm"] <= 0:
                out.append(("warning", "Das Objekt berührt eine verlustbehaftete Schicht: die Streuquerschnitte aus dem Fluss um das Objekt sind dort nicht eindeutig."))
        if any(x["thickness_nm"] <= 0 for x in st["layers"]):
            out.append(("error", "Schichtdicken > 0."))
        if st["gap_nm"] < 0 or st["depth_nm"] <= 0:
            out.append(("error", "Abstand ≥ 0 und Tiefe des Substrats bis zum PML > 0."))
        if g["kind"] == "sphere" and st["gap_nm"] < 0.1 * r:
            out.append(("warning", "Eine Kugel mit Abstand unter 0,1 r zur Oberfläche erzeugt sehr kleine Zellen im Spalt; ein Abstand von mindestens r/4 ist ratsam."))
        if not 0.0 <= st["angle_deg"] < 90.0:
            out.append(("error", "Einfallswinkel zwischen 0° und 90°."))
        if st["angle_deg"] > 0 and g["hw_nm"] < ox + 1.0 * g["h_far_nm"]:
            out.append(("warning", "Bei schrägem Einfall wirkt die Wand aus PEC (Rand der Box) nur über das PML: die Box sollte mindestens das Objekt plus ein PML breit sein."))
        out.append(("warning", "PML und Schichtstapel: das PML ist für den Index 1 ausgelegt (Substrat und Schichten haben höheren Index); Substrat und Schichten dicker als das PML-Gebiet, sonst bleibt die Reflexion am Rand."))
    return out


def validate_file(m):
    out = []
    f = m["file"]
    if not f.get("name"):
        return [("error", "Bitte eine .msh-Datei hochladen (Gmsh 4.1 ASCII, Reiter 2) und lesen.")]
    if not f.get("tags"):
        return [("error", "Die Datei wurde noch nicht gelesen: „Datei lesen“ im Reiter 2.")]
    cells = f["tags"].get("cells", {})
    if not f["object_tags"]:
        out.append(("error", "Mindestens eine Zellmarke als Streukörper wählen."))
    missing = [tag for tag in cells if f["tag_material"].get(tag) not in m["materials"]]
    if missing:
        out.append(("error", f"Zellmarken ohne Material: {', '.join(missing)} (Tabelle im Reiter 2; Luft ist ein Material der Liste)."))
    if f["pml_nm"] <= 0:
        out.append(("error", "PML-Dicke > 0."))
    else:
        bb = f["tags"].get("bbox")
        if bb:
            ext = np.array(bb[1]) - np.array(bb[0])              # metres (the worker scales the file)
            if 2 * f["pml_nm"] * 1e-9 >= ext.min() * 0.9:
                out.append(("error", "Die PML-Dicke ist zu groß für die Ausdehnung des Netzes (der Innenraum wäre leer)."))
    out.append(("warning", "Die Seitenflächen des Netzes werden als PEC-Wand behandelt; das PML-Gebiet ist der Rand der Box (die Randflächen ohne Marke bekommen die Marken 1 bis 6)."))
    return out


def validate(m):
    """[(level, text)] for the model."""
    out = []
    t = m["task"]
    if t == "ports":
        return fp3.validate(m["p3"])
    if t != "bands" and m["mesh_kind"] == "file":
        out += validate_file(m)
        if any(x[0] == "error" for x in out):
            return out
    if t != "bands" and m["mesh_kind"] == "gmsh":
        out += validate_gmsh(m)
    if m["stack"]["enabled"] and t != "ports" and (t != "scatter" or m["mesh_kind"] != "gmsh"):
        out.append(("warning", "Der Schichtstapel gilt nur für die Streuung mit einer Gmsh-Geometrie (Netz mit Ebenen an den Grenzflächen) und wird hier ignoriert."))
    if m["adaptive"]["enabled"]:
        if t != "scatter":
            out.append(("error", "Die hp-Adaptivität gibt es nur für die Streuung."))
        a_ = m["adaptive"]
        if a_["max_steps"] < 1 or not 0.05 <= a_["theta"] <= 1.0 or a_["max_dofs"] < 100:
            out.append(("error", "hp-Adaptivität: Schritte ≥ 1, θ zwischen 0,05 und 1, Obergrenze Unbekannte ≥ 100."))
        if m["order"] > 3:
            out.append(("warning", "Startordnung über 3 mit hp-Adaptivität: der Speicher wächst schnell; die Obergrenze für Unbekannte greift vor dem nächsten Schritt."))
        if m["mesh_kind"] == "builtin" and m["shape"] == "cube":
            out.append(("warning", "Der Würfel ist ein Gitternetz mit Zellen auf der Würfelfläche: die Verfeinerung teilt rote Tetraeder, die Würfelkante bleibt exakt."))
    if t == "bands":
        b = m["bands"]
        if not 0.02 <= b["radius"] < 0.5:
            out.append(("error", "Radius bzw. halbe Kante muss zwischen 0,02 und 0,5 (in Einheiten a) liegen."))
        if m["shape"] == "sphere" and not _frac_int(0.5 / b["radius"] * m["n"]):
            out.append(("error", f"Für die Kugel muss (0,5 / r) · n ganzzahlig sein; hier {0.5 / b['radius'] * m['n']:.3f}. Mit r = 0,25 geht jedes n, mit r = 0,3 jedes n = 3·j."))
        if m["shape"] == "cube":
            c = int(b["cells"])
            if c % 2:
                out.append(("error", "Zellen je Kante: gerade Zahl."))
            elif not _frac_int(b["radius"] * 2 * c):
                out.append(("error", "Die halbe Kante des Würfels muss ein ganzes Vielfaches der Zellkante 1/Zellen sein (r · 2 · Zellen ganzzahlig)."))
        if b["eps"] <= 0 or b["eps_bg"] <= 0:
            out.append(("error", "ε muss positiv sein (verlustfreie Bandstruktur)."))
        if m["shape"] == "sphere" and b["radius"] >= 0.5:
            out.append(("error", "Die Kugel muss in die Zelle passen (r < 0,5 a)."))
        return out
    a = float(m["radius_nm"])
    if m["mesh_kind"] == "builtin":
        if a <= 0:
            out.append(("error", "Radius bzw. halbe Kante > 0."))
        hw, outer, n = float(m["half_width"]), float(m["outer"]), int(m["n"])
        if not (hw > 1.0 and outer > hw):
            out.append(("error", "Es muss gelten: 1 < Innenraum (halbe Breite) < Außenrand; zwischen Innenraum und Außenrand liegt das PML."))
        if not (_frac_int(hw * n) and _frac_int(outer * n)):
            out.append(("error", f"Innenraum und Außenrand mal Zellen je Radius (n) müssen ganzzahlig sein (hier {hw * n:g} und {outer * n:g})."))
        if hw - 1.0 < 0.5:
            out.append(("warning", "Weniger als ein halber Radius Abstand zwischen Objekt und PML: das Nahfeld wird vom PML gestört."))
        if m["quarter"] and t == "resonance":
            out.append(("error", "Resonanzen brauchen das ganze Modell (die Moden sind nicht alle symmetrisch)."))
    if m["order"] < 1:
        out.append(("error", "Ordnung ≥ 1."))
    try:
        for lam, e in zip(lam_check(m), eps_list(m)):
            if e.imag < -1e-12:
                out.append(("warning", f"Im ε < 0 bei {lam:g} nm: bei exp(−iωt) bedeutet das Verstärkung."))
    except Exception as exc:
        out.append(("error", f"Material: {exc}"))
    if t == "scatter":
        if not wavelengths(m):
            out.append(("error", "Keine Wellenlänge."))
        if m["shape"] == "cube" and m["mesh_kind"] == "builtin":
            out.append(("warning", "Für den Würfel gibt es keine analytische Referenz; die Kanten sind treppenfrei, aber die Felder an den Ecken sind singulär (langsamere Konvergenz)."))
        if m["mesh_kind"] == "gmsh" and m["gmsh"]["kind"] != "sphere" or m["stack"]["enabled"]:
            out.append(("warning", "Keine Mie-Referenz für diese Geometrie; Q ist auf die Grundfläche (Projektion auf die x-y-Ebene) bezogen. Prüfen Sie die Netzkonvergenz (hp-adaptive Verfeinerung zeigt sie)."))
    if t == "resonance" and m["shape"] == "cube" and m["mesh_kind"] == "builtin":
        out.append(("warning", "Für den Würfel gibt es keine analytische Referenz."))
    return out


def job(m, max_gb=None):
    t = m["task"]
    if t == "ports":
        return fp3.job(m["p3"], max_gb if max_gb is not None else m["max_gb"])
    j = {"task": t, "name": m["name"], "shape": m["shape"], "n": int(m["n"]), "order": int(m["order"]), "half_width": float(m["half_width"]),
         "outer": float(m["outer"]), "max_gb": float(max_gb if max_gb is not None else m["max_gb"]), "fields": bool(m["fields"])}
    if t == "bands":
        b = m["bands"]
        j.update(radius=float(b["radius"]), eps=float(b["eps"]), eps_bg=float(b["eps_bg"]), bands=int(b["bands"]), n_seg=int(b["n_seg"]), cells=int(b["cells"]),
                 krylov=int(b["krylov"]), half_width=0.5, outer=0.5)
        return j
    j["radius_nm"] = float(m["radius_nm"])
    kind = m["mesh_kind"]
    j["mesh_kind"] = kind
    j["quarter"] = bool(m["quarter"]) and t == "scatter" and kind == "builtin"
    if t == "scatter":
        j["wavelengths_nm"] = wavelengths(m)
        j["eps"] = [[e.real, e.imag] for e in eps_list(m)]
    else:
        r = m["resonance"]
        e = eps_list(m)[0]
        j.update(lam0_nm=float(r["lam0_nm"]), num_modes=int(r["num_modes"]), krylov=int(r["krylov"]), eps=[e.real, e.imag])
    lams = lam_check(m)
    if kind == "gmsh":
        j["g"] = gmsh_params(m)
        st = m["stack"]
        if st["enabled"] and t == "scatter":
            j["stack"] = dict(eps=[region_eps(m, nm, lams) for nm in stack_regions(m)], thickness_nm=[float(x["thickness_nm"]) for x in st["layers"]], pol=st["pol"],
                              angle_deg=float(st["angle_deg"]), azimuth_deg=float(st["azimuth_deg"]))
    if kind == "file":
        f = m["file"]
        j["file"] = dict(path=f.get("path", ""), scale=UNIT_SCALE[f["scale"]], object_tags=[int(x) for x in f["object_tags"]], pml_nm=float(f["pml_nm"]),
                         tag_eps={str(tag): region_eps(m, nm, lams) for tag, nm in f["tag_material"].items() if nm in m["materials"]})
    ad = m["adaptive"]
    if ad["enabled"] and t == "scatter":
        j["adaptive"] = dict(enabled=True, theta=float(ad["theta"]), max_steps=int(ad["max_steps"]), max_dofs=int(ad["max_dofs"]), tol=float(ad["tol"]))
    return j


def worker_script(jb):
    """The worker script of a job (ports: fem_ports3d_worker.py, everything else fem_threed_worker.py)."""
    return fp3.SCRIPT if jb.get("task") == "ports" else "fem_threed_worker.py"


def run_inspect(py, repo, folder, path, unit, task="scatter", threads=0, timeout=120):
    """Tags of a mesh file via the worker: dict n_cells, cells {tag: count}, facets {tag: count}, bbox [m], geometry_order; or {'error': ...}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    jb = {"task": task, "path": str(path), "scale": UNIT_SCALE[unit], "file": {"path": str(path), "scale": UNIT_SCALE[unit]}}
    (folder / "job.json").write_text(json.dumps(jb), encoding="utf-8")
    try:
        r = subprocess.run([str(py), str(fr.HERE / worker_script(jb)), str(folder), "inspect"], cwd=fr._cwd(repo, folder), env=fr.worker_env(repo, threads),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "Zeitüberschreitung beim Lesen."}
    for line in r.stdout.splitlines():
        if line.startswith("INSPECT"):
            return json.loads(line[len("INSPECT"):])
    return {"error": (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "keine Ausgabe"}


def run_estimate(py, repo, folder, jb, threads=0, timeout=300):
    """Memory estimate of the job from the library (meshes the model, no solve): dict cells, dofs, bytes, human, text; or {'error': ...}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(jb), encoding="utf-8")
    try:
        r = subprocess.run([str(py), str(fr.HERE / worker_script(jb)), str(folder), "estimate"], cwd=fr._cwd(repo, folder), env=fr.worker_env(repo, threads),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "Zeitüberschreitung beim Vernetzen."}
    for line in r.stdout.splitlines():
        if line.startswith("ESTIMATE"):
            return json.loads(line[len("ESTIMATE"):])
    return {"error": (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "keine Ausgabe"}


def available_gb():
    """Available memory [GB] from /proc/meminfo (None if unknown)."""
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) * 1024 / GB
    except OSError:
        pass
    return None


def time_hint(dofs):
    """Rough solver time per solve with the only backend of the wheel (SparseLU, one thread): about 70 s at 18 000 unknowns, growing with ~ N^1.7."""
    return 70.0 * (max(dofs, 1) / 18264.0) ** 1.7


# ===================================================================================================================== figures
def section_polygons(vertices, cells, axis, coord):
    """Polygons (list of (k, 2) arrays in the two other coordinates) and the cell index of each tetrahedron cut by the plane x_axis = coord."""
    keep = [i for i in range(3) if i != axis]
    v = np.asarray(vertices)
    c = np.asarray(cells)
    d = v[c, axis] - coord
    polys, idx = [], []
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    cand = np.nonzero((d.min(axis=1) <= 0) & (d.max(axis=1) >= 0))[0]
    for i in cand:
        pts = []
        for a, b in pairs:
            da, db = d[i, a], d[i, b]
            if da == db:
                continue
            if (da <= 0 <= db) or (db <= 0 <= da):
                s = da / (da - db)
                pts.append((v[c[i, a]] + s * (v[c[i, b]] - v[c[i, a]]))[keep])
        if len(pts) >= 3:
            p = np.array(pts)
            ctr = p.mean(axis=0)
            order = np.argsort(np.arctan2(p[:, 1] - ctr[1], p[:, 0] - ctr[0]))
            polys.append(p[order])
            idx.append(int(i))
    return polys, idx


_PALETTE = ["#cfe3f5", "#d9ead3", "#e8dcf0", "#fff2cc", "#d0e0e3", "#f9cb9c", "#ead1dc", "#c9daf8"]


def _tag_colour(tag, obj_tags, all_tags):
    if tag in obj_tags:
        return "#f4b183"
    others = [x for x in all_tags if x not in obj_tags]
    return _PALETTE[others.index(tag) % len(_PALETTE)] if tag in others else "#cfe3f5"


def fig_mesh(npz, axis=1, coord=0.0, unit=1e-9, unit_label="nm", title="", obj_tags=(2,), by="tag"):
    """Cut through the mesh (straight tetrahedron edges; the curved surfaces are drawn exact in the solver) coloured by material tag (orange: object) or, after an
    hp-adaptive run (by = "order"), by the polynomial order of the cell."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    verts, cells, tags = npz["vertices"], npz["cells"], npz["tags"]
    orders = npz["orders"] if "orders" in getattr(npz, "files", npz) else None
    polys, idx = section_polygons(verts, cells, axis, coord)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    if polys:
        if by == "order" and orders is not None:
            cmap = plt.get_cmap("viridis")
            lo, hi = int(orders.min()), int(orders.max())
            cols = [cmap((int(orders[i]) - lo) / max(hi - lo, 1)) for i in idx]
        else:
            cols = [_tag_colour(int(tags[i]), obj_tags, sorted({int(x) for x in tags})) for i in idx]
        ax.add_collection(PolyCollection([p / unit for p in polys], facecolors=cols, edgecolors="0.35", linewidths=0.4))
        allp = np.vstack(polys) / unit
        ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
        ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax.set_aspect("equal")
    names = "xyz"
    keep = [names[i] for i in range(3) if i != axis]
    ax.set_xlabel(f"{keep[0]} ({unit_label})")
    ax.set_ylabel(f"{keep[1]} ({unit_label})")
    ax.set_title(title or f"Schnitt {names[axis]} = {coord / unit:g} {unit_label} " + ("(Farbe: Ordnung p)" if by == "order" and orders is not None else "(orange: Objekt)"))
    fig.tight_layout()
    return fig


def mirror(xs, ys, f, mirror_x=True, mirror_y=False):
    """Quarter-plane |E|^2 on x, y >= 0 to the full plane (|E|^2 is even under both mirrors)."""
    f = np.asarray(f)
    X, Y = np.asarray(xs), np.asarray(ys)
    if mirror_x:
        f = np.concatenate([f[:0:-1], f], axis=0)
        X = np.concatenate([-X[:0:-1], X])
    if mirror_y:
        f = np.concatenate([f[:, :0:-1], f], axis=1)
        Y = np.concatenate([-Y[:0:-1], Y])
    return X, Y, f


def object_outline(m, key):
    """Outlines [nm] of the object in the cut plane key (xz: x horizontal, z vertical; xy): list of (xs, ys, style) for the field figures; for a layer stack the
    interfaces of the xz cut as horizontal lines."""
    out = []
    if m["task"] == "bands" or m["mesh_kind"] == "file":
        return out
    th = np.linspace(0, 2 * np.pi, 200)
    if m["mesh_kind"] == "builtin":
        r = float(m["radius_nm"])
        if m["shape"] == "sphere":
            out.append((r * np.cos(th), r * np.sin(th), "w--"))
        else:
            out.append((np.array([-r, r, r, -r, -r]), np.array([-r, -r, r, r, -r]), "w--"))
        return out
    g = gmsh_params(m)
    lay = gm.layout(g)
    ox, oy, oh = gm.object_extent(g)
    z0, zc = lay["obj_z0"], lay["zc"]
    if key == "xz":
        if g["kind"] == "sphere":
            out.append((g["r"] * np.cos(th), zc + g["r"] * np.sin(th), "w--"))
        else:
            out.append((np.array([-ox, ox, ox, -ox, -ox]), np.array([z0, z0, z0 + oh, z0 + oh, z0]), "w--"))
        x1 = g["hw"]
        for z in lay["interfaces"]:
            out.append((np.array([-x1, x1]), np.array([z, z]), "c:"))
    else:
        if g["kind"] == "block":
            out.append((np.array([-ox, ox, ox, -ox, -ox]), np.array([-oy, -oy, oy, oy, -oy]), "w--"))
        else:
            out.append((g["r"] * np.cos(th), g["r"] * np.sin(th), "w--"))
    return out


def fig_field(npz, key, quarter, unit=1e-9, unit_label="nm", radius=None, shape="sphere", title="", outline=None):
    """|E|^2 / |E_0|^2 on one cut plane; xz: x horizontal, z vertical (the wave travels upwards); xy: x, y."""
    import matplotlib.pyplot as plt

    f = np.asarray(npz[key], dtype=float)
    xs = np.asarray(npz["xs"])
    other = np.asarray(npz["zs"] if key == "xz" else npz["ys"])
    if quarter:
        X, Y, F = mirror(xs, other, f, True, key == "xy")
    else:
        X, Y, F = xs, other, f
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.pcolormesh(X / unit, Y / unit, F.T, shading="auto", cmap="inferno")
    fig.colorbar(im, ax=ax, label="|E|² / |E₀|²")
    if outline is not None:
        for xs_, ys_, style in outline:
            ax.plot(np.asarray(xs_), np.asarray(ys_), style, lw=0.9)
        ax.set_xlim(X.min() / unit, X.max() / unit)
        ax.set_ylim(Y.min() / unit, Y.max() / unit)
    elif radius is not None:
        if shape == "sphere":
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(radius / unit * np.cos(t), radius / unit * np.sin(t), "w--", lw=0.9)
        else:
            r = radius / unit
            ax.plot([-r, r, r, -r, -r], [-r, -r, r, r, -r], "w--", lw=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({unit_label})")
    ax.set_ylabel(f"{'z' if key == 'xz' else 'y'} ({unit_label})")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scatter_dataframe(points):
    rows = []
    for p in points:
        r = {"λ (nm)": p["lam_nm"], "ka": p.get("size_parameter", np.nan), "Re ε": p["eps"][0], "Im ε": p["eps"][1]}
        if "error" in p:
            r["Fehler"] = p["error"]
        else:
            r.update({"Q_sca": p["q_sca"], "Q_abs": p["q_abs"], "Q_ext": p["q_ext"], "σ_sca (nm²)": p["sigma_sca"] * 1e18,
                      "σ_abs (nm²)": p["sigma_abs"] * 1e18, "σ_ext (nm²)": p["sigma_ext"] * 1e18})
            if "mie_q_sca" in p:
                r.update({"Q_sca Mie": p["mie_q_sca"], "Q_abs Mie": p["mie_q_abs"], "ΔQ_sca (%)": 100 * (p["q_sca"] - p["mie_q_sca"]) / max(abs(p["mie_q_sca"]), 1e-30),
                          "ΔQ_abs / Q_sca,Mie (%)": 100 * (p["q_abs"] - p["mie_q_abs"]) / max(abs(p["mie_q_sca"]), 1e-30)})
            if "stack_R" in p:
                r.update({"R (Stapel)": p["stack_R"], "T (Stapel)": p["stack_T"], "A (Stapel)": p["stack_A"]})
            r["Zeit (s)"] = p.get("seconds", np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def steps_dataframe(steps):
    rows = []
    for s in steps:
        r = {"Schritt": s["step"], "Zellen": s["cells"], "Unbekannte": s["dofs"], "Fehlerschätzer η": s["eta"], "p min": s["orders_min"], "p max": s["orders_max"],
             "Q_sca": s["q_sca"], "Q_abs": s["q_abs"], "h-markiert": s.get("h_marked", np.nan), "p-markiert": s.get("p_marked", np.nan), "Zeit (s)": s["seconds"]}
        if "mie_q_sca" in s:
            r["Q_sca Mie"] = s["mie_q_sca"]
            r["ΔQ_sca (%)"] = 100 * (s["q_sca"] - s["mie_q_sca"]) / max(abs(s["mie_q_sca"]), 1e-30)
        rows.append(r)
    return pd.DataFrame(rows)


def fig_convergence(steps):
    """hp-adaptive history: estimator η and Q_sca against the number of unknowns (and the deviation from the Mie series, if there is one)."""
    import matplotlib.pyplot as plt

    n = [s["dofs"] for s in steps]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axs[0].loglog(n, [s["eta"] for s in steps], "o-", color="C0")
    axs[0].set_xlabel("Unbekannte")
    axs[0].set_ylabel("Fehlerschätzer η")
    axs[0].grid(alpha=0.3, which="both")
    axs[1].semilogx(n, [s["q_sca"] for s in steps], "o-", color="C0", label="Q_sca")
    if all("mie_q_sca" in s for s in steps):
        axs[1].axhline(steps[0]["mie_q_sca"], color="k", ls="--", label="Mie-Reihe")
    axs[1].set_xlabel("Unbekannte")
    axs[1].set_ylabel("Q_sca")
    axs[1].grid(alpha=0.3, which="both")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_efficiency(points):
    import matplotlib.pyplot as plt

    ok = [p for p in points if "q_sca" in p]
    lam = [p["lam_nm"] for p in ok]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, lab, c in (("q_sca", "Streuung", "C0"), ("q_abs", "Absorption", "C3"), ("q_ext", "Extinktion", "C2")):
        ax.plot(lam, [p[key] for p in ok], "o-", color=c, label=lab)
        if all("mie_" + key in p for p in ok):
            ax.plot(lam, [p["mie_" + key] for p in ok], "x--", color=c, alpha=0.7, label=f"{lab} (Mie)")
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("Effizienz Q = σ / Querschnittsfläche")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def modes_dataframe(modes):
    rows = [{"Nr": r["index"] + 1, "λ (nm)": r["lam_nm"], "Re ω (rad/s)": r["omega"][0], "Im ω (rad/s)": r["omega"][1], "Q": r["Q"], "Residuum": r["residual"]}
            for r in modes]
    return pd.DataFrame(rows)


def bands_dataframe(points, x):
    nb = max((len(p.get("f", [])) for p in points), default=0)
    rows = []
    for i, p in enumerate(points):
        r = {"Nr": p.get("index", i), "kx (2π/a)": p["k"][0], "ky (2π/a)": p["k"][1], "kz (2π/a)": p["k"][2], "Pfad s (2π/a)": x[i]}
        for b in range(nb):
            r[f"f{b + 1}"] = p["f"][b] if b < len(p.get("f", [])) else np.nan
        r["Residuum"] = p.get("residual", np.nan)
        r["Zeit (s)"] = p.get("seconds", np.nan)
        if "error" in p:
            r["Fehler"] = p["error"]
        rows.append(r)
    return pd.DataFrame(rows)


def empty_lattice_bands(ks, n_bands, n_g=2):
    """Empty simple-cubic lattice: sorted |k + G|, each twice for the two polarisations (units 2 pi / a) for the k points ks, shape (nk, n_bands)."""
    g = np.array([(i, j, l) for i in range(-n_g, n_g + 1) for j in range(-n_g, n_g + 1) for l in range(-n_g, n_g + 1)], float)
    out = np.zeros((len(ks), n_bands))
    for i, k in enumerate(ks):
        d = np.repeat(np.sort(np.linalg.norm(np.asarray(k, float) + g, axis=1)), 2)      # two transverse polarisations per plane wave (at G: the two acoustic zeros)
        out[i] = d[:n_bands]
    return out


def copy_model(m):
    return copy.deepcopy(m)
