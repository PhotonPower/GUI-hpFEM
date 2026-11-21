"""Waveguides and ports (hp-FEM 0.5) without hpfem and without Streamlit: models, meshing with Gmsh, job documents, results and figures.

Two tasks share one model (``default_model()``), a two-dimensional device in the plane (x, y):

* ``ports``: the chip plane in the effective-index model (field polarisation H_z, in-plane E = (E_x, E_y), ``hpfem.Scattering2D``) between two
  modal ports at x = 0 (left) and x = L (right). The port modes are the guided modes of the transverse section (1D p-FEM on the port line,
  ``hpfem.WaveguidePort``); the power-normalised S-matrix of all propagating channels comes from ``hpfem.s_parameters`` (one factorisation).
  The transverse boundaries y = +-W/2 are PEC walls: keep them several decay lengths away from the guides.
* ``modes``: the cross-section of a waveguide in (x, y) in a PEC box (``hpfem.PropagatingMode``, Lee-Sun-Cendes pencil): effective indices and
  the vector mode fields (E_t on the Nedelec space, E_z on H1) of the guided modes of a straight waveguide.

Shapes (priority by order, later ones cover earlier ones; nm): ``strip`` (x0, x1, yc, w), ``rect`` (x0, x1, y0, y1), ``disc`` (cx, cy, r),
``ring`` (cx, cy, r, w; r = centre line), ``poly`` (points). Tags in the mesh: 1 + index of the material for the surfaces; the lines
left / right / bottom / top carry 101 / 102 / 103 / 104.

Units: nm in the models, SI in the solver. Time factor exp(-i omega t); loss means Im eps > 0.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd

import fem_materials as fm

TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP = 101, 102, 103, 104
TASKS = {"ports": "Ports: S-Matrix des Bauteils in der Chipebene", "modes": "Moden: Querschnitt eines Wellenleiters"}
SHAPES = {"strip": "Streifen (waagerecht)", "rect": "Rechteck", "disc": "Scheibe", "ring": "Ring", "poly": "Polygon"}
FIELDS = {"none": "keine", "first": "erste Wellenlänge", "all": "alle Wellenlängen"}


# ===================================================================================================================== models
def default_model():
    """Straight strip waveguide (effective-index model of a 220 nm SOI slab: n = 2.4 in silica 1.44), 450 nm wide."""
    return {
        "name": "Streifenwellenleiter", "task": "ports",
        "materials": {"Silica": {"type": "index", "n": 1.44, "k": 0.0}, "Kern": {"type": "index", "n": 2.4, "k": 0.0}},
        "background": "Silica", "length_nm": 3000.0, "width_nm": 3000.0,
        "shapes": [{"type": "strip", "material": "Kern", "x0": -100.0, "x1": 3100.0, "yc": 0.0, "w": 450.0}],
        "ports": {"left": {"enabled": True, "modes": 2}, "right": {"enabled": True, "modes": 2}},
        "sweep": {"mode": "single", "start": 1550.0, "stop": 1600.0, "n": 5},
        "mesh": {"cells_per_wavelength": 5.0, "order": 3, "interface_factor": 0.8},
        "modes": {"num": 4}, "field": "first", "excite": {"port": 0, "mode": 0},
    }


def _strip(x0, x1, yc, w, material="Kern"):
    return {"type": "strip", "material": material, "x0": x0, "x1": x1, "yc": yc, "w": w}


def presets():
    base = default_model()
    out = {"Streifenwellenleiter": base}
    step = copy.deepcopy(base)
    step.update(name="Stufe: Streifen 450 → 900 nm")
    step["shapes"] = [_strip(-100.0, 1500.0, 0.0, 450.0), _strip(1500.0, 3100.0, 0.0, 900.0)]
    step["ports"] = {"left": {"enabled": True, "modes": 3}, "right": {"enabled": True, "modes": 4}}
    out[step["name"]] = step
    rib = copy.deepcopy(base)
    rib.update(name="Querschnitt: Streifen 450 × 220 nm", task="modes", length_nm=3000.0, width_nm=2000.0)
    rib["shapes"] = [{"type": "rect", "material": "Kern", "x0": 1500.0 - 225.0, "x1": 1500.0 + 225.0, "y0": -110.0, "y1": 110.0}]
    rib["materials"] = {"Silica": {"type": "index", "n": 1.44, "k": 0.0}, "Kern": {"type": "index", "n": 3.48, "k": 0.0}}
    rib["mesh"] = {"cells_per_wavelength": 8.0, "order": 3, "interface_factor": 0.6}
    out[rib["name"]] = rib
    return out


def sweep_list(sw):
    if sw["mode"] == "single":
        return [float(sw["start"])]
    return [float(round(v, 6)) for v in np.linspace(float(sw["start"]), float(sw["stop"]), max(int(sw["n"]), 1))]


# ===================================================================================================================== geometry helpers
def shape_bounds(s):
    t = s["type"]
    if t == "strip":
        return float(s["x0"]), float(s["x1"]), float(s["yc"]) - float(s["w"]) / 2, float(s["yc"]) + float(s["w"]) / 2
    if t == "rect":
        return float(s["x0"]), float(s["x1"]), float(s["y0"]), float(s["y1"])
    if t == "disc":
        return s["cx"] - s["r"], s["cx"] + s["r"], s["cy"] - s["r"], s["cy"] + s["r"]
    if t == "ring":
        ro = s["r"] + s["w"] / 2
        return s["cx"] - ro, s["cx"] + ro, s["cy"] - ro, s["cy"] + ro
    pts = np.asarray(s["points"], float)
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def inside(s, x, y):
    t = s["type"]
    if t in ("strip", "rect"):
        x0, x1, y0, y1 = shape_bounds(s)
        return x0 <= x <= x1 and y0 <= y <= y1
    if t == "disc":
        return math.hypot(x - s["cx"], y - s["cy"]) <= s["r"]
    if t == "ring":
        return abs(math.hypot(x - s["cx"], y - s["cy"]) - s["r"]) <= s["w"] / 2
    pts = s["points"]
    ins, j = False, len(pts) - 1
    for i in range(len(pts)):
        xi, yi, xj, yj = pts[i][0], pts[i][1], pts[j][0], pts[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi:
            ins = not ins
        j = i
    return ins


def material_at(model, x, y):
    for s in reversed(model["shapes"]):
        if inside(s, x, y):
            return s["material"]
    return model["background"]


def used_materials(m):
    return {m["background"]} | {s["material"] for s in m["shapes"]}


def validate(m, lam_nm=None):
    """List of (level, text) with level "error" or "warn"."""
    out = []
    L, W = float(m["length_nm"]), float(m["width_nm"])
    if L <= 0 or W <= 0:
        out.append(("error", "Länge und Breite des Gebiets müssen positiv sein."))
        return out
    mats = m["materials"]
    for n in used_materials(m):
        if n not in mats:
            out.append(("error", f"Material „{n}“ fehlt."))
        elif fm.is_crystal(mats[n]):
            out.append(("error", f"{n}: Kristalle sind in diesem Modus nicht möglich (isotrope Materialien)."))
    if any(lv == "error" for lv, _ in out):
        return out
    lam = lam_nm if lam_nm else sweep_list(m["sweep"])[0]
    for n in used_materials(m):
        try:
            e = fm.eps_at(mats[n], lam)
        except Exception as exc:
            out.append(("error", f"{n}: {exc}"))
            continue
        if abs(complex(e).imag) > 1e-9 and m["task"] == "modes":
            out.append(("error", f"{n}: verlustbehaftet; die Modenanalyse braucht verlustfreie Materialien."))
        elif abs(complex(e).imag) > 1e-9:
            out.append(("warn", f"{n}: verlustbehaftet (Im ε > 0). Die Portmoden sind für verlustfreie Schnitte gedacht; Verlust nahe der Ports vermeiden."))
    for i, s in enumerate(m["shapes"]):
        if s["material"] not in mats:
            out.append(("error", f"Form {i + 1}: Material „{s['material']}“ fehlt."))
            continue
        try:
            x0, x1, y0, y1 = shape_bounds(s)
        except Exception:
            out.append(("error", f"Form {i + 1}: ungültige Angaben."))
            continue
        if s["type"] in ("strip", "rect") and (x1 <= x0 or y1 <= y0):
            out.append(("error", f"Form {i + 1}: Ausdehnung ≤ 0."))
        if s["type"] in ("disc", "ring") and (s["r"] <= 0 or (s["type"] == "ring" and not 0 < s["w"] < 2 * s["r"])):
            out.append(("error", f"Form {i + 1}: Radius und Breite müssen 0 < w < 2r erfüllen."))
        if s["type"] == "poly" and len(s["points"]) < 3:
            out.append(("error", f"Form {i + 1}: ein Polygon braucht mindestens 3 Punkte."))
        if x1 < 0 or x0 > L or y1 < -W / 2 or y0 > W / 2:
            out.append(("warn", f"Form {i + 1} liegt außerhalb des Gebiets (0 … {L:g}) × (±{W / 2:g}) nm."))
        elif y0 < -W / 2 + 1 or y1 > W / 2 - 1:
            out.append(("warn", f"Form {i + 1} berührt die Wand (PEC) bei y = ±{W / 2:g} nm."))
    if m["task"] == "ports":
        ports = m["ports"]
        if not (ports["left"]["enabled"] or ports["right"]["enabled"]):
            out.append(("error", "Mindestens ein Port muss eingeschaltet sein."))
        for side, tag in (("left", "links"), ("right", "rechts")):
            if ports[side]["enabled"] and int(ports[side]["modes"]) < 1:
                out.append(("error", f"Port {tag}: mindestens 1 Mode."))
        wall = [s for s in m["shapes"] if shape_bounds(s)[2] < -W / 2 + 300 or shape_bounds(s)[3] > W / 2 - 300]
        if wall:
            out.append(("warn", "Ein Wellenleiter kommt der PEC-Wand näher als 300 nm: die Moden werden verfälscht. Gebiet breiter wählen."))
        e_bg = fm.eps_at(mats[m["background"]], lam) if m["background"] in mats else 1.0
        if complex(e_bg).real <= 0:
            out.append(("error", "Das Hintergrundmaterial muss einen positiven Realteil von ε haben."))
        ref = [s for s in m["shapes"] if s["type"] in ("strip", "rect")]
        if ports["left"]["enabled"] and not any(shape_bounds(s)[0] <= 0.0 and shape_bounds(s)[2] <= 0 <= shape_bounds(s)[3] for s in ref) and not ref:
            out.append(("warn", "Am linken Port liegt kein Wellenleiter; es gibt dann keine geführten Moden."))
    else:
        if int(m["modes"]["num"]) < 1:
            out.append(("error", "Mindestens 1 Mode."))
        if not m["shapes"]:
            out.append(("error", "Der Querschnitt braucht mindestens einen Kern."))
    mesh = m["mesh"]
    if int(mesh["order"]) < 1 or float(mesh["cells_per_wavelength"]) <= 0:
        out.append(("error", "Ordnung ≥ 1 und Zellen je Wellenlänge > 0."))
    return out


# ===================================================================================================================== mesh
def _surface(occ, s):
    """OCC surface (list of dimtags) of a shape."""
    t = s["type"]
    x0, x1, y0, y1 = shape_bounds(s)
    if t in ("strip", "rect"):
        return [(2, occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0))]
    if t == "disc":
        return [(2, occ.addDisk(s["cx"], s["cy"], 0, s["r"], s["r"]))]
    if t == "ring":
        outer = occ.addDisk(s["cx"], s["cy"], 0, s["r"] + s["w"] / 2, s["r"] + s["w"] / 2)
        inner = occ.addDisk(s["cx"], s["cy"], 0, s["r"] - s["w"] / 2, s["r"] - s["w"] / 2)
        cut, _ = occ.cut([(2, outer)], [(2, inner)])
        return [d for d in cut if d[0] == 2]
    pts = [occ.addPoint(float(p[0]), float(p[1]), 0) for p in s["points"]]
    lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    return [(2, occ.addPlaneSurface([occ.addCurveLoop(lines)]))]


def _surface_material(m, tag):
    """Material of a surface entity: majority vote of material_at over parametric sample points inside the surface (the centre of mass of a
    surface with a hole may lie in another material)."""
    import gmsh

    (u0, v0), (u1, v1) = gmsh.model.getParametrizationBounds(2, tag)
    votes = {}
    for u in np.linspace(u0, u1, 9)[1:-1]:
        for v in np.linspace(v0, v1, 9)[1:-1]:
            if gmsh.model.isInside(2, tag, [u, v], True):
                x, y, _ = gmsh.model.getValue(2, tag, [u, v])
                mat = material_at(m, x, y)
                votes[mat] = votes.get(mat, 0) + 1
    if not votes:
        x, y, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        return material_at(m, x, y)
    return max(votes, key=votes.get)


def _size(spec, lam_min, cells):
    try:
        n = fm.nk(spec, lam_min)[0]
    except Exception:
        n = 1.0
    return lam_min / (cells * max(float(n), 1.0))


def build_mesh(m, lam_min_nm, out_dir):
    """Gmsh mesh (MSH 4.1) of the device: out_dir/mesh.msh, mesh_plot.npz (not needed); returns statistics."""
    import gmsh

    errs = [t for lv, t in validate(m) if lv == "error"]
    if errs:
        raise ValueError("Modell fehlerhaft: " + " | ".join(errs))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L, W = float(m["length_nm"]), float(m["width_nm"])
    names = list(m["materials"])
    cells = float(m["mesh"]["cells_per_wavelength"])
    iface = float(m["mesh"].get("interface_factor", 0.8))
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-4)
        gmsh.model.add("wg")
        occ = gmsh.model.occ
        box = occ.addRectangle(0, -W / 2, 0, L, W)
        tools = []
        for s in m["shapes"]:
            for dt in _surface(occ, s):
                inter, _ = occ.intersect([dt], [(2, box)], removeObject=True, removeTool=False)
                tools += [d for d in inter if d[0] == 2]
        occ.fragment([(2, box)], tools)
        occ.synchronize()
        e = 1e-6 * max(L, W)
        sizes = {n: _size(m["materials"][n], lam_min_nm, cells) for n in names}
        surf_mat, groups = {}, {}
        for _, tag in gmsh.model.getEntities(2):
            mat = _surface_material(m, tag)
            surf_mat[tag] = mat
            groups.setdefault(mat, []).append(tag)
        for mat, tags in groups.items():
            gmsh.model.addPhysicalGroup(2, tags, 1 + names.index(mat), mat)
        lines = {TAG_LEFT: [], TAG_RIGHT: [], TAG_BOTTOM: [], TAG_TOP: []}
        for dim, tag in gmsh.model.getEntities(1):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(dim, tag)
            if abs(x0) < e and abs(x1) < e:
                lines[TAG_LEFT].append(tag)
            elif abs(x0 - L) < e and abs(x1 - L) < e:
                lines[TAG_RIGHT].append(tag)
            elif abs(y0 + W / 2) < e and abs(y1 + W / 2) < e:
                lines[TAG_BOTTOM].append(tag)
            elif abs(y0 - W / 2) < e and abs(y1 - W / 2) < e:
                lines[TAG_TOP].append(tag)
        for t, ls in lines.items():
            gmsh.model.addPhysicalGroup(1, ls, t, {TAG_LEFT: "left", TAG_RIGHT: "right", TAG_BOTTOM: "bottom", TAG_TOP: "top"}[t])
        pt_size = {}
        for tag, mat in surf_mat.items():
            h = sizes[mat]
            for _, p in gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False, recursive=True):
                pt_size[p] = min(pt_size.get(p, 1e9), h)
        # material interfaces: a finer size on the lines between two materials
        for dim, tag in gmsh.model.getEntities(1):
            up, _ = gmsh.model.getAdjacencies(dim, tag)
            mats = {surf_mat[u] for u in up if u in surf_mat}
            if len(mats) > 1:
                h = min(sizes[x] for x in mats) * iface
                for _, p in gmsh.model.getBoundary([(1, tag)], combined=False, oriented=False):
                    pt_size[p] = min(pt_size.get(p, 1e9), h)
        for p, h in pt_size.items():
            gmsh.model.mesh.setSize([(0, p)], h)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(sizes.values()))
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out_dir / "mesh.msh"))
        etypes, _, _ = gmsh.model.mesh.getElements(2)
        n_cells = int(sum(len(t) for t in gmsh.model.mesh.getElementsByType(2)[:1])) if 2 in etypes else 0
        return {"cells": n_cells, "h_min_nm": min(sizes.values()) * min(iface, 1.0), "h_max_nm": max(sizes.values())}
    finally:
        gmsh.finalize()


# ===================================================================================================================== jobs and results
def job(m):
    """The job document for the worker (a copy of the model plus the wavelength list)."""
    d = copy.deepcopy(m)
    d["wavelengths_nm"] = sweep_list(m["sweep"])
    return d


def ports_dataframe(points):
    """One row per wavelength: transmission, reflection and loss of the excitation (port 0, mode 0)."""
    rows = []
    for p in points:
        if "error" in p:
            continue
        rows.append({"λ (nm)": p["lam_nm"], **{k: p[k] for k in ("T", "R", "loss") if k in p}})
    return pd.DataFrame(rows)


def channel_label(ch):
    return f"{'links' if ch['side'] == 'left' else 'rechts'} · Mode {ch['mode']}"


def s_matrix_frame(point, kind="abs2"):
    ch = point["channels"]
    s = np.array(point["S_re"]) + 1j * np.array(point["S_im"])
    labels = [channel_label(c) for c in ch]
    data = np.abs(s) ** 2 if kind == "abs2" else np.degrees(np.angle(s))
    return pd.DataFrame(data, index=[f"aus: {x}" for x in labels], columns=[f"Anregung: {x}" for x in labels])


def fig_spectrum(points, channels_from=None):
    """Transmission, reflection and loss of the excitation channel over the wavelength (matplotlib)."""
    import matplotlib.pyplot as plt

    pts = [p for p in points if "error" not in p]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    lam = [p["lam_nm"] for p in pts]
    ax.plot(lam, [p["T"] for p in pts], "o-", label="Transmission (Summe aller Moden des anderen Ports)", ms=3)
    ax.plot(lam, [p["R"] for p in pts], "s-", label="Reflexion (Summe aller Moden des Eingangsports)", ms=3)
    ax.plot(lam, [p["loss"] for p in pts], "^-", label="Verlust / Abstrahlung (1 − Σ|S|²)", ms=3)
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("Leistungsanteil")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_neff(points):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    pts = [p for p in points if "error" not in p]
    nmax = max((len(p["port_modes"]["left"] if "left" in p.get("port_modes", {}) else []) for p in pts), default=0)
    for side, marker in (("left", "o"), ("right", "x")):
        k_max = max((len(p.get("port_modes", {}).get(side, [])) for p in pts), default=0)
        for k in range(k_max):
            xs = [p["lam_nm"] for p in pts if len(p.get("port_modes", {}).get(side, [])) > k]
            ys = [p["port_modes"][side][k]["n_eff"] for p in pts if len(p.get("port_modes", {}).get(side, [])) > k]
            ax.plot(xs, ys, marker, ls="-" if side == "left" else "--", ms=4, label=f"{'links' if side == 'left' else 'rechts'} · Mode {k}")
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("n_eff (Portmoden)")
    ax.grid(alpha=0.3)
    if pts and nmax >= 0:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def fig_field(npz, quantity="abs", title="", model=None):
    """Field map from the triangulated field (points nm, simplices, values (n, 2) = E_x, E_y [V/m, arbitrary scale])."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    pts, tri, val = npz["points"], npz["simplices"], npz["values"]
    t = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    if quantity == "abs":
        z, label, cmap = np.sqrt(np.abs(val[:, 0]) ** 2 + np.abs(val[:, 1]) ** 2), "|E|", "inferno"
    elif quantity == "re_x":
        z, label, cmap = val[:, 0].real, "Re E_x", "RdBu_r"
    else:
        z, label, cmap = val[:, 1].real, "Re E_y", "RdBu_r"
    zmax = float(np.max(np.abs(z))) or 1.0
    if quantity == "abs":
        h = ax.tripcolor(t, z / zmax, shading="gouraud", cmap=cmap, vmin=0, vmax=1)
    else:
        h = ax.tripcolor(t, z / zmax, shading="gouraud", cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(h, ax=ax, label=f"{label} / max")
    if model:
        _outline(ax, model)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def fig_mode(npz, k, quantity="abs", title="", model=None):
    """Mode k of a cross-section: |E_t|, |E_z| or the components (arrays et_k (n, 2), ez_k (n,))."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    pts, tri = npz["points"], npz["simplices"]
    et, ez = npz[f"et_{k}"], npz[f"ez_{k}"]
    t = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
    ref = max(float(np.max(np.abs(et))), float(np.max(np.abs(ez)))) or 1.0
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if quantity == "abs_t":
        z, label, cmap, signed = np.sqrt(np.abs(et[:, 0]) ** 2 + np.abs(et[:, 1]) ** 2), "|E_t|", "inferno", False
    elif quantity == "abs_z":
        z, label, cmap, signed = np.abs(ez), "|E_z|", "inferno", False
    elif quantity == "ex":
        z, label, cmap, signed = (et[:, 0] * np.exp(-1j * np.angle(et[np.argmax(np.abs(et[:, 0])), 0]))).real, "Re E_x", "RdBu_r", True
    else:
        z, label, cmap, signed = (et[:, 1] * np.exp(-1j * np.angle(et[np.argmax(np.abs(et[:, 1])), 1]))).real, "Re E_y", "RdBu_r", True
    z = z / ref
    h = ax.tripcolor(t, z, shading="gouraud", cmap=cmap, vmin=-1 if signed else 0, vmax=1)
    fig.colorbar(h, ax=ax, label=f"{label} / max(|E|)")
    if model:
        _outline(ax, model)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def _outline(ax, model):
    """Draws the shapes as outlines (preview and overlays)."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    for s in model["shapes"]:
        kw = dict(fill=False, edgecolor="w", lw=0.8, alpha=0.9)
        t = s["type"]
        if t in ("strip", "rect"):
            x0, x1, y0, y1 = shape_bounds(s)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **kw))
        elif t == "disc":
            ax.add_patch(Circle((s["cx"], s["cy"]), s["r"], **kw))
        elif t == "ring":
            ax.add_patch(Circle((s["cx"], s["cy"]), s["r"] + s["w"] / 2, **kw))
            ax.add_patch(Circle((s["cx"], s["cy"]), s["r"] - s["w"] / 2, **kw))
        else:
            ax.add_patch(Polygon(np.asarray(s["points"], float), closed=True, **kw))


def fig_preview(m):
    """The device: materials as colours, shapes in priority order, ports as arrows."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    L, W = float(m["length_nm"]), float(m["width_nm"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cmap = plt.get_cmap("tab10")
    color = {n: cmap(i % 10) for i, n in enumerate(m["materials"])}
    ax.add_patch(Rectangle((0, -W / 2), L, W, facecolor=color.get(m["background"], "0.8"), alpha=0.25, edgecolor="k", lw=1.2))
    for s in m["shapes"]:
        c = color.get(s["material"], "0.5")
        kw = dict(facecolor=c, alpha=0.7, edgecolor="k", lw=0.6)
        t = s["type"]
        if t in ("strip", "rect"):
            x0, x1, y0, y1 = shape_bounds(s)
            x0, x1 = max(x0, 0.0), min(x1, L)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **kw))
        elif t == "disc":
            ax.add_patch(Circle((s["cx"], s["cy"]), s["r"], **kw))
        elif t == "ring":
            ax.add_patch(Circle((s["cx"], s["cy"]), s["r"] + s["w"] / 2, **kw))
            ax.add_patch(Circle((s["cx"], s["cy"]), s["r"] - s["w"] / 2, facecolor=color.get(m["background"], "w"), alpha=1.0, edgecolor="k", lw=0.6))
        else:
            ax.add_patch(Polygon(np.asarray(s["points"], float), closed=True, **kw))
    if m["task"] == "ports":
        for side, x, dx in (("left", 0.0, 1.0), ("right", L, -1.0)):
            if m["ports"][side]["enabled"]:
                ax.annotate("", xy=(x + dx * 0.06 * L, 0), xytext=(x - dx * 0.0, 0), arrowprops=dict(arrowstyle="->", color="r", lw=2))
                ax.plot([x, x], [-W / 2, W / 2], "r-", lw=2.5, alpha=0.8)
    ax.set_xlim(-0.04 * L, 1.04 * L)
    ax.set_ylim(-W / 2 * 1.04, W / 2 * 1.04)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=color[n], alpha=0.7, label=n) for n in m["materials"] if n in used_materials(m)], fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig
