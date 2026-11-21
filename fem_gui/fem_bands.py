"""Band structures of two-dimensional photonic crystals (hp-FEM 0.5) without hpfem and without Streamlit: models, meshing with Gmsh, job
documents, results and figures.

The solver is ``hpfem.BandStructure2D`` (Floquet-Bloch, lossless materials, wave vector in the plane): the in-plane electric field
E = (E_x, E_y) on the Nedelec space, i.e. the polarisation with H_z out of the plane (E in the plane, "TE" in the book of Joannopoulos et al.).
The out-of-plane polarisation (E_z, "TM") is not covered by this solver. One unit cell, scale free: all lengths are in units of the lattice
constant a (a = 1), frequencies are given as f = omega a / (2 pi c) = a / lambda, wave vectors in units of 2 pi / a.

Lattices: ``square`` (a1 = (1, 0), a2 = (0, 1); path Gamma-X-M-Gamma) and ``hex`` (triangular lattice, a1 = (1, 0), a2 = (1/2, sqrt(3)/2); path
Gamma-M-K-Gamma). The cell is centred at the origin; shapes (disc, rect, ring, poly; priority by order, later ones cover earlier ones) are placed
relative to the centre and repeated on the whole lattice, so any layout is periodic (a shape that crosses the cell edge continues on the opposite edge).
Tags in the mesh: surfaces 1 + index of the material; the edge pairs left / right = 101 / 102 (translation a1), bottom / top = 103 / 104 (a2).

Only constant permittivities (type ``eps`` or ``index``) are offered: a band structure has no wavelength to evaluate dispersion at.
Time factor exp(-i omega t).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd

import fem_wg as fw

TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP = 101, 102, 103, 104
LATTICES = {"square": "quadratisch", "hex": "hexagonal (Dreiecksgitter)"}
SHAPES = {k: v for k, v in fw.SHAPES.items() if k != "strip"}
FIELDS = {"none": "keine", "corners": "an den Eckpunkten des Pfads"}
SQRT3 = math.sqrt(3.0)


# ===================================================================================================================== models
def default_model():
    """Square lattice of silicon-like rods (eps = 12, r = 0.2 a): no gap for the in-plane polarisation (the well-known TM gap needs E_z)."""
    return {
        "name": "Quadratisches Gitter: Stäbe ε = 12, r = 0,2 a", "lattice": "square",
        "materials": {"Luft": {"type": "eps", "re": 1.0, "im": 0.0}, "Dielektrikum": {"type": "eps", "re": 12.0, "im": 0.0}},
        "background": "Luft",
        "shapes": [{"type": "disc", "material": "Dielektrikum", "cx": 0.0, "cy": 0.0, "r": 0.2}],
        "n_seg": 6, "bands": 6, "mesh": {"cells_per_wavelength": 3.0, "order": 3, "interface_factor": 0.8, "f_max": 1.0},
        "field": "none", "krylov": 0,
    }


def presets():
    base = default_model()
    out = {base["name"]: base}
    empty = copy.deepcopy(base)
    empty.update(name="Leeres Gitter (Referenz: Bänder |k + G|)", shapes=[], bands=6)
    out[empty["name"]] = empty
    hexempty = copy.deepcopy(empty)
    hexempty.update(name="Leeres Dreiecksgitter (Referenz)", lattice="hex")
    out[hexempty["name"]] = hexempty
    holes = copy.deepcopy(base)
    holes.update(name="Dreiecksgitter: Luftlöcher r = 0,4 a in ε = 13", lattice="hex", background="Dielektrikum",
                 shapes=[{"type": "disc", "material": "Luft", "cx": 0.0, "cy": 0.0, "r": 0.4}], bands=6)
    holes["materials"] = {"Luft": {"type": "eps", "re": 1.0, "im": 0.0}, "Dielektrikum": {"type": "eps", "re": 13.0, "im": 0.0}}
    out[holes["name"]] = holes
    return out


def eps_of(spec):
    """Permittivity of a material of the model (constant types only), complex."""
    return complex(spec["re"], spec["im"]) if spec["type"] == "eps" else complex(spec["n"], spec["k"]) ** 2


def lattice_vectors(m):
    return (np.array([1.0, 0.0]), np.array([0.0, 1.0])) if m["lattice"] == "square" else (np.array([1.0, 0.0]), np.array([0.5, SQRT3 / 2]))


def cell_polygon(m):
    """Corners of the unit cell, centred at the origin: c0 = -(a1 + a2)/2, c0 + a1, c0 + a1 + a2, c0 + a2."""
    a1, a2 = lattice_vectors(m)
    c0 = -(a1 + a2) / 2
    return [c0, c0 + a1, c0 + a1 + a2, c0 + a2]


def translations(m, n=1):
    a1, a2 = lattice_vectors(m)
    return [(i, j, i * a1 + j * a2) for i in range(-n, n + 1) for j in range(-n, n + 1)]


def material_at(m, x, y):
    """Material at (x, y) of the periodic structure (all lattice images of the shapes; later shapes cover earlier ones)."""
    for s in reversed(m["shapes"]):
        for _, _, t in translations(m, 2):
            if fw.inside(s, x - t[0], y - t[1]):
                return s["material"]
    return m["background"]


def used_materials(m):
    return {m["background"]} | {s["material"] for s in m["shapes"]}


def validate(m):
    """List of (level, text) with level "error" or "warn"."""
    out = []
    mats = m["materials"]
    for n in used_materials(m):
        if n not in mats:
            out.append(("error", f"Material „{n}“ fehlt."))
            continue
        e = eps_of(mats[n])
        if abs(e.imag) > 1e-12:
            out.append(("error", f"{n}: verlustbehaftete Materialien (Im ε ≠ 0) rechnet der Bandlöser nicht (verlustfrei)."))
        elif e.real <= 0:
            out.append(("error", f"{n}: ε muss positiv sein (Metalle mit negativem ε sind nicht möglich)."))
    for i, s in enumerate(m["shapes"]):
        t = s["type"]
        if t not in SHAPES:
            out.append(("error", f"Form {i + 1}: unbekannter Typ „{t}“."))
        elif t == "disc" and s["r"] <= 0:
            out.append(("error", f"Form {i + 1}: der Radius muss positiv sein."))
        elif t == "ring" and (s["w"] <= 0 or s["r"] - s["w"] / 2 <= 0):
            out.append(("error", f"Form {i + 1}: Ring: Breite positiv und kleiner als der doppelte Radius der Mittellinie."))
        elif t == "rect" and (s["x1"] <= s["x0"] or s["y1"] <= s["y0"]):
            out.append(("error", f"Form {i + 1}: Rechteck mit x0 < x1 und y0 < y1 nötig."))
        elif t == "poly" and len(s["points"]) < 3:
            out.append(("error", f"Form {i + 1}: ein Polygon braucht mindestens drei Punkte."))
        if s.get("material") not in mats:
            out.append(("error", f"Form {i + 1}: Material „{s.get('material')}“ fehlt."))
    if not 1 <= int(m["bands"]) <= 20:
        out.append(("error", "Die Zahl der Bänder muss zwischen 1 und 20 liegen."))
    if not 1 <= int(m["n_seg"]) <= 40:
        out.append(("error", "Die Zahl der Punkte je Pfadsegment muss zwischen 1 und 40 liegen."))
    if m["lattice"] not in LATTICES:
        out.append(("error", f"Unbekanntes Gitter „{m['lattice']}“."))
    mesh = m["mesh"]
    if mesh["cells_per_wavelength"] < 1 or not 1 <= int(mesh["order"]) <= 8 or mesh["f_max"] <= 0:
        out.append(("error", "Netz: Elemente je Wellenlänge ≥ 1, Ordnung 1 bis 8, größte Frequenz > 0."))
    if not out:
        n_max = max(math.sqrt(eps_of(mats[n]).real) for n in used_materials(m))
        h = 1.0 / (mesh["f_max"] * mesh["cells_per_wavelength"] * n_max)
        if h < 0.02:
            out.append(("warn", f"Sehr feines Netz (Elementgröße {h:.3f} a): das braucht viel Zeit und Speicher."))
        if m["lattice"] == "square" and any(s["type"] == "disc" and s["r"] > 0.5 for s in m["shapes"]):
            out.append(("warn", "Ein Radius über 0,5 a lässt sich im quadratischen Gitter mit seinen Nachbarn überlappen (spätere Formen decken frühere)."))
        if int(m["bands"]) >= 12:
            out.append(("warn", "Viele Bänder: höhere Bänder brauchen ein feineres Netz und eine größere Ordnung."))
    return out


# ===================================================================================================================== path in the Brillouin zone
def path_corners(m):
    """[(label, k)] corners of the irreducible path, k in units of 2 pi / a."""
    if m["lattice"] == "square":
        return [("Γ", (0.0, 0.0)), ("X", (0.5, 0.0)), ("M", (0.5, 0.5)), ("Γ", (0.0, 0.0))]
    return [("Γ", (0.0, 0.0)), ("M", (0.0, 1.0 / SQRT3)), ("K", (1.0 / 3.0, 1.0 / SQRT3)), ("Γ", (0.0, 0.0))]


def k_path(m):
    """(k (n, 2) in units of 2 pi / a, x (n,) the path length in the same units, ticks [(x, label)], corner indices)."""
    corners = path_corners(m)
    n = int(m["n_seg"])
    ks, xs, ticks, corner_idx = [], [], [], []
    x = 0.0
    for c in range(len(corners) - 1):
        k0, k1 = np.array(corners[c][1]), np.array(corners[c + 1][1])
        if c == 0:
            ticks.append((0.0, corners[0][0]))
            corner_idx.append(0)
        for s in range(1, n + 1):
            k = k0 + (k1 - k0) * s / n
            x_ = x + np.linalg.norm(k - k0)
            ks.append(k)
            xs.append(x_)
        x += float(np.linalg.norm(k1 - k0))
        ticks.append((x, corners[c + 1][0]))
        corner_idx.append(len(ks))
    ks = np.vstack([np.array(corners[0][1])] + ks)
    xs = np.array([0.0] + xs)
    return ks, xs, ticks, corner_idx


# ===================================================================================================================== meshing
def _edge_lines(m):
    c = cell_polygon(m)
    a1, a2 = lattice_vectors(m)
    return {TAG_LEFT: (c[0], a2), TAG_RIGHT: (c[1], a2), TAG_BOTTOM: (c[0], a1), TAG_TOP: (c[3], a1)}


def _dist(p, line):
    p0, d = line
    d = d / np.linalg.norm(d)
    r = np.asarray(p[:2]) - p0
    return abs(r[0] * d[1] - r[1] * d[0])


def build_mesh(m, out_dir):
    """Gmsh mesh (MSH 4.1, units of a) of the periodic unit cell with matching nodes on opposite edges: out_dir/mesh.msh; returns statistics."""
    import gmsh

    errs = [t for lv, t in validate(m) if lv == "error"]
    if errs:
        raise ValueError("Modell fehlerhaft: " + " | ".join(errs))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(m["materials"])
    mesh_s = m["mesh"]
    iface = float(mesh_s.get("interface_factor", 0.8))
    sizes = {n: 1.0 / (mesh_s["f_max"] * mesh_s["cells_per_wavelength"] * max(math.sqrt(eps_of(m["materials"][n]).real), 1.0)) for n in names}
    a1, a2 = lattice_vectors(m)
    corners = cell_polygon(m)
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-7)
        gmsh.model.add("cell")
        occ = gmsh.model.occ
        pts = [occ.addPoint(float(p[0]), float(p[1]), 0) for p in corners]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        cell = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        tools = []
        for s in m["shapes"]:
            sx0, sx1, sy0, sy1 = fw.shape_bounds(s)
            for _, _, t in translations(m, 2):
                if sx1 + t[0] < min(xs) - 1e-9 or sx0 + t[0] > max(xs) + 1e-9 or sy1 + t[1] < min(ys) - 1e-9 or sy0 + t[1] > max(ys) + 1e-9:
                    continue
                shifted = dict(s)
                if s["type"] == "poly":
                    shifted["points"] = [[p[0] + t[0], p[1] + t[1]] for p in s["points"]]
                elif s["type"] == "rect":
                    shifted.update(x0=s["x0"] + t[0], x1=s["x1"] + t[0], y0=s["y0"] + t[1], y1=s["y1"] + t[1])
                else:
                    shifted.update(cx=s["cx"] + t[0], cy=s["cy"] + t[1])
                for dt in fw._surface(occ, shifted):
                    inter, _ = occ.intersect([dt], [(2, cell)], removeObject=True, removeTool=False)
                    tools += [d for d in inter if d[0] == 2]
        occ.fragment([(2, cell)], tools)
        occ.synchronize()
        e = 1e-7
        surf_mat, groups = {}, {}
        for _, tag in gmsh.model.getEntities(2):
            mat = _surface_material(m, tag)
            surf_mat[tag] = mat
            groups.setdefault(mat, []).append(tag)
        for mat, tags in groups.items():
            gmsh.model.addPhysicalGroup(2, tags, 1 + names.index(mat), mat)
        edges = _edge_lines(m)
        found = {t: [] for t in edges}
        for dim, tag in gmsh.model.getEntities(1):
            u0, u1 = gmsh.model.getParametrizationBounds(1, tag)
            u0, u1 = float(np.ravel(u0)[0]), float(np.ravel(u1)[0])
            probe = [gmsh.model.getValue(1, tag, [u]) for u in (u0, 0.5 * (u0 + u1), u1)]
            for t, line in edges.items():
                if all(_dist(p, line) < e for p in probe):
                    found[t].append((tag, np.asarray(probe[1][:2])))
        names_e = {TAG_LEFT: "left", TAG_RIGHT: "right", TAG_BOTTOM: "bottom", TAG_TOP: "top"}
        for t, items in found.items():
            if not items:
                raise ValueError("Zellrand ohne Kurven: Netzerzeugung fehlgeschlagen.")
            gmsh.model.addPhysicalGroup(1, [i[0] for i in items], t, names_e[t])

        def ordered(tag, direction):
            d = direction / np.linalg.norm(direction)
            return [i[0] for i in sorted(found[tag], key=lambda it: float(it[1] @ d))]

        for master, slave, shift in ((TAG_LEFT, TAG_RIGHT, a1), (TAG_BOTTOM, TAG_TOP, a2)):
            if len(found[master]) != len(found[slave]):
                raise ValueError("Die Formen zerteilen gegenüberliegende Zellränder verschieden: die Struktur ist nicht periodisch darstellbar.")
            along = a2 if master == TAG_LEFT else a1
            gmsh.model.mesh.setPeriodic(1, ordered(slave, along), ordered(master, along),
                                        [1, 0, 0, float(shift[0]), 0, 1, 0, float(shift[1]), 0, 0, 1, 0, 0, 0, 0, 1])
        pt_size = {}
        for tag, mat in surf_mat.items():
            h = sizes[mat]
            for _, p in gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False, recursive=True):
                pt_size[p] = min(pt_size.get(p, 1e9), h)
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
        n_cells = int(sum(len(t) for t in gmsh.model.mesh.getElementsByType(2)[:1]))
        return {"cells": n_cells, "h_min": min(sizes.values()) * min(iface, 1.0), "h_max": max(sizes.values())}
    finally:
        gmsh.finalize()


def _surface_material(m, tag):
    """Material of a surface entity (majority vote over interior sample points of the surface)."""
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


# ===================================================================================================================== jobs and results
def job(m):
    """The job document for the worker: the model plus the k path (units of 2 pi / a) and the corner indices."""
    d = copy.deepcopy(m)
    ks, xs, ticks, corner_idx = k_path(m)
    d["k"] = ks.tolist()
    d["x"] = xs.tolist()
    d["ticks"] = [[float(x), lab] for x, lab in ticks]
    d["corner_idx"] = [int(i) for i in corner_idx]
    return d


def clean_bands(values, n_bands, at_gamma):
    """The lowest ``n_bands`` normalised frequencies; at Gamma the solver finds a second zero (the constant field E = const is curl free but no
    gradient of a periodic function): it is dropped, so that the band index is the same along the whole path."""
    v = sorted(float(x) for x in values)
    if at_gamma and len(v) > 1 and v[1] < 1e-3:
        del v[1]
    return v[:n_bands]


def bands_matrix(points, n_bands=None):
    """(n_k, n_bands) normalised frequencies of the points that have a result (NaN for failed points)."""
    ok = [p for p in points if "f" in p]
    nb = n_bands or (max(len(p["f"]) for p in ok) if ok else 0)
    out = np.full((len(points), nb), np.nan)
    for i, p in enumerate(points):
        if "f" in p:
            v = p["f"][:nb]
            out[i, :len(v)] = v
    return out


def band_gaps(f, min_width=1e-4):
    """Gaps between neighbouring bands over the whole path: [(band, lower edge, upper edge, width, mid, width / mid in %)] (band = the lower of the
    two, counted from 1; the zero-frequency band at Gamma counts as band 1)."""
    out = []
    for i in range(f.shape[1] - 1):
        lo, hi = np.nanmax(f[:, i]), np.nanmin(f[:, i + 1])
        if np.isfinite(lo) and np.isfinite(hi) and hi - lo > min_width:
            mid = 0.5 * (hi + lo)
            out.append((i + 1, float(lo), float(hi), float(hi - lo), float(mid), float(100 * (hi - lo) / mid)))
    return out


def gaps_dataframe(points):
    f = bands_matrix(points)
    rows = [{"zwischen Band": f"{b} und {b + 1}", "untere Kante f": lo, "obere Kante f": hi, "Breite Δf": w, "Mitte f": mid, "Δf/Mitte (%)": pct}
            for b, lo, hi, w, mid, pct in band_gaps(f)]
    return pd.DataFrame(rows, columns=["zwischen Band", "untere Kante f", "obere Kante f", "Breite Δf", "Mitte f", "Δf/Mitte (%)"])


def bands_dataframe(points, x=None, ticks=None):
    f = bands_matrix(points)
    rows = []
    for i, p in enumerate(points):
        r = {"Nr": p.get("index", i), "kx (2π/a)": p.get("k", [np.nan, np.nan])[0], "ky (2π/a)": p.get("k", [np.nan, np.nan])[1]}
        if x is not None:
            r["Pfad s (2π/a)"] = x[i]
        for b in range(f.shape[1]):
            r[f"f{b + 1}"] = f[i, b]
        if "error" in p:
            r["Fehler"] = p["error"]
        else:
            r["Residuum"] = p.get("residual", np.nan)
            r["Zeit (s)"] = p.get("time_s", np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def fig_bands(points, x, ticks, gaps=True, empty_lattice=None):
    """Band diagram f = omega a / 2 pi c along the path; gaps over the whole path are shaded."""
    import matplotlib.pyplot as plt

    f = bands_matrix(points)
    x = np.asarray(x, float)[:len(points)]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for b in range(f.shape[1]):
        ax.plot(x, f[:, b], "o-", ms=3, lw=1.2, color="C0")
    if empty_lattice is not None:
        for b in range(empty_lattice.shape[1]):
            ax.plot(x, empty_lattice[:, b], "-", lw=0.8, color="0.6", zorder=0)
    if gaps:
        for b, lo, hi, w, mid, pct in band_gaps(f):
            ax.axhspan(lo, hi, color="C1", alpha=0.25)
            ax.text(x[-1], mid, f" {pct:.1f} %", va="center", fontsize=8, color="C1")
    for xt, _ in ticks:
        ax.axvline(xt, color="0.7", lw=0.6)
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, None)
    ax.set_ylabel("f = ω a / (2π c)")
    ax.set_xlabel("Wellenvektor entlang des Pfads")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return fig


def empty_lattice_bands(m, n_bands, n_g=4):
    """Bands of the empty lattice |k + G| a / 2 pi along the path (reference lines), shape (n_k, n_bands)."""
    a1, a2 = lattice_vectors(m)
    area = abs(a1[0] * a2[1] - a1[1] * a2[0])
    b1 = np.array([a2[1], -a2[0]]) / area                      # reciprocal vectors in units of 2 pi / a: b_i . a_j = delta_ij
    b2 = np.array([-a1[1], a1[0]]) / area
    g = np.array([i * b1 + j * b2 for i in range(-n_g, n_g + 1) for j in range(-n_g, n_g + 1)])
    ks = k_path(m)[0]
    return np.array([np.sort(np.linalg.norm(g + k, axis=1))[:n_bands] for k in ks])


def _outline_cell(ax, m, color="k", lw=1.4):
    from matplotlib.patches import Polygon

    ax.add_patch(Polygon(np.array(cell_polygon(m)), closed=True, fill=False, edgecolor=color, lw=lw))


def fig_preview(m, tiles=1):
    """The unit cell (bold outline) with the shapes and their lattice images in the neighbouring cells (lighter)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    cmap = plt.get_cmap("tab10")
    color = {n: cmap(i % 10) for i, n in enumerate(m["materials"])}
    a1, a2 = lattice_vectors(m)
    for i in range(-tiles, tiles + 1):
        for j in range(-tiles, tiles + 1):
            t = i * a1 + j * a2
            central = i == 0 and j == 0
            poly = np.array(cell_polygon(m)) + t
            ax.add_patch(Polygon(poly, closed=True, facecolor=color.get(m["background"], "0.8"), alpha=0.5 if central else 0.2,
                                 edgecolor="k" if central else "0.5", lw=1.4 if central else 0.5))
            for s in m["shapes"]:
                kw = dict(facecolor=color.get(s["material"], "0.5"), alpha=0.85 if central else 0.35, edgecolor="k", lw=0.5)
                tt = s["type"]
                if tt == "disc":
                    ax.add_patch(Circle((s["cx"] + t[0], s["cy"] + t[1]), s["r"], **kw))
                elif tt == "ring":
                    ax.add_patch(Circle((s["cx"] + t[0], s["cy"] + t[1]), s["r"] + s["w"] / 2, **kw))
                    ax.add_patch(Circle((s["cx"] + t[0], s["cy"] + t[1]), s["r"] - s["w"] / 2, facecolor=color.get(m["background"], "w"),
                                        alpha=1.0 if central else 0.6, edgecolor="k", lw=0.5))
                elif tt == "rect":
                    ax.add_patch(Rectangle((s["x0"] + t[0], s["y0"] + t[1]), s["x1"] - s["x0"], s["y1"] - s["y0"], **kw))
                else:
                    ax.add_patch(Polygon(np.asarray(s["points"], float) + t, closed=True, **kw))
    ext = tiles + 0.6
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext * 0.9, ext * 0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (a)")
    ax.set_ylabel("y (a)")
    ax.legend(handles=[Patch(facecolor=color[n], alpha=0.8, label=n) for n in m["materials"] if n in used_materials(m)], fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig_mode(npz, band, quantity="abs2", title="", model=None):
    """|E|^2 (or one component) of one band at one k point on the subdivided mesh, repeated over the 2 x 2 neighbouring cells optionally."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    pts, simp = npz["points"], npz["simplices"]
    e = npz[f"et_{band}"]
    if quantity == "abs2":
        val, label, cmap = np.abs(e[:, 0]) ** 2 + np.abs(e[:, 1]) ** 2, "|E|²", "inferno"
    elif quantity == "ex":
        val, label, cmap = e[:, 0].real, "Re E_x", "RdBu_r"
    else:
        val, label, cmap = e[:, 1].real, "Re E_y", "RdBu_r"
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], simp)
    kw = dict(cmap=cmap, shading="gouraud")
    if quantity != "abs2":
        v = float(np.abs(val).max()) or 1.0
        kw.update(vmin=-v, vmax=v)
    im = ax.tripcolor(tri, val, **kw)
    fig.colorbar(im, ax=ax, label=label)
    if model is not None:
        from matplotlib.patches import Polygon

        ax.add_patch(Polygon(np.array(cell_polygon(model)), closed=True, fill=False, edgecolor="w", lw=0.8))
    ax.set_aspect("equal")
    ax.set_xlabel("x (a)")
    ax.set_ylabel("y (a)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig
