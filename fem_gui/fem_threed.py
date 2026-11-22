"""3D structures in the FEM model builder (hp-FEM 0.5): models, job, checks and figures. Pure Python (no hpfem needed); worker: fem_threed_worker.py,
user interface: fem_threed_app.py.

Three tasks with the 3D solvers of the library: ``scatter`` (Scattering3D: plane wave on a sphere or cube in a PML box, cross sections, Mie series as the
reference for the sphere), ``resonance`` (Resonance3D: complex eigenfrequencies, Q) and ``bands`` (BandStructure3D: simple-cubic photonic crystal).
Meshes come from the generators of the library (box_with_ball: curved sphere; box: cube aligned with the grid), no Gmsh needed.
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import fem_materials as fm
import fem_run as fr

TASKS = {"scatter": "Streuung an Kugel oder Würfel (Wirkungsquerschnitte)", "resonance": "Resonanzen (Quasi-Normalmoden, Q-Faktor)",
         "bands": "Bandstruktur (kubisches photonisches Kristallgitter)"}
SHAPES = {"sphere": "Kugel (gekrümmte Elemente)", "cube": "Würfel (an das Gitter angepasst)"}
SYM = {False: "ganzes Modell", True: "Viertelmodell (x > 0, y > 0, Spiegelsymmetrie)"}
GB = 1e9
OBJ = "Objekt"


def default_model():
    return {"name": "Kugel in Luft", "task": "scatter", "shape": "sphere", "radius_nm": 100.0,
            "materials": {OBJ: {"type": "index", "n": 2.0, "k": 0.0}}, "object": OBJ,
            "half_width": 2.0, "outer": 3.0, "n": 2, "order": 2, "quarter": True, "fields": True, "max_gb": 16.0,
            "sweep": {"start": 314.159, "stop": 314.159, "count": 1},
            "resonance": {"lam0_nm": 700.0, "num_modes": 3, "krylov": 0},
            "bands": {"radius": 0.25, "eps": 12.0, "eps_bg": 1.0, "bands": 6, "n_seg": 2, "cells": 4, "krylov": 0}}


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
    return p


def wavelengths(m):
    s = m["sweep"]
    return [float(v) for v in np.linspace(s["start"], s["stop"], int(s["count"]))] if int(s["count"]) > 1 else [float(s["start"])]


def lam_check(m):
    """Wavelengths at which the object material is evaluated."""
    if m["task"] == "resonance":
        return [float(m["resonance"]["lam0_nm"])]
    return wavelengths(m) if m["task"] == "scatter" else []


def eps_list(m):
    spec = m["materials"][m["object"]]
    return [fm.eps_at(spec, lam) for lam in lam_check(m)]


def _frac_int(x, tol=1e-9):
    return abs(x - round(x)) < tol


def validate(m):
    """[(level, text)] for the model."""
    out = []
    t = m["task"]
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
        if m["shape"] == "cube":
            out.append(("warning", "Für den Würfel gibt es keine analytische Referenz; die Kanten sind treppenfrei, aber die Felder an den Ecken sind singulär (langsamere Konvergenz)."))
    if t == "resonance" and m["shape"] == "cube":
        out.append(("warning", "Für den Würfel gibt es keine analytische Referenz."))
    return out


def job(m, max_gb=None):
    t = m["task"]
    j = {"task": t, "name": m["name"], "shape": m["shape"], "n": int(m["n"]), "order": int(m["order"]), "half_width": float(m["half_width"]),
         "outer": float(m["outer"]), "max_gb": float(max_gb if max_gb is not None else m["max_gb"]), "fields": bool(m["fields"])}
    if t == "bands":
        b = m["bands"]
        j.update(radius=float(b["radius"]), eps=float(b["eps"]), eps_bg=float(b["eps_bg"]), bands=int(b["bands"]), n_seg=int(b["n_seg"]), cells=int(b["cells"]),
                 krylov=int(b["krylov"]), half_width=0.5, outer=0.5)
        return j
    j["radius_nm"] = float(m["radius_nm"])
    j["quarter"] = bool(m["quarter"]) and t == "scatter"
    if t == "scatter":
        j["wavelengths_nm"] = wavelengths(m)
        j["eps"] = [[e.real, e.imag] for e in eps_list(m)]
    else:
        r = m["resonance"]
        e = eps_list(m)[0]
        j.update(lam0_nm=float(r["lam0_nm"]), num_modes=int(r["num_modes"]), krylov=int(r["krylov"]), eps=[e.real, e.imag])
    return j


def run_estimate(py, repo, folder, jb, threads=0, timeout=300):
    """Memory estimate of the job from the library (meshes the model, no solve): dict cells, dofs, bytes, human, text; or {'error': ...}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(jb), encoding="utf-8")
    try:
        r = subprocess.run([str(py), str(fr.HERE / "fem_threed_worker.py"), str(folder), "estimate"], cwd=fr._cwd(repo, folder), env=fr.worker_env(repo, threads),
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


def fig_mesh(npz, axis=1, coord=0.0, unit=1e-9, unit_label="nm", title=""):
    """Cut through the mesh (straight tetrahedron edges; the curved sphere surface is drawn exact in the solver) coloured by material tag."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    verts, cells, tags = npz["vertices"], npz["cells"], npz["tags"]
    polys, idx = section_polygons(verts, cells, axis, coord)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    if polys:
        cols = np.array(["#cfe3f5" if tags[i] == 0 else "#f4b183" for i in idx])
        ax.add_collection(PolyCollection([p / unit for p in polys], facecolors=cols, edgecolors="0.35", linewidths=0.4))
        allp = np.vstack(polys) / unit
        ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
        ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax.set_aspect("equal")
    names = "xyz"
    keep = [names[i] for i in range(3) if i != axis]
    ax.set_xlabel(f"{keep[0]} ({unit_label})")
    ax.set_ylabel(f"{keep[1]} ({unit_label})")
    ax.set_title(title or f"Schnitt {names[axis]} = {coord / unit:g} {unit_label} (orange: Objekt)")
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


def fig_field(npz, key, quarter, unit=1e-9, unit_label="nm", radius=None, shape="sphere", title=""):
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
    if radius is not None:
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
            r["Zeit (s)"] = p.get("seconds", np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


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
