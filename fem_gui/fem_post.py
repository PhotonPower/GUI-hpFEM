"""Post-processing of the FEM model builder: results table, derived fields (|E|, intensity, absorbed power density and absorption per material)
and all figures. Fields are in the solver frame: x along the period, y vertical, z along the invariant direction (lines of a grating)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import fem_geometry as fg

EPS0 = 8.8541878128e-12
_trapz = getattr(np, "trapezoid", None) or np.trapz          # NumPy 2 renamed trapz
SWEEP_LABEL = {"wavelength": "Wellenlänge (nm)", "theta": "Einfallswinkel θ (°)", "phi": "Azimut φ (°)", "none": "Einzelpunkt"}


# ----------------------------------------------------------------------------------------------------------------- results
def load_results(folder):
    p = Path(folder) / "results.json"
    if not p.is_file():
        return None
    res = json.loads(p.read_text(encoding="utf-8"))
    for key in ("points", "pscan"):
        for pt in res.get(key, []):
            for k in ("R_orders", "T_orders"):
                if k in pt:
                    pt[k] = {int(m): v for m, v in pt[k].items()}
    return res


def results_dataframe(res, mode):
    rows = []
    for p in res.get("points", []):
        if "error" in p:
            rows.append({"Nr": p.get("index"), SWEEP_LABEL.get(mode, "Wert"): p.get("value"), "Fehler": p["error"]})
            continue
        row = {"Nr": p["index"], "λ (nm)": p["lam_nm"], "θ (°)": p["theta"], "φ (°)": p["phi"], "R": p["R"], "T": p["T"], "A": p["A"],
               "A inkl. Substrat": p["A_incl_substrate"], "R flach": p["ref_R"], "T flach": p["ref_T"]}
        for m, v in sorted(p["R_orders"].items(), key=lambda t: (abs(t[0]), t[0] > 0)):
            row[f"R{m}"] = v
        for m, v in sorted(p["T_orders"].items(), key=lambda t: (abs(t[0]), t[0] > 0)):
            row[f"T{m}"] = v
        ab = p.get("absorbed")
        if ab:
            for name, v in ab["by_material"].items():
                row[f"A[{name}]"] = v
            row["A exakt (Summe)"] = ab["total"]
        row["Freiheitsgrade"], row["Zeit (s)"] = p["dofs"], p["time_s"]
        rows.append(row)
    return pd.DataFrame(rows)


def axis_of(df, mode):
    return {"wavelength": "λ (nm)", "theta": "θ (°)", "phi": "φ (°)"}.get(mode)


def fig_spectrum(df, mode):
    import matplotlib.pyplot as plt

    ok = df[df.get("R").notna()] if "R" in df else df.iloc[0:0]
    if len(ok) == 0:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "keine Ergebnisse", ha="center")
        ax.axis("off")
        return fig
    ax_col = axis_of(ok, mode)
    if ax_col is None or len(ok) == 1:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        r = ok.iloc[0]
        labels, vals = [], []
        for c in ok.columns:
            if c[0] in "RT" and c[1:].lstrip("-").isdigit():
                labels.append(c)
                vals.append(r[c])
        axs[0].bar(labels, vals, color=["#1f77b4" if l[0] == "R" else "#2ca02c" for l in labels])
        axs[0].set_ylabel("Beugungseffizienz")
        axs[0].set_title("Beugungsordnungen")
        tot = {"R": r["R"], "T": r["T"], "A": r["A"] if pd.notna(r["A"]) else r["A inkl. Substrat"]}
        tot = {k: v for k, v in tot.items() if pd.notna(v)}
        axs[1].bar(list(tot), list(tot.values()), color=["#1f77b4", "#2ca02c", "#d62728"][:len(tot)])
        axs[1].set_title("Bilanz" + ("" if pd.notna(r["T"]) else " (A inkl. Substrat)"))
        for a in axs:
            a.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        return fig
    x = ok[ax_col]
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    a = axs[0]
    a.plot(x, ok["R"], "o-", color="#1f77b4", label="R")
    a.plot(x, ok["R flach"], "--", color="#1f77b4", alpha=0.6, label="R ebener Stapel")
    if ok["T"].notna().all():
        a.plot(x, ok["T"], "s-", color="#2ca02c", label="T")
        a.plot(x, ok["T flach"], "--", color="#2ca02c", alpha=0.6, label="T ebener Stapel")
        a.plot(x, ok["A"], "^-", color="#d62728", label="A = 1 − R − T")
    else:
        a.plot(x, ok["A inkl. Substrat"], "^-", color="#d62728", label="A inkl. Substrat = 1 − R")
    a.set_ylabel("Anteil der einfallenden Leistung")
    a.legend(fontsize=8, ncol=2)
    a.grid(alpha=0.3)
    a = axs[1]
    for c in ok.columns:
        if c[0] in "RT" and c[1:].lstrip("-").isdigit():
            a.plot(x, ok[c], "o-" if c[0] == "R" else "s--", ms=3, label=c)
    a.set_ylabel("Beugungseffizienz je Ordnung")
    a.set_xlabel(SWEEP_LABEL[mode])
    a.legend(fontsize=8, ncol=3)
    a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_pscan(ps):
    import matplotlib.pyplot as plt

    ok = [p for p in ps if "error" not in p]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    if len(ok) < 2:
        axs[0].text(0.5, 0.5, "zu wenige Punkte", ha="center")
        return fig
    dofs = np.array([p["dofs"] for p in ok], float)
    order = [p["order"] for p in ok]
    for key, lab in (("R", "R"), ("T", "T"), ("A", "A")):
        v = np.array([np.nan if p.get(key) is None else p[key] for p in ok], float)
        if np.isfinite(v).all():
            axs[0].plot(order, v, "o-", label=lab)
            d = np.abs(v - v[-1])[:-1]
            axs[1].semilogy(dofs[:-1], np.maximum(d, 1e-12), "o-", label=f"|Δ{lab}| gegen höchste Ordnung")
    axs[0].set_xlabel("Polynomordnung p")
    axs[0].set_ylabel("Wert")
    axs[0].legend()
    axs[1].set_xlabel("Freiheitsgrade")
    axs[1].set_ylabel("Abweichung")
    axs[1].legend(fontsize=7)
    axs[2].plot(order, [p["time_s"] for p in ok], "o-")
    axs[2].set_xlabel("Polynomordnung p")
    axs[2].set_ylabel("Rechenzeit (s)")
    for a in axs:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------------------------ fields
def load_map(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def derived(mp, model, A_energy=None):
    """Fields and derived quantities on the map grid; absorption per material from the absorbed power density."""
    x, y = mp["x_nm"], mp["y_nm"]
    E = mp["E"].astype(complex)
    X, Y = np.meshgrid(x, y, indexing="ij")
    idx = fg.material_index_map(model, X, Y)
    eps = np.asarray(mp["eps"], dtype=complex)[idx]
    E2 = (np.abs(E) ** 2).sum(axis=2)
    Q = 0.5 * float(mp["omega"]) * EPS0 * eps.imag * E2                       # W/m^3 for |E0| = 1 V/m
    P_m = (x[-1] - x[0]) * 1e-9
    names = [str(n) for n in mp["eps_names"]]
    regions = {}
    for i, n in enumerate(names):
        mask = idx == i
        if mask.any() and np.any(eps[mask].imag > 0):
            integral = _trapz(_trapz(np.where(mask, Q, 0.0), y * 1e-9, axis=1), x * 1e-9, axis=0)
            regions[n] = float(integral / (float(mp["S_inc"]) * P_m))
    A_map = float(sum(regions.values())) if regions else 0.0
    norm = {n: v * A_energy / A_map for n, v in regions.items()} if (A_energy is not None and A_map > 0) else None
    return dict(x=x, y=y, X=X, Y=Y, E=E, Eabs=np.sqrt(E2), E2=E2, Q=Q, idx=idx, regions=regions, A_map=A_map, regions_norm=norm,
                map_quality=(A_map / A_energy if (A_energy and A_energy > 1e-9) else None))


QUANTITIES = {
    "|E| (Betrag)": ("Eabs", "|E| / |E₀|"),
    "|E|² (Intensität)": ("E2", "|E|² / |E₀|²"),
    "|E_x|": ("c0", "|E_x|"), "|E_y| (vertikal)": ("c1", "|E_y|"), "|E_z| (entlang der Linien)": ("c2", "|E_z|"),
    "Re E_x": ("r0", "Re E_x"), "Re E_y": ("r1", "Re E_y"), "Re E_z": ("r2", "Re E_z"),
    "Im E_x": ("i0", "Im E_x"), "Im E_y": ("i1", "Im E_y"), "Im E_z": ("i2", "Im E_z"),
    "Absorbierte Leistungsdichte Q": ("Q", "Q (W/m³) für |E₀| = 1 V/m"),
}


def quantity(d, key):
    code = QUANTITIES[key][0]
    if code in ("Eabs", "E2", "Q"):
        return d[code]
    kind, comp = code[0], int(code[1])
    c = d["E"][:, :, comp]
    return {"c": np.abs(c), "r": c.real, "i": c.imag}[kind]


def fig_map(mp, d, model, key, periods=1, cmap=None, vmax=None, log=False, mesh_npz=None, show_geometry=True, figsize=(7.4, 6.2)):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    x, y = d["x"], d["y"]
    f = quantity(d, key)
    P = float(mp["period_nm"]) if "period_nm" in mp else x[-1] - x[0]
    if periods > 1:
        kx = float(mp["kx"]) if "kx" in mp else 0.0
        phase = np.exp(1j * kx * P * 1e-9)
        parts, xs = [f], [x]
        sym = key.startswith("Re") or key.startswith("Im")
        for k in range(1, periods):
            if sym:                                                         # Bloch phase for the real / imaginary part
                c = d["E"][:, :, int(QUANTITIES[key][0][1])] * phase ** k
                parts.append(c.real if key.startswith("Re") else c.imag)
            else:
                parts.append(f)
            xs.append(x + k * P)
        f = np.concatenate([parts[0]] + [p[1:] for p in parts[1:]], axis=0)
        x = np.concatenate([xs[0]] + [xx[1:] for xx in xs[1:]])
    signed = key.startswith("Re") or key.startswith("Im")
    top = vmax if vmax else float(np.nanmax(np.abs(f))) or 1.0
    if signed:
        norm, cm = Normalize(-top, top), cmap or "RdBu_r"
    elif log and f.min() >= 0:
        lo = max(top * 1e-4, float(f[f > 0].min()) if (f > 0).any() else 1e-12)
        norm, cm = LogNorm(lo, top), cmap or "inferno"
    else:
        norm, cm = Normalize(0, top), cmap or "inferno"
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(x, y, f.T, shading="gouraud", cmap=cm, norm=norm)
    fig.colorbar(im, ax=ax, label=QUANTITIES[key][1], fraction=0.046)
    if show_geometry:
        lay = fg.layout(model)
        for yi in lay["interfaces"]:
            ax.axhline(yi, color="w", lw=0.6, ls="--", alpha=0.7)
        for k in range(periods):
            for s in model["shapes"]:
                p = fg.shape_polygon(s)
                for off in (-P, 0.0, P):
                    q = p + np.array([off + k * P, 0.0])
                    if q[:, 0].max() > x[0] and q[:, 0].min() < x[-1]:
                        ax.plot(np.append(q[:, 0], q[0, 0]), np.append(q[:, 1], q[0, 1]), color="w", lw=0.9)
    if mesh_npz is not None:
        from matplotlib.collections import LineCollection
        z = np.load(mesh_npz)
        segs = np.concatenate([z["xy"][z["tri"][:, [0, 1]]], z["xy"][z["tri"][:, [1, 2]]], z["xy"][z["tri"][:, [2, 0]]]])
        ax.add_collection(LineCollection(segs, colors="w", linewidths=0.15, alpha=0.5))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"{key}   λ = {float(mp['lam_nm']):.1f} nm, θ = {float(mp['theta']):.1f}°, φ = {float(mp['phi']):.1f}°, {str(mp['pol'])}", fontsize=9)
    fig.tight_layout()
    return fig


def fig_cut(d, model, axis, pos_nm, comps=("Eabs", "c0", "c1", "c2")):
    """Line plot of |E| and its components along x (at height pos_nm) or along y (at x = pos_nm)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    x, y = d["x"], d["y"]
    labels = {"Eabs": "|E|", "c0": "|E_x|", "c1": "|E_y|", "c2": "|E_z|"}
    if axis == "x":
        j = int(np.argmin(abs(y - pos_nm)))
        for c in comps:
            v = d["Eabs"][:, j] if c == "Eabs" else np.abs(d["E"][:, j, int(c[1])])
            ax.plot(x, v, label=labels[c], lw=2 if c == "Eabs" else 1.2)
        ax.set_xlabel("x (nm)")
        ax.set_title(f"Schnitt bei y = {y[j]:.0f} nm")
    else:
        i = int(np.argmin(abs(x - pos_nm)))
        for c in comps:
            v = d["Eabs"][i, :] if c == "Eabs" else np.abs(d["E"][i, :, int(c[1])])
            ax.plot(y, v, label=labels[c], lw=2 if c == "Eabs" else 1.2)
        for yi in fg.layout(model)["interfaces"]:
            ax.axvline(yi, color="k", ls=":", lw=0.7)
        ax.set_xlabel("y (nm)")
        ax.set_title(f"Schnitt bei x = {x[i]:.0f} nm (gepunktet: Grenzflächen des Stapels)")
    ax.set_ylabel("Feldamplitude / |E₀|")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_absorption(regions, total_energy=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.4))
    names = list(regions)
    ax.bar(names, [regions[n] for n in names], color="#d62728")
    if total_energy is not None:
        ax.axhline(total_energy, color="k", ls="--", label=f"1 − R − T = {total_energy:.4f}")
        ax.legend(fontsize=8)
    ax.set_ylabel("absorbierter Anteil")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_materials(model, lam_min, lam_max):
    """n and k of the materials of the model over the wavelength range (the sweep range, widened a little)."""
    import matplotlib.pyplot as plt

    import fem_materials as fm

    lo, hi = lam_min, lam_max
    if hi - lo < 1.0:
        lo, hi = 0.85 * lam_min, 1.15 * lam_max
    lams = np.linspace(lo, hi, 80)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, spec in model["materials"].items():
        n, k = [], []
        for l in lams:
            try:
                a, b = fm.nk(spec, l)
            except Exception:
                a = b = np.nan
            n.append(a)
            k.append(b)
        if np.isfinite(n).any():
            axs[0].plot(lams, n, label=name)
            axs[1].plot(lams, k, label=name)
    axs[0].set_ylabel("n")
    axs[1].set_ylabel("k")
    for a in axs:
        a.set_xlabel("Wellenlänge (nm)")
        a.grid(alpha=0.3)
    axs[1].set_yscale("symlog", linthresh=0.01)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_adaptive_steps(steps):
    """Convergence of an hp-adaptive run: observables, change against the last step and the estimator over the DoFs, h and p over the steps."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    if len(steps) < 1:
        axs[0].text(0.5, 0.5, "keine Schritte", ha="center")
        return fig
    dofs = np.array([s["dofs"] for s in steps], float)
    R = np.array([s["R"] for s in steps], float)
    T = np.array([np.nan if s.get("T") is None else s["T"] for s in steps], float)
    axs[0].semilogx(dofs, R, "o-", label="R")
    if np.isfinite(T).all():
        axs[0].semilogx(dofs, T, "s-", label="T")
    axs[0].set_xlabel("Freiheitsgrade")
    axs[0].set_ylabel("Wert")
    axs[0].legend()
    d = np.abs(R - R[-1])[:-1]
    if len(d):
        axs[1].loglog(dofs[:-1], np.maximum(d, 1e-12), "o-", label="|R − R(letzter Schritt)|")
        if np.isfinite(T).all():
            axs[1].loglog(dofs[:-1], np.maximum(np.abs(T - T[-1])[:-1], 1e-12), "s-", label="|T − T(letzter Schritt)|")
    axs[1].loglog(dofs, [max(s["eta"], 1e-12) for s in steps], "^--", color="0.4", label="Fehlerschätzer η")
    axs[1].set_xlabel("Freiheitsgrade")
    axs[1].legend(fontsize=7)
    axs[2].plot([s["step"] for s in steps], [s["max_p"] for s in steps], "o-", label="größtes p")
    ax2 = axs[2].twinx()
    ax2.semilogy([s["step"] for s in steps], [s["cells"] for s in steps], "s--", color="C1", label="Dreiecke")
    axs[2].set_xlabel("Schritt")
    axs[2].set_ylabel("größtes p")
    ax2.set_ylabel("Dreiecke")
    for a in axs:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_hpmesh(npz_path, model=None, zoom=None, figsize=(7.4, 6.2)):
    """Final hp mesh coloured by the polynomial order of every cell."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    z = np.load(npz_path, allow_pickle=False)
    xy, tri, order = z["xy_nm"], z["tri"], z["order"]
    fig, ax = plt.subplots(figsize=figsize)
    pc = PolyCollection(xy[tri], array=order.astype(float), cmap="viridis", edgecolors="0.3", linewidths=0.2)
    pc.set_clim(order.min() - 0.5, order.max() + 0.5)
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, label="Polynomordnung p", ticks=range(int(order.min()), int(order.max()) + 1))
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    if model is not None:
        for s in model["shapes"]:
            q = fg.shape_polygon(s)
            ax.plot(np.append(q[:, 0], q[0, 0]), np.append(q[:, 1], q[0, 1]), color="r", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"hp-Netz: {len(tri)} Dreiecke, p = {int(order.min())} bis {int(order.max())}", fontsize=9)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------- absorption over the sweep (exact, from the mesh quadrature)
def fig_absorption_sweep(df, mode):
    """Absorbed fraction per material over the sweep (volume quadrature of the Joule heating, hpfem M15 F4), with the energy balance as reference."""
    import matplotlib.pyplot as plt

    cols = [c for c in df.columns if c.startswith("A[")]
    ok = df[df["R"].notna()] if "R" in df else df
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x_col = axis_of(ok, mode)
    if not cols or len(ok) == 0:
        ax.text(0.5, 0.5, "keine exakten Absorptionswerte (hpfem ohne absorbed_power_by_tag)", ha="center")
        ax.axis("off")
        return fig
    x = ok[x_col] if (x_col and len(ok) > 1) else np.arange(len(ok))
    for c in cols:
        ax.plot(x, ok[c], "o-", label=c[2:-1])
    if "A exakt (Summe)" in ok:
        ax.plot(x, ok["A exakt (Summe)"], "k:", label="Summe (exakt)")
    ref = ok["A"] if ("A" in ok and ok["A"].notna().all()) else ok["A inkl. Substrat"]
    ax.plot(x, ref, "--", color="0.4", label="Energiebilanz 1 − R − T" if ("A" in ok and ok["A"].notna().all()) else "Energiebilanz 1 − R")
    ax.set_xlabel(SWEEP_LABEL.get(mode, "Punkt") if (x_col and len(ok) > 1) else "Punkt Nr.")
    ax.set_ylabel("absorbierter Anteil")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------- exact element representation (triangulate)
def load_tri(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def derived_tri(tri, model):
    """Fields, absorbed power density (exact material per element) on the subdivided mesh."""
    pts, simp = tri["points_nm"], tri["simplices"].astype(int)
    E = tri["values"].astype(complex)
    vtag = np.zeros(len(pts), dtype=int)
    vtag[simp] = tri["tag"][:, None]
    eps = np.asarray(tri["eps"], dtype=complex)
    eps_v = eps[np.clip(vtag - 1, 0, len(eps) - 1)]
    E2 = (np.abs(E) ** 2).sum(axis=1)
    Q = 0.5 * float(tri["omega"]) * EPS0 * eps_v.imag * E2
    return dict(points=pts, simplices=simp, E=E, Eabs=np.sqrt(E2), E2=E2, Q=Q, vtag=vtag)


def fig_map_tri(tri, d, model, key, periods=1, cmap=None, vmax=None, log=False, show_geometry=True, figsize=(7.4, 6.2)):
    """Map on the element triangulation: every element in its own sub-triangles, so jumps of the field across material boundaries stay sharp."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.tri import Triangulation

    P = float(tri["period_nm"])
    pts, simp, E = d["points"], d["simplices"], d["E"]
    code = QUANTITIES[key][0]
    if code in ("Eabs", "E2", "Q"):
        f = d[code]
        sym = False
    else:
        kind, comp = code[0], int(code[1])
        f = {"c": np.abs(E[:, comp]), "r": E[:, comp].real, "i": E[:, comp].imag}[kind]
        sym = kind in ("r", "i")
    X, S_, F = [pts[:, 0]], [simp], [f]
    if periods > 1:
        phase = np.exp(1j * float(tri["kx"]) * P * 1e-9)
        off = len(pts)
        for k in range(1, periods):
            fk = f
            if sym:
                ck = E[:, int(code[1])] * phase ** k
                fk = ck.real if code[0] == "r" else ck.imag
            X.append(pts[:, 0] + k * P)
            S_.append(simp + k * off)
            F.append(fk)
    x = np.concatenate(X)
    y = np.tile(pts[:, 1], periods)
    simplices = np.concatenate(S_)
    values = np.concatenate(F)
    top = vmax if vmax else float(np.nanmax(np.abs(values))) or 1.0
    if sym:
        norm, cm = Normalize(-top, top), cmap or "RdBu_r"
    elif log and values.min() >= 0:
        lo = max(top * 1e-4, float(values[values > 0].min()) if (values > 0).any() else 1e-12)
        norm, cm = LogNorm(lo, top), cmap or "inferno"
        values = np.maximum(values, lo)
    else:
        norm, cm = Normalize(0, top), cmap or "inferno"
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.tripcolor(Triangulation(x, y, simplices), values, shading="gouraud", cmap=cm, norm=norm)
    fig.colorbar(im, ax=ax, label=QUANTITIES[key][1], fraction=0.046)
    if show_geometry:
        lay = fg.layout(model)
        for yi in lay["interfaces"]:
            ax.axhline(yi, color="w", lw=0.6, ls="--", alpha=0.7)
        for k in range(periods):
            for s_ in model["shapes"]:
                p = fg.shape_polygon(s_)
                for off_ in (-P, 0.0, P):
                    q = p + np.array([off_ + k * P, 0.0])
                    if q[:, 0].max() > x.min() and q[:, 0].min() < x.max():
                        ax.plot(np.append(q[:, 0], q[0, 0]), np.append(q[:, 1], q[0, 1]), color="w", lw=0.9)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"{key}   λ = {float(tri['lam_nm']):.1f} nm, θ = {float(tri['theta']):.1f}°, φ = {float(tri['phi']):.1f}°, {str(tri['pol'])}  (Elementdarstellung)", fontsize=9)
    fig.tight_layout()
    return fig
