"""Worker for the cross grating (hp-FEM M21): runs in the Python that has hpfem, called by the app.

    python fem_xgrating_worker.py <job folder>             computation
    python fem_xgrating_worker.py --import <fg.json> <out.json>   FourierGrating file -> model of the app (hpfem.fouriergrating)

Computation: reads <folder>/job.json (fem_xgrating.job) and runs ``hpfem.crossed.solve`` per wavelength (one factorisation serves s and p), then,
if the job asks for it, the convergence study in the polynomial order at the first wavelength (fixed ``elements_per_wavelength``). Writes
<folder>/results.json after every solve and log lines on stdout (PROGRESS i/n, ERROR ..., CANCELLED, DONE). Cancellation: the file
<folder>/cancel, checked between the solves.

Units: nm and degrees in the job, metres and radians in hpfem. Stack normal z, periodic in x and y.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_materials as fm  # noqa: E402

NM = 1e-9


def say(*a):
    print(*a, flush=True)


def material(h, job, name, lam_nm):
    spec = job["materials"][name]
    if fm.is_crystal(spec):
        raise ValueError(f"Material '{name}': anisotrope Kristalle sind im Kreuzgitter nicht möglich")
    return h.Material(complex(fm.eps_at(spec, lam_nm)), 1.0)


def build_grating(h, cg, job, lam_nm):
    """``hpfem.crossed.CrossedGrating`` from the job (nm -> m, degrees -> radians) with the materials evaluated at ``lam_nm``."""
    layers = []
    for la in job["layers"]:
        shapes = []
        for s in la.get("shapes", []):
            mat, t = material(h, job, s["material"], lam_nm), s["type"]
            c = (s.get("cx", 0.0) * NM, s.get("cy", 0.0) * NM)
            ang = math.radians(s.get("angle", 0.0))
            if t == "circle":
                shapes.append(cg.Shape("circle", mat, center=c, radius=s["radius"] * NM))
            elif t == "ellipse":
                shapes.append(cg.Shape("ellipse", mat, center=c, rx=s["rx"] * NM, ry=s["ry"] * NM, angle=ang))
            elif t == "rect":
                shapes.append(cg.Shape("rectangle", mat, center=c, width=s["wx"] * NM, height=s["wy"] * NM, angle=ang))
            elif t == "polygon":
                pts = [(float(x) * NM, float(y) * NM) for x, y in s["points"]]
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
                shapes.append(cg.Shape("polygon", mat, points=pts if area > 0 else pts[::-1]))      # counter-clockwise
            else:
                raise ValueError(f"unbekannte Form '{t}'")
        layers.append(cg.Layer(la["thickness_nm"] * NM, material(h, job, la["material"], lam_nm), shapes))
    return cg.CrossedGrating((job["period_x_nm"] * NM, job["period_y_nm"] * NM), material(h, job, job["cover"], lam_nm),
                             material(h, job, job["substrate"], lam_nm), layers)


def summary(res):
    return dict(R=float(res.R), T=float(res.T), A=None if res.absorption is None else float(res.absorption),
                balance=None if res.balance is None else float(res.balance),
                rows=[dict(side=r["side"], m=int(r["m"]), n=int(r["n"]), efficiency=float(r["efficiency"])) for r in res.table()])


def solve_point(h, cg, job, lam_nm, order, epw, field_folder=None, index=0):
    inc = job["incidence"]
    t0 = time.time()
    g = build_grating(h, cg, job, lam_nm)
    res = cg.solve(g, lam_nm * NM, math.radians(inc["theta"]), math.radians(inc["phi"]), inc["pol"], order=int(order),
                   elements_per_wavelength=epw or None)
    out = dict(lam_nm=lam_nm, order=int(order), time_s=time.time() - t0, dofs=int(res.num_dofs), cells=int(res.num_cells), **summary(res))
    out["pols"] = {k: summary(res.polarised(k)) for k in ("s", "p")}
    if field_folder is not None:
        try:
            out["field_file"] = sample_planes(res, job, lam_nm, field_folder, f"field_{index}")
        except Exception as exc:          # the table is still valid without the field
            say(f"ERROR Nahfeld: {exc}")
            out["field_error"] = str(exc)
    return out


def sample_planes(res, job, lam_nm, folder, name):
    """|E|²/|E₀|² (total field, the job's polarisation) on the cut planes xy, xz, yz -> <folder>/<name>.npz; NaN outside the mesh."""
    import numpy as np
    fl = job["field"]
    n = max(int(fl.get("resolution", 80)), 8)
    px, py = job["period_x_nm"] * NM, job["period_y_nm"] * NM
    depth = sum(la["thickness_nm"] for la in job["layers"]) * NM
    lam = lam_nm * NM
    xs, ys = np.linspace(0, px, n), np.linspace(0, py, max(int(round(n * py / px)), 8))
    zs = np.linspace(-depth - 0.5 * lam, 0.5 * lam, n)
    x0, y0, z0 = fl["x_nm"] * NM, fl["y_nm"] * NM, fl["z_nm"] * NM

    def e2(a, b, build):
        A, B = np.meshgrid(a, b, indexing="ij")
        pts = build(A.ravel(), B.ravel())
        return (np.abs(res.field(pts)) ** 2).sum(axis=1).reshape(A.shape)

    xy = e2(xs, ys, lambda u, v: np.column_stack([u, v, np.full_like(u, z0)]))
    xz = e2(xs, zs, lambda u, v: np.column_stack([u, np.full_like(u, y0), v]))
    yz = e2(ys, zs, lambda u, v: np.column_stack([np.full_like(u, x0), u, v]))
    np.savez_compressed(folder / f"{name}.npz", xs=xs, ys=ys, zs=zs, xy=xy, xz=xz, yz=yz, x_nm=fl["x_nm"], y_nm=fl["y_nm"], z_nm=fl["z_nm"])
    return f"{name}.npz"


def main_run(folder):
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    from hpfem import crossed as cg
    hpfem.set_log_level("warn")
    inc, me = job["incidence"], job["mesh"]
    epw = float(me.get("elements_per_wavelength") or 0.0)
    lams = [float(v) for v in job["wavelengths_nm"]]
    conv_orders = [int(o) for o in job.get("convergence_orders", [])]
    steps = [("conv", o) for o in conv_orders if o != int(me["order"])] + [("main", i) for i in range(len(lams))]
    say(f"hpfem {getattr(hpfem, '__version__', '?')}, {len(lams)} Wellenlänge(n), Ordnung p = {me['order']}"
        + (f", Konvergenzstudie p = {conv_orders}" if conv_orders else ""))
    out = dict(meta=dict(name=job.get("name", ""), started=time.strftime("%Y-%m-%d %H:%M:%S"), hpfem=str(getattr(hpfem, "__version__", "?")),
                         order=int(me["order"]), elements_per_wavelength=epw, theta=inc["theta"], phi=inc["phi"], pol=inc["pol"],
                         period_nm=[job["period_x_nm"], job["period_y_nm"]]), points=[], convergence=[])
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    for i, (kind, v) in enumerate(steps):
        if (folder / "cancel").exists():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return 0
        lam, order = (lams[0], v) if kind == "conv" else (lams[v], int(me["order"]))
        say(f"PROGRESS {i + 1}/{len(steps)} lambda = {lam:.3f} nm, p = {order}" + (" (Konvergenzstudie)" if kind == "conv" else ""))
        try:
            res = solve_point(hpfem, cg, job, lam, order, epw, folder if kind == "main" and job.get("field") else None, v)
            res["index"] = v if kind == "main" else None
            (out["convergence"] if kind == "conv" else out["points"]).append(res)
            if kind == "main" and v == 0 and order in conv_orders:
                out["convergence"].append(res)
            bal = "" if res["balance"] is None else f", Bilanz {res['balance']:+.2e}"
            say(f"  {res['time_s']:.1f} s, {res['dofs']} Unbekannte: ΣR = {res['R']:.5f}, ΣT = {res['T']:.5f}{bal}")
        except Exception as exc:
            say(f"ERROR lambda = {lam:g} nm, p = {order}: {exc}")
            (out["convergence"] if kind == "conv" else out["points"]).append(dict(lam_nm=lam, order=order, error=str(exc)))
        out["convergence"].sort(key=lambda c: c.get("order", 0))
        flush()
    say("DONE")
    return 0


# ---------------------------------------------------------------------------------------------------------------- FourierGrating import
def to_model(fg_path):
    """Model of the app (``fem_xgrating.default_model`` format) from a FourierGrating file, through ``hpfem.fouriergrating`` (frames, layer order)."""
    from hpfem import fouriergrating as fgi
    spec = json.loads(Path(fg_path).read_text(encoding="utf-8"))
    mdl = fgi.load(spec)
    g = mdl.grating
    mats = {}
    for name, raw in spec.get("materials", {}).items():
        try:
            mats[name] = fm.normalize_spec(raw if isinstance(raw, dict) else {"n": raw})
        except ValueError:
            if isinstance(raw, dict) and "eps" in raw:
                e = raw["eps"]
                mats[name] = {"type": "eps", "re": float(e[0] if isinstance(e, (list, tuple)) else e), "im": float(e[1]) if isinstance(e, (list, tuple)) else 0.0}
            else:
                raise
    if not mats:
        raise ValueError("Die Datei enthält keine Materialien")

    def name_of(material_obj):
        n = complex(material_obj.refractive_index)
        for k, sp in mats.items():
            if abs(fm.eps_at(sp, 500.0) - n * n) < 1e-9 * max(1.0, abs(n * n)):
                return k
        raise ValueError(f"Material mit n = {n:g} nicht in der Datei gefunden")

    layers = []
    for la in g.layers:
        shapes = []
        for s in la.shapes:
            d = dict(kind="hole" if name_of(s.material) == name_of(g.superstrate) else "pillar", material=name_of(s.material))
            if s.kind == "circle":
                d.update(type="circle", cx=s.center[0] / NM, cy=s.center[1] / NM, radius=s.radius / NM)
            elif s.kind == "ellipse":
                d.update(type="ellipse", cx=s.center[0] / NM, cy=s.center[1] / NM, rx=s.rx / NM, ry=s.ry / NM, angle=math.degrees(s.angle))
            elif s.kind == "rectangle":
                d.update(type="rect", cx=s.center[0] / NM, cy=s.center[1] / NM, wx=s.width / NM, wy=s.height / NM, angle=math.degrees(s.angle))
            else:
                d.update(type="polygon", points=[[x / NM, y / NM] for x, y in s.points])
            shapes.append(d)
        layers.append(dict(thickness_nm=la.thickness / NM, material=name_of(la.material), shapes=shapes))
    incs = mdl.incidences
    first = incs[0]
    pol = {(1.0, 0.0): "s", (0.0, -1.0): "p"}.get((round(first.polarisation[0], 9) + 0.0, round(first.polarisation[1], 9) + 0.0))
    if pol is None:
        raise ValueError(f"δ = {first.fg.get('delta_deg')}° (gemischte Polarisation) ist nicht umsetzbar: δ = 0° (s) oder 90° (p) verwenden")
    lams = sorted({round(i.wavelength / NM, 6) for i in incs})
    model = dict(kind="xgrating", name=Path(fg_path).stem + " (FourierGrating)", period_x_nm=g.period[0] / NM, period_y_nm=g.period[1] / NM,
                 materials=mats, cover=name_of(g.superstrate), substrate=name_of(g.substrate), layers=layers,
                 incidence=dict(theta=math.degrees(first.theta), phi=math.degrees(first.phi) + 0.0, pol=pol, wavelength_nm=lams[0]),
                 sweep=dict(mode="single", start=lams[0], stop=lams[-1], n=len(lams)))
    notes = []
    if len(lams) > 1:
        if all(abs(lams[k + 1] - lams[k] - (lams[1] - lams[0])) < 1e-6 for k in range(len(lams) - 1)):
            model["sweep"]["mode"] = "wavelength"
        else:
            notes.append(f"{len(lams)} ungleich verteilte Wellenlängen: nur die erste ({lams[0]:g} nm) übernommen")
    if len({(round(i.theta, 9), round(i.phi, 9), i.polarisation) for i in incs}) > 1:
        notes.append("Winkel- oder Polarisationsdurchlauf in der Datei: nur der erste Einfall übernommen")
    return model, notes


def main_import(src, dst):
    try:
        model, notes = to_model(src)
    except Exception as exc:
        say(f"ERROR {type(exc).__name__}: {exc}")
        return 1
    Path(dst).write_text(json.dumps(dict(model=model, notes=notes), ensure_ascii=False), encoding="utf-8")
    say("DONE")
    return 0


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "--import":
        return main_import(argv[1], argv[2])
    return main_run(Path(argv[0]))


if __name__ == "__main__":
    raise SystemExit(main())
