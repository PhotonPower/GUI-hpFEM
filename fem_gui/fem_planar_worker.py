"""Worker for the emitter above a planar layer stack (no mesh): runs in the Python that has hpfem, called by the app.

    python fem_planar_worker.py <job folder>

Reads <folder>/job.json (written by fem_emit.planar_job) and computes for every wavelength the emitter response table of
hpfem.dataports.planar_emitter_response (Sommerfeld / Fresnel, semi-analytic): A_tot = diag(F_par, F_par, F_perp) (Purcell factors),
A_rad (radiated into the cover and the lossless substrate), A_det of the detector regions (numerical apertures) and the far field of
the unit dipoles. Writes <folder>/results.json after every wavelength, <folder>/table_<i>.json (the BindOptiX envelope of the table)
and log.txt lines on stdout (PROGRESS i/n, ERROR ..., CANCELLED, DONE). Cancellation: the file <folder>/cancel, checked between the wavelengths.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_materials as fm  # noqa: E402

NM = 1e-9


def say(*a):
    print(*a, flush=True)


def build_stack(h, spec, lam_nm):
    """hpfem.LayerStack2D of the job's stack at lam_nm (top interface at z = 0)."""
    mats = spec["materials"]

    def mat(name):
        e = complex(fm.eps_at(mats[name], lam_nm))
        return h.Material(e), e

    cover, e_cov = mat(spec["cover"])
    if abs(e_cov.imag) > 1e-9 or e_cov.real <= 0:
        raise ValueError("Das Einfallsmedium (Wirt des Emitters) muss verlustfrei sein")
    layers = [h.Layer(mat(ly["material"])[0], float(ly["thickness_nm"]) * NM) for ly in spec["layers"]]
    sub, e_sub = mat(spec["substrate"])
    return h.LayerStack2D(cover, layers, sub, 0.0), e_cov, e_sub


def detectors_for(dp, spec, e_cov, e_sub):
    """dataports.Detector regions from numerical apertures (NA = n sin(theta) in the medium of the side)."""
    out = []
    for d in spec["detectors"]:
        n = math.sqrt(e_cov.real) if d["side"] == "cover" else math.sqrt(abs(e_sub))
        lo, hi = float(d.get("na_min", 0.0)), float(d.get("na_max", n))
        t0 = math.asin(min(lo / n, 1.0))
        t1 = math.asin(min(hi / n, 1.0))
        out.append(dp.Detector(str(d["name"]), str(d["side"]), t0, t1))
    return out


def pattern(ff, n_theta_out=181):
    """dP/dOmega in units of the free dipole's power (per sr), mean over phi, for the dipoles in the plane (x, y) and along z,
    on theta (from the outward normal of the half-space), shape (nz, ntheta)."""
    f = np.abs(ff.field) ** 2                                   # (nz, nt, np, 3 components, 3 dipoles)
    i_x = f[..., 0].sum(axis=-1)
    i_y = f[..., 1].sum(axis=-1)
    i_z = f[..., 2].sum(axis=-1)
    par = 0.5 * (i_x + i_y).mean(axis=2) * ff.factor            # (nz, nt)
    perp = i_z.mean(axis=2) * ff.factor
    # the weights are solid-angle weights: factor * |E|^2 is the power per steradian in units of the free dipole's power
    return np.asarray(ff.theta), par, perp


def run_point(h, dp, spec, lam_nm, folder, i):
    stack, e_cov, e_sub = build_stack(h, spec, lam_nm)
    heights = np.asarray(spec["heights_nm"], dtype=float) * NM
    dets = detectors_for(dp, spec, e_cov, e_sub)
    t0 = time.time()
    resp = dp.planar_emitter_response(stack, lam_nm * NM, heights, detectors=dets, theta_points=int(spec.get("theta_points", 96)),
                                      phi_points=int(spec.get("phi_points", 16)))
    t = resp.table
    a_tot, a_rad, a_det = t.a_tot[0, 0], t.a_rad[0, 0], t.a_det[0, 0]      # (nz,3,3), (nz,3,3), (nz,D,3,3)
    res = dict(index=i, lam_nm=float(lam_nm), n_host=float(t.n_host), heights_nm=list(map(float, heights / NM)),
               F_par=a_tot[:, 0, 0].tolist(), F_perp=a_tot[:, 2, 2].tolist(),
               rad_par=a_rad[:, 0, 0].tolist(), rad_perp=a_rad[:, 2, 2].tolist(),
               detectors=[dict(name=d.name, side=d.side, theta_min=d.theta_min, theta_max=d.theta_max) for d in dets],
               det_par=a_det[:, :, 0, 0].tolist(), det_perp=a_det[:, :, 2, 2].tolist(),
               convergence=resp.convergence, time_s=time.time() - t0, substrate_lossless=bool(abs(e_sub.imag) < 1e-9))
    far = {}
    for side, ff in resp.far_field.items():
        th, par, perp = pattern(ff)
        p3 = np.einsum("tp,ztpjc->zj", ff.weights, np.abs(ff.field) ** 2) * ff.factor        # power per dipole x, y, z (free dipole = 1)
        far[side] = dict(theta=th.tolist(), par=par.tolist(), perp=perp.tolist(), n=float(ff.index),
                         power_par=(0.5 * (p3[:, 0] + p3[:, 1])).tolist(), power_perp=p3[:, 2].tolist())
    res["far"] = far
    (Path(folder) / f"table_{i}.json").write_text(t.to_json(), encoding="utf-8")
    return res


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    from hpfem import dataports as dp
    hpfem.set_log_level("warn")
    spec = job["planar"]
    values = [float(v) for v in spec["wavelengths_nm"]]
    out = dict(meta=dict(name=spec.get("name", ""), started=time.strftime("%Y-%m-%d %H:%M:%S"), hpfem=str(getattr(hpfem, "__version__", "?"))), points=[])
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    say(f"hpfem {out['meta']['hpfem']}")
    for i, lam in enumerate(values):
        if (folder / "cancel").exists():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return 0
        say(f"PROGRESS {i + 1}/{len(values)} lambda = {lam:.3f} nm")
        try:
            res = run_point(hpfem, dp, spec, lam, folder, i)
            out["points"].append(res)
            say(f"  {res['time_s']:.1f} s: F_par = {res['F_par'][0]:.4g}, F_perp = {res['F_perp'][0]:.4g} at {res['heights_nm'][0]:g} nm")
        except Exception as exc:
            say(f"ERROR point {i}: {exc}")
            out["points"].append(dict(index=i, lam_nm=lam, error=str(exc)))
        flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
