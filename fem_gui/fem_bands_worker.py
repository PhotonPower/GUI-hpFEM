"""Worker for band structures of photonic crystals (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_bands_worker.py <job folder>

Reads <folder>/job.json (fem_bands.job) and <folder>/mesh.msh (fem_bands.build_mesh; lengths in units of the lattice constant a = 1, tags 101 / 102
left / right with the translation a1, 103 / 104 bottom / top with a2). Per k point of the path: hpfem.BandStructure2D.bands(k) (in-plane E, lossless):
the lowest bands f = omega a / (2 pi c) (the second zero at Gamma is dropped, see fem_bands.clean_bands) and the residual; optionally the band fields at the
corner points of the path (mode_<i>.npz: points, simplices, et_<band> (n, 2)).

Writes <folder>/results.json after every k point and log lines on stdout (PROGRESS i/n, ERROR ..., CANCELLED, DONE). Cancellation: the file
<folder>/cancel, checked between the k points.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_bands as fb  # noqa: E402
import fem_wg_worker as ww  # noqa: E402

TWO_PI = 2 * np.pi


def say(*a):
    print(*a, flush=True)


def build_problem(h, job, mesh):
    """(nd, h1, BandStructure2D) for the job: one PeriodicPair2D per lattice vector, one extra band for the spurious zero at Gamma."""
    p = int(job["mesh"]["order"])
    nd, h1 = h.NedelecDofMap2D(mesh, p), h.DofMap2D(mesh, p)
    a1, a2 = fb.lattice_vectors(job)
    setup = h.BandStructureSetup2D()
    setup.lattice = [h.PeriodicPair2D(fb.TAG_LEFT, fb.TAG_RIGHT, [float(a1[0]), float(a1[1])]), h.PeriodicPair2D(fb.TAG_BOTTOM, fb.TAG_TOP, [float(a2[0]), float(a2[1])])]
    setup.num_bands = int(job["bands"]) + 1
    setup.krylov_dimension = int(job["krylov"]) if int(job.get("krylov", 0)) > 0 else 40 + 4 * (int(job["bands"]) + 1)   # automatic (0) stalls at Gamma
    setup.keep_modes = job.get("field", "none") != "none"
    for i, name in enumerate(job["materials"]):
        setup.materials.set(1 + i, h.Material(complex(fb.eps_of(job["materials"][name]).real)))
    return nd, h1, h.BandStructure2D(nd, h1, setup)


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    mesh = hpfem.read_gmsh(str(folder / "mesh.msh"), 1.0, 2)
    say(f"hpfem {getattr(hpfem, '__version__', '?')}, mesh: {mesh.num_cells} cells")
    nd, h1, problem = build_problem(hpfem, job, mesh)
    ks = np.asarray(job["k"], dtype=float)
    n_bands = int(job["bands"])
    want_field = set(job["corner_idx"]) if job.get("field", "none") == "corners" else set()
    out = dict(meta=dict(name=job.get("name", ""), lattice=job["lattice"], started=time.strftime("%Y-%m-%d %H:%M:%S"), hpfem=str(getattr(hpfem, "__version__", "?")),
                         cells=int(mesh.num_cells), order=int(job["mesh"]["order"]), dofs=int(nd.num_dofs), bands=n_bands), points=[], x=job["x"], ticks=job["ticks"],
               corner_idx=job["corner_idx"])
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    for i, kf in enumerate(ks):
        if (folder / "cancel").exists():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return 0
        say(f"PROGRESS {i + 1}/{len(ks)} k = ({kf[0]:.4f}, {kf[1]:.4f}) 2pi/a")
        t0 = time.time()
        try:
            bands = problem.bands(np.array([TWO_PI * kf[0], TWO_PI * kf[1]]))
            f = fb.clean_bands(bands.normalised(1.0), n_bands, bool(np.linalg.norm(kf) < 1e-12))
            res = dict(index=i, k=[float(kf[0]), float(kf[1])], f=f, residual=float(max(bands.residual)), time_s=time.time() - t0)
            if i in want_field:
                modes = np.asarray(bands.modes)
                gamma = np.linalg.norm(kf) < 1e-12
                cols = list(range(modes.shape[1]))
                arrays = {}
                order = np.argsort(bands.wavenumber)
                if gamma and len(order) > 1 and sorted(bands.normalised(1.0))[1] < 1e-3:
                    order = np.delete(order, 1)
                for b, col in enumerate(order[:n_bands]):
                    pts, simp, e, _ = ww.triangulation(hpfem, mesh, nd, modes[:, cols[int(col)]])
                    arrays.update({"points": pts, "simplices": simp, f"et_{b}": e})
                np.savez_compressed(folder / f"mode_{i}.npz", **arrays)
                res["modes_file"] = f"mode_{i}.npz"
            out["points"].append(res)
            say(f"  {res['time_s']:.1f} s: f = " + ", ".join(f"{v:.4f}" for v in f))
        except Exception as exc:
            say(f"ERROR point {i}: {exc}")
            out["points"].append(dict(index=i, k=[float(kf[0]), float(kf[1])], error=str(exc)))
        flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
