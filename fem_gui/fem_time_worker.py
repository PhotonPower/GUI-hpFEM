"""Worker for the time domain (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_time_worker.py <job folder>

Reads <folder>/job.json (fem_time.job). 2D: <folder>/mesh.msh (fem_time.build_mesh; nm, tags 1 + index of the material for the surfaces, 101 ...
104 for the sides); 3D: the mesh is hpfem.box_with_ball (SI, background tag 0, sphere tag 2, faces 1 ... 6). Solver: hpfem.TimeDomain2D / 3D
(Newmark-beta, lossless materials, conductivity by tag, PEC / absorbing facets, current density with a TimeSignal).

Per step: the energy, the probe fields and g'(t); every ``stride`` steps a field frame (2D: on the subdivided mesh; 3D: on a regular grid in the plane
z = frame_z). Writes <folder>/results.json (meta, t_fs, energy, drive, probes) and <folder>/field_frames.npz (frames) after every chunk and at the end;
log lines on stdout (PROGRESS i/n over the chunks of steps, ERROR ..., CANCELLED, DONE). Cancellation: the file <folder>/cancel, checked between chunks.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_time as ft  # noqa: E402

NM = 1e-9
SUB = 2                                                              # subdivisions per cell for the 2D field maps
CHUNKS = 40


def say(*a):
    print(*a, flush=True)


def build_signal(h, src):
    """hpfem.TimeSignal of the job: the library pulses for the modulated / Gaussian kinds, the logistic continuous wave through the Python callbacks."""
    if src["kind"] == "modulated":
        return h.modulated_gaussian(src["omega"], src["t0"], src["tau"])
    if src["kind"] == "gaussian":
        return h.gaussian_pulse(src["t0"], src["tau"])
    return h.TimeSignal(lambda t: float(ft.signal_value("cw", src["omega"], src["t0"], src["tau"], t)),
                        lambda t: float(ft.signal_derivative("cw", src["omega"], src["t0"], src["tau"], t)))


def build_current(src, dim):
    pos, size = np.asarray(src["pos"], float), float(src["size"])
    pol = np.asarray(src["pol"], dtype=complex) * float(src["amplitude"])
    s2 = 2.0 * size * size

    def current(x):
        d = np.asarray(x, dtype=float) - pos
        return pol * np.exp(-float(d @ d) / s2)

    return current


def build_problem(h, job, mesh):
    """(nd, problem) of the job."""
    dim = job["dim"]
    nd = (h.NedelecDofMap2D if dim == "2d" else h.NedelecDofMap3D)(mesh, int(job["order"]))
    if dim == "3d" and nd.num_dofs > int(job["max_dofs"]):
        raise RuntimeError(f"{nd.num_dofs} Freiheitsgrade: für 3D über der Grenze von {job['max_dofs']} (Faktorisierung zu teuer). Ordnung oder Netz verkleinern.")
    setup = (h.TimeDomainSetup2D if dim == "2d" else h.TimeDomainSetup3D)()
    setup.dt = float(job["dt"])
    setup.gamma, setup.beta = float(job["gamma"]), float(job["beta"])
    setup.pec_tags = [int(t) for t in job["pec_tags"]]
    setup.absorbing_tags = [int(t) for t in job["absorbing_tags"]]
    bg = job["materials"][job["background"]]
    mm = h.MaterialMap(h.Material(complex(bg["eps"])))
    sigma = {}
    if dim == "2d":
        for name, tag in job["material_tags"].items():
            mm.set(int(tag), h.Material(complex(job["materials"][name]["eps"])))
            if job["materials"][name]["sigma"] > 0:
                sigma[int(tag)] = float(job["materials"][name]["sigma"])
    else:
        sph = job["materials"][job["sphere"]["material"]]
        mm.set(2, h.Material(complex(sph["eps"])))
        if bg["sigma"] > 0:
            sigma[0] = float(bg["sigma"])
        if sph["sigma"] > 0:
            sigma[2] = float(sph["sigma"])
    setup.materials = mm
    if sigma:
        setup.conductivity = sigma
    setup.signal = build_signal(h, job["source"])
    setup.current = build_current(job["source"], dim)
    return nd, (h.TimeDomain2D if dim == "2d" else h.TimeDomain3D)(nd, setup)


def make_mesh(h, job, folder):
    if job["dim"] == "2d":
        return h.read_gmsh(str(folder / "mesh.msh"), NM, 2)
    sp = job["sphere"]
    n, outer, r = int(sp["cells"]), float(sp["outer"]), float(sp["radius"])
    return h.box_with_ball(n, r, float(sp["half"]), outer)


class Probes:
    """Located probe points: field values (n_probes, dim) of a coefficient vector."""

    def __init__(self, h, mesh, nd, points, dim):
        loc = (h.PointLocator2D if dim == "2d" else h.PointLocator3D)(mesh)
        self.h, self.nd, self.cells, self.xi = h, nd, [], []
        for i, p in enumerate(points):
            lp = loc.locate(np.asarray(p, dtype=float))
            if lp is None:
                raise RuntimeError(f"Sonde {i + 1} liegt nicht im Netz.")
            self.cells.append(int(lp.cell))
            self.xi.append(np.asarray(lp.xi, dtype=float))
        self.xi = np.array(self.xi, dtype=float).reshape(len(points), dim == "3d" and 3 or 2)

    def values(self, u):
        if not self.cells:
            return np.zeros((0, self.xi.shape[1]))
        return np.asarray(self.h.sample_hcurl(self.nd, np.asarray(u, dtype=complex), self.cells, self.xi)).real


class Frames:
    """Field frames on the subdivided mesh (2D) or a regular grid in the plane z = frame_z (3D)."""

    def __init__(self, h, job, mesh, nd):
        self.h, self.nd, self.dim = h, nd, job["dim"]
        if self.dim == "2d":
            sub = h.subdivide(mesh, SUB)
            self.points = np.asarray(sub.mesh.vertices)[:, :2] / NM
            self.simplices = np.asarray(sub.mesh.cells, dtype=int)
            self.cells = list(map(int, sub.vertex_parent))
            self.xi = np.asarray(sub.vertex_xi, dtype=float)
            self.shape = (len(self.cells),)
            self.mask = None
        else:
            n, R = int(job["grid"]), float(job["sphere"]["outer"])
            xs = np.linspace(-R, R, n) * 0.999
            gx, gy = np.meshgrid(xs, xs)
            loc = h.PointLocator3D(mesh)
            cells, xis, idx = [], [], []
            for k, (x, y) in enumerate(zip(gx.ravel(), gy.ravel())):
                lp = loc.locate(np.array([x, y, float(job["frame_z"])]))
                if lp is not None:
                    cells.append(int(lp.cell))
                    xis.append(np.asarray(lp.xi, dtype=float))
                    idx.append(k)
            self.cells, self.xi, self.idx = cells, np.array(xis, dtype=float).reshape(len(cells), 3), np.array(idx, dtype=int)
            self.gx, self.gy = gx / NM, gy / NM
            self.shape = gx.shape
        self.times, self.data = [], {"ex": [], "ey": []}
        if self.dim == "3d":
            self.data["ez"] = []

    def add(self, t, u):
        vals = np.asarray(self.h.sample_hcurl(self.nd, np.asarray(u, dtype=complex), self.cells, self.xi)).real
        self.times.append(t / 1e-15)
        for c, name in enumerate(self.data):
            if self.dim == "2d":
                self.data[name].append(vals[:, c].astype(np.float32))
            else:
                full = np.full(self.shape[0] * self.shape[1], np.nan, dtype=np.float32)
                full[self.idx] = vals[:, c]
                self.data[name].append(full.reshape(self.shape))

    def save(self, path):
        arrays = {"times_fs": np.array(self.times)}
        arrays.update({k: np.array(v) for k, v in self.data.items()})
        if self.dim == "2d":
            arrays.update(points=self.points, simplices=self.simplices)
        else:
            arrays.update(gx=self.gx, gy=self.gy)
        np.savez_compressed(path, **arrays)


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    dim, steps = job["dim"], int(job["steps"])
    t_setup = time.time()
    mesh = make_mesh(hpfem, job, folder)
    say(f"hpfem {getattr(hpfem, '__version__', '?')}, mesh: {mesh.num_cells} cells")
    try:
        nd, problem = build_problem(hpfem, job, mesh)
        probes = Probes(hpfem, mesh, nd, job["probes"], dim)
        frames = Frames(hpfem, job, mesh, nd) if int(job["frames"]) > 0 else None
    except Exception as exc:
        say(f"ERROR {exc}")
        return 1
    say(f"{nd.num_dofs} DoFs ({problem.num_free_dofs} frei), Aufbau und Faktorisierung {time.time() - t_setup:.1f} s, {steps} Schritte mit dt = {job['dt'] / 1e-15:.4g} fs")
    comps = ("ex", "ey") if dim == "2d" else ("ex", "ey", "ez")
    src = job["source"]
    out = dict(meta=dict(name=job.get("name", ""), dim=dim, started=time.strftime("%Y-%m-%d %H:%M:%S"), hpfem=str(getattr(hpfem, "__version__", "?")),
                         cells=int(mesh.num_cells), dofs=int(nd.num_dofs), order=int(job["order"]), dt_fs=job["dt"] / 1e-15, steps=steps, estimate=job["estimate"],
                         frame_z_nm=job.get("frame_z", 0.0) / NM, gamma=job["gamma"], beta=job["beta"], source=src["kind"]),
               t_fs=[0.0], t_energy_fs=[0.0], energy=[0.0], drive=[float(ft.signal_derivative(src["kind"], src["omega"], src["t0"], src["tau"], 0.0))],
               probes=[{c: [0.0] for c in comps} | {"pos_nm": [v / NM for v in p]} for p in job["probes"]])
    state = problem.initialize()
    if frames:
        frames.add(0.0, state.u)
    stride = max(1, steps // max(int(job["frames"]), 1))
    counter = {"k": 0}
    e_stride = max(1, steps // 600)

    def observe(s):
        counter["k"] += 1
        t = float(s.time)
        out["t_fs"].append(t / 1e-15)
        if counter["k"] % e_stride == 0 or counter["k"] == steps:                  # the energy costs as much as a step for large meshes
            out["t_energy_fs"].append(t / 1e-15)
            out["energy"].append(float(problem.energy(s)))
        out["drive"].append(float(ft.signal_derivative(src["kind"], src["omega"], src["t0"], src["tau"], t)))
        v = probes.values(s.u)
        for i, p in enumerate(out["probes"]):
            for c, name in enumerate(comps):
                p[name].append(float(v[i, c]))
        if frames and (counter["k"] % stride == 0 or counter["k"] == steps):
            frames.add(t, s.u)

    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")
        if frames:
            frames.save(folder / "field_frames.npz")

    chunk = max(1, int(np.ceil(steps / CHUNKS)))
    done, n_chunks = 0, int(np.ceil(steps / chunk))
    t_run = time.time()
    try:
        while done < steps:
            if (folder / "cancel").exists():
                say("CANCELLED")
                out["cancelled"] = True
                flush()
                return 0
            n = min(chunk, steps - done)
            say(f"PROGRESS {done // chunk + 1}/{n_chunks} t = {state.time / 1e-15:.2f} fs")
            problem.run(state, n, observe)
            done += n
            flush()
    except Exception as exc:
        say(f"ERROR {exc}")
        out["error"] = str(exc)
        flush()
        return 1
    out["meta"]["run_s"] = time.time() - t_run
    flush()
    e = np.asarray(out["energy"])
    say(f"max. Energie {e.max():.3e} J, Endenergie {e[-1]:.3e} J, {out['meta']['run_s']:.1f} s")
    say("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
