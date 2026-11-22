"""Gmsh mesh of the 3D add-drop ring resonator (task "Wellenleiter und Ports", device "ring"; hp-FEM 0.5). Port of examples/ring_resonator_3d/ring_mesh.py of the library.
Needs only the Python module ``gmsh``, not hpfem, so it can be tested without the library and runs inside the worker (fem_ports3d_worker.py). Lengths in nm.

Geometry (x along the buses, y across, z up): a ring of centre-line radius R, width w and height h at the origin and two straight buses of the same cross-section at
y = -/+ (R + w + gap), all in cladding between the PEC walls of a box; second-order (curved) tetrahedra, graded from ``cell_core`` within the cores to ``cell_far`` at
``grade`` from them. Physical groups: volumes 1 cladding, 2 core material; surfaces 10 walls, 11 the face x = -X, 12 the face x = +X (the worker splits each face at y = 0
into the ports of the two buses), 15 the mid-plane z = h/2 of the half model (``half``: only z >= h/2 is meshed; the cladding is the same above and below, so the TE-like
fields are even about the mid-plane of the cores and the plane is a magnetic wall; the S-parameters are those of the full device).
"""
from __future__ import annotations

TAG_CLAD, TAG_CORE = 1, 2
TAG_WALL, TAG_LEFT, TAG_RIGHT, TAG_SYMMETRY = 10, 11, 12, 15
PORT_TAGS = (TAG_LEFT, TAG_RIGHT, 13, 14)       # 1 input (bus A, x = -X), 2 through (bus A, +X), 3 drop (bus B, -X), 4 add (bus B, +X)


def geometry(**kw):
    """Mesh parameters [nm]; the defaults are those of the library example (production mesh)."""
    g = dict(radius=2000.0, width=450.0, height=220.0, gap=200.0, margin_x=1000.0, margin_y=750.0, margin_z=605.0, cell_core=100.0, cell_far=300.0, grade=600.0, half=False)
    g.update(kw)
    return g


def extent(g):
    """(bus_y, half_x, half_y): the centre lines of the buses lie at y = -/+ bus_y, the box is [-half_x, half_x] x [-half_y, half_y] [nm]."""
    bus_y = g["radius"] + g["width"] + g["gap"]
    return bus_y, g["radius"] + g["width"] / 2 + g["margin_x"], bus_y + g["width"] / 2 + g["margin_y"]


def core_distance(g, points_nm):
    """Distance [nm] of points (n, 3) from the nearest core (ring or bus), from the exact shapes."""
    import numpy as np

    x = np.asarray(points_nm, dtype=float)
    bus_y = extent(g)[0]
    d_bus = np.maximum(np.abs(np.abs(x[:, 1]) - bus_y) - g["width"] / 2, 0.0)
    d_ring = np.maximum(np.abs(np.hypot(x[:, 0], x[:, 1]) - g["radius"]) - g["width"] / 2, 0.0)
    d_z = np.maximum(np.maximum(-x[:, 2], x[:, 2] - g["height"]), 0.0)
    return np.hypot(np.minimum(d_bus, d_ring), d_z)


def build(g, path):
    """Writes the Gmsh 4.1 ASCII mesh to `path` and returns it."""
    import gmsh

    bus_y, X, Y = extent(g)
    w, hgt = g["width"], g["height"]
    z0, z1 = -g["margin_z"], hgt + g["margin_z"]
    zc = hgt / 2 if g["half"] else 0.0               # bottom of the cores in the model
    if g["half"]:
        z0 = zc
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ring_resonator_3d")
        occ = gmsh.model.occ
        box = occ.addBox(-X, -Y, z0, 2 * X, 2 * Y, z1 - z0)
        hc = hgt - zc
        outer = occ.addCylinder(0, 0, zc, 0, 0, hc, g["radius"] + w / 2)
        inner = occ.addCylinder(0, 0, zc, 0, 0, hc, g["radius"] - w / 2)
        ring, _ = occ.cut([(3, outer)], [(3, inner)])
        buses = [occ.addBox(-X, s * bus_y - w / 2, zc, 2 * X, w, hc) for s in (-1, 1)]
        cores = ring + [(3, b) for b in buses]
        _, parts = occ.fragment([(3, box)], cores)
        occ.synchronize()
        silicon = sorted({t for part in parts[1:] for _, t in part})
        cladding = sorted({t for _, t in parts[0]} - set(silicon))
        if len(silicon) != 3 or len(cladding) != 1:
            raise RuntimeError(f"ring mesh: {len(silicon)} cores and {len(cladding)} claddings instead of 3 and 1 (check gap and margins)")
        gmsh.model.addPhysicalGroup(3, cladding, TAG_CLAD, "cladding")
        gmsh.model.addPhysicalGroup(3, silicon, TAG_CORE, "core")
        tol = 1e-3 * g["cell_core"]
        left, right, walls, symmetry = [], [], [], []
        for dim, tag in gmsh.model.getBoundary([(3, t) for t in cladding + silicon], oriented=False):
            x0, y0, zz0, x1, y1, zz1 = gmsh.model.getBoundingBox(dim, tag)
            if x1 < -X + tol:
                left.append(tag)
            elif x0 > X - tol:
                right.append(tag)
            elif g["half"] and zz1 < z0 + tol:
                symmetry.append(tag)
            elif y1 < -Y + tol or y0 > Y - tol or zz1 < z0 + tol or zz0 > z1 - tol:
                walls.append(tag)
        gmsh.model.addPhysicalGroup(2, walls, TAG_WALL, "wall")
        gmsh.model.addPhysicalGroup(2, left, TAG_LEFT, "port_left")
        gmsh.model.addPhysicalGroup(2, right, TAG_RIGHT, "port_right")
        if symmetry:
            gmsh.model.addPhysicalGroup(2, symmetry, TAG_SYMMETRY, "symmetry")
        core_faces = [t for _, t in gmsh.model.getBoundary([(3, t) for t in silicon], oriented=False)]
        field = gmsh.model.mesh.field
        dist = field.add("Distance")
        field.setNumbers(dist, "SurfacesList", core_faces)
        field.setNumber(dist, "Sampling", 60)
        thr = field.add("Threshold")
        field.setNumber(thr, "InField", dist)
        field.setNumber(thr, "SizeMin", g["cell_core"])
        field.setNumber(thr, "SizeMax", g["cell_far"])
        field.setNumber(thr, "DistMin", 0.5 * g["cell_core"])
        field.setNumber(thr, "DistMax", g["grade"])
        field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # HXT
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(path)
    finally:
        gmsh.finalize()
    return path
