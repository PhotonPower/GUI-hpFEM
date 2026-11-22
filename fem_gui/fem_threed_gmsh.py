"""Gmsh meshes for the 3D modes of the FEM model builder (hp-FEM 0.5). Needs the Python module ``gmsh`` and NumPy, not hpfem, so it can be tested without the library
and runs inside the worker (fem_threed_worker.py, fem_ports3d_worker.py). Lengths are in nm; the worker reads the file with ``hpfem.read_gmsh(path, scale=1e-9, dim=3)``.

``scene(g, path)``: one object (sphere, cylinder with axis z, block) in a box [-hw, hw]^2 x [z_lo - pml, z_hi + pml], optionally above a planar layer stack
(interfaces z = 0, -d1, -d1 - d2, ...; the last region is the substrate). Every region of the stack is its own volume, so the interfaces are mesh facets (a
requirement of the scattered-field formulation with a layered background). Physical groups: volumes TAG_OBJ (object) and TAG_REGION0 + k (region k from the top:
0 = incidence medium), surfaces 1..6 = the box sides X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX (the numbering of ``hpfem.box_tag``). Second-order (curved) elements by
default. The straight guide and the coupler of the task "ports" use structured meshes (fem_ports3d_worker.py), not Gmsh.
"""
from __future__ import annotations

TAG_OBJ = 2
TAG_REGION0 = 11
KINDS = {"sphere": "Kugel", "cylinder": "Zylinder (Achse z)", "block": "Quader"}


def object_extent(g):
    """(half width in x, half width in y, height) of the object [nm]."""
    k = g["kind"]
    if k == "sphere":
        return g["r"], g["r"], 2 * g["r"]
    if k == "cylinder":
        return g["r"], g["r"], g["h"]
    return g["lx"] / 2, g["ly"] / 2, g["lz"]


def layout(g):
    """Vertical layout [nm]: dict with z_lo, z_hi (interior of the PML box), interfaces (top to bottom), obj_z0 (lowest z of the object), zc (centre of the object),
    regions = [(zmin, zmax)] from the top (region 0 reaches z_hi)."""
    ox, oy, oh = object_extent(g)
    layers = [float(t) for t in g.get("layers", [])]
    stack = bool(g.get("stack", False))
    if stack:
        gap = float(g.get("gap", 0.0))
        z0 = gap
        ifaces = [0.0]
        for t in layers:
            ifaces.append(ifaces[-1] - t)
        z_lo = ifaces[-1] - float(g.get("depth", 0.0))
        z_hi = z0 + oh + float(g.get("clear", 0.0))
        regions = [(ifaces[0], z_hi)] + [(ifaces[i + 1], ifaces[i]) for i in range(len(ifaces) - 1)] + [(z_lo, ifaces[-1])]
        zc = z0 + oh / 2
    else:
        z0 = -oh / 2
        ifaces = []
        z_lo, z_hi = -oh / 2 - float(g.get("clear", 0.0)), oh / 2 + float(g.get("clear", 0.0))
        regions = [(z_lo, z_hi)]
        zc = 0.0
    return dict(z_lo=z_lo, z_hi=z_hi, interfaces=ifaces, obj_z0=z0, zc=zc, regions=regions)


def scene(g, path):
    """Writes the Gmsh 4.1 ASCII mesh to `path` and returns info: tags, interior box, number of regions."""
    import gmsh

    lay = layout(g)
    hw, pml = float(g["hw"]), float(g["pml"])
    X = hw + pml
    zmin, zmax = lay["z_lo"] - pml, lay["z_hi"] + pml
    cuts = sorted([z for z in lay["interfaces"]] + [lay["z_lo"], lay["z_hi"]]) if lay["interfaces"] else [lay["z_lo"], lay["z_hi"]]
    cuts = sorted({round(z, 9) for z in cuts})
    edges = [zmin] + cuts + [zmax]
    edges = sorted(set(round(e, 9) for e in edges))
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("scene3d")
        occ = gmsh.model.occ
        slabs = [occ.addBox(-X, -X, a, 2 * X, 2 * X, b - a) for a, b in zip(edges[:-1], edges[1:])]
        k = g["kind"]
        zc = lay["zc"]
        if k == "sphere":
            obj = occ.addSphere(0, 0, zc, g["r"])
        elif k == "cylinder":
            obj = occ.addCylinder(0, 0, lay["obj_z0"], 0, 0, g["h"], g["r"])
        else:
            obj = occ.addBox(-g["lx"] / 2, -g["ly"] / 2, lay["obj_z0"], g["lx"], g["ly"], g["lz"])
        _, parts = occ.fragment([(3, s) for s in slabs], [(3, obj)])
        occ.synchronize()
        obj_vols = {t for _, t in parts[-1]}
        # region of every remaining volume by its z centre (the slabs are cut at the interfaces, so every piece lies in exactly one region)
        regions = lay["regions"]
        by_region = {i: [] for i in range(len(regions))}
        for part in parts[:-1]:
            for _, t in part:
                if t in obj_vols:
                    continue
                bb = gmsh.model.getBoundingBox(3, t)
                zmid = 0.5 * (bb[2] + bb[5])
                idx = _region_of(regions, zmid, edges, lay)
                if t not in by_region[idx]:
                    by_region[idx].append(t)
        for i, vols in by_region.items():
            if vols:
                gmsh.model.addPhysicalGroup(3, vols, TAG_REGION0 + i, f"region{i}")
        gmsh.model.addPhysicalGroup(3, sorted(obj_vols), TAG_OBJ, "object")
        all_vols = sorted(obj_vols | {t for vols in by_region.values() for t in vols})
        tol = 1e-6 * max(hw, 1.0)
        sides = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        for dim, tag in gmsh.model.getBoundary([(3, t) for t in all_vols], oriented=False):
            x0, y0, z0_, x1, y1, z1_ = gmsh.model.getBoundingBox(dim, tag)
            if x1 < -X + tol:
                sides[1].append(tag)
            elif x0 > X - tol:
                sides[2].append(tag)
            elif y1 < -X + tol:
                sides[3].append(tag)
            elif y0 > X - tol:
                sides[4].append(tag)
            elif z1_ < zmin + tol:
                sides[5].append(tag)
            elif z0_ > zmax - tol:
                sides[6].append(tag)
        for t, faces in sides.items():
            if faces:
                gmsh.model.addPhysicalGroup(2, sorted(set(faces)), t, f"side{t}")
        _sizing(g, lay, regions, all_vols, obj_vols, hw, X, edges)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        order = int(g.get("order", 2))
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        if order > 1:
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return dict(regions=[TAG_REGION0 + i for i, v in by_region.items() if v], object_tag=TAG_OBJ, interior=[-hw, hw, lay["z_lo"], lay["z_hi"]], layout=lay,
                outer=[X, zmin, zmax])


def _region_of(regions, zmid, edges, lay):
    """Index of the region [zmin, zmax] containing zmid (regions are listed from the top; region 0 = above the first interface)."""
    for i, (a, b) in enumerate(regions):
        if a - 1e-9 <= zmid <= b + 1e-9:
            return i
    # PML slabs above / below the interior: the first / last region
    return 0 if zmid > lay["z_hi"] else len(regions) - 1


def _sizing(g, lay, regions, all_vols, obj_vols, hw, X, edges):
    import gmsh

    field = gmsh.model.mesh.field
    h_obj, h_far = float(g["h_obj"]), float(g["h_far"])
    grade = float(g.get("grade", 3 * h_far))
    faces = [t for _, t in gmsh.model.getBoundary([(3, t) for t in sorted(obj_vols)], oriented=False)]
    dist = field.add("Distance")
    field.setNumbers(dist, "SurfacesList", faces)
    field.setNumber(dist, "Sampling", 40)
    thr = field.add("Threshold")
    field.setNumber(thr, "InField", dist)
    field.setNumber(thr, "SizeMin", h_obj)
    field.setNumber(thr, "SizeMax", h_far)
    field.setNumber(thr, "DistMin", 0.3 * h_obj)
    field.setNumber(thr, "DistMax", grade)
    fields = [thr]
    ns = g.get("region_n", [])
    for i, (a, b) in enumerate(regions):
        n = float(ns[i]) if i < len(ns) else 1.0
        size = h_far / max(n, 1.0)
        if i == len(regions) - 1 or i == 0:
            a_ext = a - float(g["pml"]) if i == len(regions) - 1 else a
            b_ext = b + float(g["pml"]) if i == 0 else b
        else:
            a_ext, b_ext = a, b
        if len(regions) > 2 and 0 < i < len(regions) - 1:
            size = min(size, 0.5 * (b - a))                     # a finite layer carries at least two cells
        box = field.add("Box")
        field.setNumber(box, "VIn", size)
        field.setNumber(box, "VOut", 10 * h_far)
        field.setNumber(box, "XMin", -X)
        field.setNumber(box, "XMax", X)
        field.setNumber(box, "YMin", -X)
        field.setNumber(box, "YMax", X)
        field.setNumber(box, "ZMin", a_ext)
        field.setNumber(box, "ZMax", b_ext)
        field.setNumber(box, "Thickness", 0.0)
        fields.append(box)
    mn = field.add("Min")
    field.setNumbers(mn, "FieldsList", fields)
    field.setAsBackgroundMesh(mn)
