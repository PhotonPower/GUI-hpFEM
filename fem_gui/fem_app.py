"""FEM-Modellwerkstatt: periodische Struktur eingeben, mit Gmsh vernetzen, mit hp-FEM rechnen, Ergebnisse als Schaubilder und Karten ansehen.

Start (im Ordner dieser Dateien):   streamlit run fem_app.py
Benötigt für die App:               streamlit numpy pandas matplotlib gmsh       (pip install ...)
Benötigt für die Rechnung:          ein Python mit gebautem hpfem (z. B. MSYS2), siehe Seitenleiste
"""
from __future__ import annotations

import copy
import io
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_geometry as fg  # noqa: E402
import fem_materials as fm  # noqa: E402
import fem_post as fp  # noqa: E402
import fem_run as fr  # noqa: E402

HERE = Path(__file__).resolve().parent
S = st.session_state
st.set_page_config(page_title="FEM-Modellwerkstatt", layout="wide")

# ---- all files of the app must come from the same version: name what is missing instead of a traceback
_REQUIRED = {
    "fem_materials.py": (fm, ["library_available", "eps_at", "nk", "describe", "valid_range", "LIBRARY"]),
    "fem_geometry.py": (fg, ["presets", "validate", "layout", "preview_figure", "mesh_figure", "build_mesh", "suggest_domain", "pml_plan", "shift_shapes",
                             "centering_shift", "shapes_x_extent", "material_index_map", "load_model", "new_shape", "parse_points", "shape_polygon", "reader_check"]),
    "fem_post.py": (fp, ["load_results", "results_dataframe", "fig_spectrum", "fig_pscan", "load_map", "derived", "fig_map", "fig_cut", "fig_absorption",
                         "fig_absorption_sweep", "fig_adaptive_steps", "fig_hpmesh", "load_tri", "derived_tri", "fig_map_tri", "fig_materials"]),
    "fem_run.py": (fr, ["sweep_values", "wavelength_range", "make_job", "write_job", "start_worker", "read_log", "progress_info", "log_flags", "library_warnings",
                        "default_python"]),
}
_old = [f"{name}: fehlt {', '.join(miss)}" for name, (mod, names) in _REQUIRED.items() if (miss := [n for n in names if not hasattr(mod, n)])]
_worker = HERE / "fem_worker.py"
if _worker.is_file():
    _wtext = _worker.read_text(encoding="utf-8", errors="replace")
    _wmiss = [k for k in ("absorbed_exact", "triangulation_data", "set_periodic", "symmetrise_periodic", "design_pml") if k not in _wtext]
    if _wmiss:
        _old.append(f"fem_worker.py: veraltet (es fehlt {', '.join(_wmiss)})")
else:
    _old.append("fem_worker.py: nicht gefunden (muss neben fem_app.py liegen)")
if _old:
    st.error("Die Dateien der App stammen aus verschiedenen Versionen. Ersetze diese Dateien durch die neuesten und starte die App neu:\n\n" + "\n".join(f"- {x}" for x in _old))
    st.caption(f"Ordner: {HERE}. Zusammengehörig sind fem_app.py, fem_geometry.py, fem_materials.py, fem_post.py, fem_run.py und fem_worker.py.")
    st.stop()

TYPE_LABELS = {"library": "Bibliothek (hpfem)", "index": "n + ik (konstant)", "eps": "ε (konstant)", "drude": "Drude-Metall", "table": "Tabelle λ, n, k"}
LIB_CHOICES = ["Si", "Ag", "Au", "Al", "GaAs", "MAPbI3", "SiO2", "TiO2", "water", "air", "vacuum"]
DEFAULT_SPEC = {"library": {"type": "library", "name": "SiO2"}, "index": {"type": "index", "n": 1.5, "k": 0.0},
                "eps": {"type": "eps", "re": 2.25, "im": 0.0}, "drude": {"type": "drude", "eps_inf": 1.0, "omega_p_eV": 9.0, "gamma_eV": 0.07},
                "table": {"type": "table", "rows": [[400.0, 1.5, 0.0], [800.0, 1.5, 0.0]]}}
ENGINE_SHORT = {"conical": "konischer Löser", "inplane": "In-Ebenen-Löser", "inplane_hp": "In-Ebenen-Löser, hp-adaptiv"}
SWEEP_MODES = {"none": "kein Durchlauf (ein Punkt)", "wavelength": "Wellenlänge", "theta": "Einfallswinkel θ", "phi": "Azimut φ (konisch)"}


# ------------------------------------------------------------------------------------------------------------------- helpers
def table_show(obj):
    try:
        st.dataframe(obj, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(obj, use_container_width=True, hide_index=True)


def show_fig(fig, name, key):
    import matplotlib.pyplot as plt

    st.pyplot(fig)
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    st.download_button("Bild herunterladen (PNG)", b.getvalue(), file_name=name, mime="image/png", key=key)
    plt.close(fig)


def csv_bytes(df):
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def next_uid():
    S.uid += 1
    return S.uid


def ensure_uids(model):
    for item in model["layers"] + model["shapes"]:
        if "uid" not in item:
            item["uid"] = next_uid()


def clean(model):
    """Model without the widget ids (for signatures and files)."""
    m = copy.deepcopy(model)
    for item in m["layers"] + m["shapes"]:
        item.pop("uid", None)
    return m


def set_model(m):
    S.model = copy.deepcopy(m)
    ensure_uids(S.model)
    S.ver += 1
    S.pop("view_dir", None)


def init_state():
    if "model" in S:
        return
    S.uid, S.ver = 1000, 0
    S.model = copy.deepcopy(next(iter(fg.presets().values())))
    ensure_uids(S.model)
    S.mesh_settings = dict(cells_per_wavelength=3.0, skin_cells=1.0, interface_factor=0.7, curved=True, pml_cells_per_wavelength=6.0)
    S.solver = dict(order=4, backend="AUTO", pml_target=1e-3, pml_angle_cap_deg=80.0, orders_max=3, order_points=64, engine="conical",
                    adaptive=dict(p0=3, steps=12, max_dofs=150000, dorfler=0.5, tol=1e-4, reuse_mesh=True))
    S.maps = dict(enabled=True, indices=[0], res_nm=4.0, triangulate=True, subdiv=3)
    S.mesh_sig, S.mesh_stats, S.run_proc, S.run_msg = None, None, None, ""


def p_eff():
    """Polynomial order that sets the size of the problem: the start order of an hp-adaptive run, else p."""
    sv = S.solver
    return int(sv["adaptive"]["p0"]) if sv.get("engine") == "inplane_hp" else int(sv["order"])


def geom_signature(model, ms):
    m = clean(model)
    keep = {k: m[k] for k in ("period_nm", "materials", "cover", "layers", "substrate", "shapes", "domain")}
    return json.dumps([keep, ms, fr.wavelength_range(model)], sort_keys=True, default=str)


def est_dofs(cells, p):
    return int(1.4 * 0.9 * (p + 1) * (p + 2) * cells)


def used_materials(model):
    u = {model["cover"], model["substrate"]}
    u |= {l["material"] for l in model["layers"]}
    u |= {s["material"] for s in model["shapes"]}
    return u


def rename_material(model, old, new):
    model["materials"] = {(new if k == old else k): v for k, v in model["materials"].items()}
    if model["cover"] == old:
        model["cover"] = new
    if model["substrate"] == old:
        model["substrate"] = new
    for item in model["layers"] + model["shapes"]:
        if item["material"] == old:
            item["material"] = new


def move(listname, i, delta):
    lst = S.model[listname]
    j = i + delta
    if 0 <= j < len(lst):
        lst[i], lst[j] = lst[j], lst[i]


def delete_item(listname, i):
    del S.model[listname][i]


def duplicate_shape(i):
    c = copy.deepcopy(S.model["shapes"][i])
    c["uid"] = next_uid()
    S.model["shapes"].insert(i + 1, c)


def shift_cb(key):
    dx = float(S[key])
    if dx:
        fg.shift_shapes(S.model, dx)
        S.ver += 1


def center_cb():
    dx = fg.centering_shift(S.model)
    if abs(dx) > 1e-12:
        fg.shift_shapes(S.model, dx)
        S.ver += 1


def add_shape(kind):
    names = list(S.model["materials"])
    default = next((n for n in names if n != S.model["cover"]), names[0])
    s = fg.new_shape(kind, default)
    s["uid"] = next_uid()
    S.model["shapes"].append(s)


def add_layer():
    names = list(S.model["materials"])
    S.model["layers"].append(dict(material=next((n for n in names if n != S.model["cover"]), names[0]), thickness_nm=100.0, uid=next_uid()))


def unique_name(model, base):
    n, k = base, 2
    while n in model["materials"]:
        n, k = f"{base} {k}", k + 1
    return n


# --------------------------------------------------------------------------------------------------------------------- state
init_state()
model = S.model
ver = S.ver

# ------------------------------------------------------------------------------------------------------------------ sidebar
st.sidebar.title("FEM-Modellwerkstatt")
presets = fg.presets()
choice = st.sidebar.selectbox("Vorlage", ["– auswählen –"] + list(presets), key="preset_choice",
                              help="Beispielmodelle zum Ausprobieren und als Ausgangspunkt für eigene Modelle.")
if st.sidebar.button("Vorlage laden", key="load_preset", disabled=choice.startswith("–")):
    set_model(presets[choice])
    S.mesh_sig = None
    st.rerun()
up = st.sidebar.file_uploader("Modell laden (JSON)", type=["json"], key="model_upload")
if up is not None and S.get("loaded_upload") != (up.name, up.size):
    try:
        set_model(fg.load_model(up.getvalue().decode("utf-8")))
        S.loaded_upload = (up.name, up.size)
        S.mesh_sig = None
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Datei nicht lesbar: {exc}")
st.sidebar.download_button("Modell speichern (JSON)", json.dumps(clean(model), indent=1, ensure_ascii=False).encode("utf-8"),
                           file_name="modell.json", mime="application/json", key="save_model")
st.sidebar.divider()
st.sidebar.subheader("Rechner")
py = st.sidebar.text_input("Python mit hpfem", value=fr.default_python(), key="solver_py",
                           help="Das Python, in dem hpfem gebaut ist (MSYS2: C:/msys64/ucrt64/bin/python3.exe).")
def _default_repo():
    import os
    for c in (os.environ.get("HPFEM_REPO"), HERE.parent.parent / "hp-FEM", HERE.parent / "hp-FEM", HERE):
        if c and (Path(c) / "python").is_dir():
            return str(c)
    return str(HERE)


repo = st.sidebar.text_input("hp-FEM-Ordner (enthält python/)", value=_default_repo(), key="solver_repo")
fm.set_repo(repo)
work = Path(st.sidebar.text_input("Arbeitsordner", value=str(HERE / "fem_work"), key="work_dir"))
threads = st.sidebar.number_input("Threads (0 = Standard)", 0, 64, 0, key="threads")
try:
    import gmsh  # noqa: F401
    st.sidebar.success("Gmsh gefunden")
except Exception:
    st.sidebar.error("Gmsh fehlt: pip install gmsh")
if not (Path(repo) / "python").is_dir():
    st.sidebar.warning("Im hp-FEM-Ordner fehlt python/ (hpfem).")
if not fm.library_available():
    st.sidebar.warning("Materialdaten der Bibliothek nicht gefunden (python/hpfem/data): Bibliotheksmaterialien lassen sich nicht anzeigen; "
                       "die Rechnung selbst kann sie trotzdem nutzen.")

# ------------------------------------------------------------------------------------------------ a run is in progress
proc = S.get("run_proc")
if proc is not None:
    log = fr.read_log(S.run_dir)
    st.title("Rechnung läuft …")
    if proc.poll() is None:
        frac, label = fr.progress_info(log)
        st.progress(frac, text=label)
        st.code("\n".join(log.splitlines()[-18:]) or "(noch keine Ausgabe)")
        if st.button("Abbrechen", key="cancel_run"):
            proc.terminate()
            S.run_proc, S.run_msg = None, "Rechnung abgebrochen (bis dahin berechnete Punkte stehen unter Ergebnisse)."
            S.view_dir = S.run_dir
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        code = proc.returncode
        S.run_proc = None
        S.view_dir = S.run_dir
        S.run_msg = ("Fertig." if code == 0 and not flags["errors"] else
                     f"Beendet mit Code {code}" + (f", {len(flags['errors'])} Fehlermeldung(en)." if flags["errors"] else "."))
        S.run_ok = code == 0 and not flags["errors"]
        st.rerun()
    st.stop()

# ------------------------------------------------------------------------------------------------------------------ tabs
st.title("FEM-Modellwerkstatt")
st.caption("Periodische Struktur (Gitter, Metaoberfläche, Schichtstapel) in einer Elementarzelle: eingeben, vernetzen, mit hp-FEM rechnen, auswerten.")
t_model, t_mesh, t_run, t_res, t_info = st.tabs(["1 Modell", "2 Netz", "3 Rechnung", "4 Ergebnisse", "5 Info"])

# ==================================================================================================================== 1 Modell
with t_model:
    left, right = st.columns([5, 4])
    inc, sw, dom = model["incidence"], model["sweep"], model["domain"]
    with left:
        model["name"] = st.text_input("Name des Modells", value=model["name"], key=f"name_{ver}")
        new_period = float(st.number_input("Periode P (nm)", value=float(model["period_nm"]), min_value=1.0, step=10.0, format="%.6g",
                                           key=f"period_{ver}", help="Breite der Elementarzelle in x. Das Gitter setzt sich periodisch fort."))
        if abs(new_period - model["period_nm"]) > 1e-12:
            if S.get("follow_period") and model["shapes"]:                       # keep the shapes centred when the period changes
                fg.shift_shapes(model, (new_period - model["period_nm"]) / 2)
                S.ver += 1
                model["period_nm"] = new_period
                st.rerun()
            model["period_nm"] = new_period
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.number_input("Alle Formen in x verschieben um (nm)", value=0.0, step=10.0, format="%.6g", key=f"shiftdx_{ver}",
                        help="Verschiebt alle Formen gemeinsam entlang x (die Schichten hängen nicht von x ab). Formen, die über den Zellrand ragen, setzen sich periodisch fort.")
        c2.button("Verschieben", key=f"shift_btn_{ver}", on_click=shift_cb, args=(f"shiftdx_{ver}",), disabled=not model["shapes"])
        ext = fg.shapes_x_extent(model)
        c3.button("Zentrieren", key=f"center_btn_{ver}", on_click=center_cb, disabled=not model["shapes"],
                  help="Verschiebt alle Formen so, dass die Mitte ihrer x-Ausdehnung in der Mitte der Zelle (P/2) liegt." +
                       (f" Jetzt: Ausdehnung {ext[0]:.0f} bis {ext[1]:.0f} nm, Mitte {(ext[0] + ext[1]) / 2:.0f} nm, Zellmitte {model['period_nm'] / 2:.0f} nm." if ext else ""))
        st.checkbox("Beim Ändern der Periode die Formen mitzentrieren (um die halbe Änderung verschieben)", key="follow_period",
                    help="Praktisch, wenn die Struktur in der Zelle mittig sitzen soll: die Periode ändern, die Formen wandern mit.")

        # ---------------------------------------------------------------- materials
        st.subheader("Materialien")
        for name in list(model["materials"]):
            spec = model["materials"][name]
            with st.expander(f"{name}: {fm.describe(spec)}"):
                c1, c2 = st.columns(2)
                newname = c1.text_input("Name", value=name, key=f"mn_{ver}_{name}")
                typ = c2.selectbox("Typ", list(TYPE_LABELS), index=list(TYPE_LABELS).index(spec["type"]), format_func=TYPE_LABELS.get,
                                   key=f"mt_{ver}_{name}")
                if typ != spec["type"]:
                    model["materials"][name] = copy.deepcopy(DEFAULT_SPEC[typ])
                    S.ver += 1
                    st.rerun()
                if typ == "library":
                    spec["name"] = st.selectbox("Bibliotheksmaterial", list(fm.LIBRARY), index=list(fm.LIBRARY).index(spec["name"]),
                                                key=f"ml_{ver}_{name}", help="Gemessene Daten (n, k) bzw. Sellmeier-Formel, mit Gültigkeitsbereich.")
                    r = fm.valid_range(spec)
                    if r:
                        st.caption(f"Gültig von {r[0]:.0f} bis {r[1]:.0f} nm.")
                elif typ == "index":
                    c1, c2 = st.columns(2)
                    spec["n"] = float(c1.number_input("n", value=float(spec["n"]), step=0.1, format="%.5g", key=f"mi_n_{ver}_{name}"))
                    spec["k"] = float(c2.number_input("k (Extinktion, ≥ 0)", value=float(spec["k"]), min_value=0.0, step=0.1, format="%.5g",
                                                       key=f"mi_k_{ver}_{name}"))
                elif typ == "eps":
                    c1, c2 = st.columns(2)
                    spec["re"] = float(c1.number_input("Re ε", value=float(spec["re"]), step=0.1, format="%.6g", key=f"me_r_{ver}_{name}"))
                    spec["im"] = float(c2.number_input("Im ε (Verlust > 0)", value=float(spec["im"]), step=0.1, format="%.6g", key=f"me_i_{ver}_{name}",
                                                        help="Konvention exp(−iωt): Verlust bedeutet Im ε > 0."))
                elif typ == "drude":
                    c1, c2, c3 = st.columns(3)
                    spec["eps_inf"] = float(c1.number_input("ε∞", value=float(spec["eps_inf"]), step=0.1, format="%.5g", key=f"md_e_{ver}_{name}"))
                    spec["omega_p_eV"] = float(c2.number_input("ħωp (eV)", value=float(spec["omega_p_eV"]), min_value=0.01, step=0.1, format="%.5g",
                                                                key=f"md_p_{ver}_{name}"))
                    spec["gamma_eV"] = float(c3.number_input("ħγ (eV)", value=float(spec["gamma_eV"]), min_value=0.0, step=0.01, format="%.5g",
                                                              key=f"md_g_{ver}_{name}"))
                else:
                    txt = st.text_area("Zeilen: Wellenlänge (nm), n, k", value="\n".join(f"{a:g}, {b:g}, {c:g}" for a, b, c in spec["rows"]),
                                       key=f"mtab_{ver}_{name}", height=120, help="Linear interpoliert, keine Extrapolation.")
                    try:
                        rows = [[float(v) for v in line.replace(";", ",").split(",")] for line in txt.splitlines() if line.strip()]
                        if len(rows) >= 2 and all(len(r) == 3 for r in rows):
                            spec["rows"] = rows
                        else:
                            st.error("Mindestens zwei Zeilen mit je drei Zahlen.")
                    except ValueError:
                        st.error("Zahlen nicht lesbar.")
                lam0 = inc["wavelength_nm"]
                try:
                    n_, k_ = fm.nk(spec, lam0)
                    st.caption(f"bei {lam0:g} nm: n = {n_:.4f}, k = {k_:.4f}, ε = {fm.eps_at(spec, lam0).real:.4f} {fm.eps_at(spec, lam0).imag:+.4f} i")
                except Exception as exc:
                    st.caption(f"bei {lam0:g} nm: {exc}")
                b1, b2 = st.columns(2)
                if newname != name and newname.strip():
                    if newname in model["materials"]:
                        b1.error("Name schon vergeben.")
                    else:
                        rename_material(model, name, newname.strip())
                        S.ver += 1
                        st.rerun()
                in_use = name in used_materials(model)
                if b2.button("Material löschen", key=f"mdel_{ver}_{name}", disabled=in_use or len(model["materials"]) <= 1,
                             help="Nur möglich, wenn das Material nirgends verwendet wird."):
                    del model["materials"][name]
                    S.ver += 1
                    st.rerun()
        c1, c2, c3 = st.columns([3, 2, 2])
        lib_pick = c1.selectbox("Aus der Bibliothek hinzufügen", ["–"] + [n for n in LIB_CHOICES], key=f"libadd_{ver}")
        if c2.button("Bibliotheksmaterial hinzufügen", key=f"libadd_btn_{ver}", disabled=lib_pick == "–"):
            model["materials"][unique_name(model, lib_pick)] = {"type": "library", "name": lib_pick}
            S.ver += 1
            st.rerun()
        if c3.button("Eigenes Material hinzufügen", key=f"matadd_{ver}"):
            model["materials"][unique_name(model, "Material")] = copy.deepcopy(DEFAULT_SPEC["index"])
            S.ver += 1
            st.rerun()
        names = list(model["materials"])

        # ---------------------------------------------------------------- stack
        st.subheader("Schichtaufbau (von oben nach unten)")
        model["cover"] = st.selectbox("Einfallsmedium (oben, verlustfrei)", names, index=names.index(model["cover"]) if model["cover"] in names else 0,
                                      key=f"cover_{ver}", help="Hier kommt das Licht an. Muss verlustfrei sein (Luft, Glas, Wasser …).")
        for i, layer in enumerate(model["layers"]):
            uid = layer["uid"]
            c = st.columns([4, 3, 1, 1, 1])
            layer["material"] = c[0].selectbox(f"Schicht {i + 1}", names, index=names.index(layer["material"]) if layer["material"] in names else 0,
                                               key=f"ly_m_{uid}_{ver}")
            layer["thickness_nm"] = float(c[1].number_input("Dicke (nm)", value=float(layer["thickness_nm"]), min_value=0.1, step=10.0, format="%.6g",
                                                            key=f"ly_t_{uid}_{ver}"))
            c[2].button("↑", key=f"ly_up_{uid}_{ver}", on_click=move, args=("layers", i, -1), help="nach oben")
            c[3].button("↓", key=f"ly_dn_{uid}_{ver}", on_click=move, args=("layers", i, 1), help="nach unten")
            c[4].button("✕", key=f"ly_del_{uid}_{ver}", on_click=delete_item, args=("layers", i), help="Schicht entfernen")
        st.button("Schicht hinzufügen", key=f"ly_add_{ver}", on_click=add_layer)
        model["substrate"] = st.selectbox("Substrat (unten, halbunendlich)", names, index=names.index(model["substrate"]) if model["substrate"] in names else 0,
                                          key=f"sub_{ver}")
        st.caption("y = 0 ist die Oberkante des Schichtaufbaus (Unterseite des Einfallsmediums). Formen liegen in diesen Koordinaten.")

        # ---------------------------------------------------------------- shapes
        st.subheader("Strukturen (Formen in der Zelle)")
        st.caption("Eine spätere Form überdeckt frühere Formen und die Schichten, z. B. eine Nut aus Luft in einem Substrat. "
                   "Formen, die über den Zellrand ragen, setzen sich periodisch fort.")
        for i, s in enumerate(model["shapes"]):
            uid = s["uid"]
            with st.expander(f"Form {i + 1}: {fg.SHAPE_TYPES[s['type']][0]} aus {s['material']}", expanded=False):
                s["material"] = st.selectbox("Material", names, index=names.index(s["material"]) if s["material"] in names else 0, key=f"sh_m_{uid}_{ver}")
                if s["type"] == "polygon":
                    txt = st.text_area("Eckpunkte (eine Zeile pro Punkt: x, y in nm)", value="\n".join(f"{x:g}, {y:g}" for x, y in s["points"]),
                                       key=f"sh_pts_{uid}_{ver}", height=130)
                    try:
                        pts = fg.parse_points(txt)
                        if len(pts) >= 3:
                            s["points"] = pts
                        else:
                            st.error("Mindestens drei Punkte.")
                    except ValueError as exc:
                        st.error(str(exc))
                else:
                    params = fg.SHAPE_TYPES[s["type"]][1]
                    cols = st.columns(2)
                    for k, (key, label) in enumerate(params):
                        s[key] = float(cols[k % 2].number_input(label, value=float(s[key]), step=10.0, format="%.6g", key=f"sh_{key}_{uid}_{ver}"))
                c = st.columns(4)
                c[0].button("↑ nach vorn/oben", key=f"sh_up_{uid}_{ver}", on_click=move, args=("shapes", i, -1), help="früher zeichnen (wird überdeckt)")
                c[1].button("↓ nach hinten", key=f"sh_dn_{uid}_{ver}", on_click=move, args=("shapes", i, 1), help="später zeichnen (überdeckt andere)")
                c[2].button("Duplizieren", key=f"sh_dup_{uid}_{ver}", on_click=duplicate_shape, args=(i,))
                c[3].button("Löschen", key=f"sh_del_{uid}_{ver}", on_click=delete_item, args=("shapes", i))
        c1, c2 = st.columns([3, 2])
        kind = c1.selectbox("Form hinzufügen", list(fg.SHAPE_TYPES), format_func=lambda k: fg.SHAPE_TYPES[k][0], key=f"shadd_{ver}")
        c2.button("Hinzufügen", key=f"shadd_btn_{ver}", on_click=add_shape, args=(kind,))

        # ---------------------------------------------------------------- illumination
        st.subheader("Beleuchtung")
        c = st.columns(4)
        inc["pol"] = c[0].radio("Polarisation", ["TE", "TM"], index=["TE", "TM"].index(inc["pol"]), horizontal=True, key=f"pol_{ver}",
                                help="TE: E senkrecht zur Einfallsebene (bei Gittern entlang der Linien). TM: E in der Einfallsebene.")
        inc["theta"] = float(c[1].number_input("Winkel θ (°)", value=float(inc["theta"]), min_value=0.0, max_value=89.0, step=5.0, format="%.4g",
                                               key=f"theta_{ver}", help="gegen die Normale, im Einfallsmedium"))
        inc["phi"] = float(c[2].number_input("Azimut φ (°)", value=float(inc["phi"]), step=5.0, format="%.4g", key=f"phi_{ver}",
                                             help="0: Einfallsebene senkrecht zu den Linien. Größer 0: konischer Einfall."))
        inc["wavelength_nm"] = float(c[3].number_input("Wellenlänge (nm)", value=float(inc["wavelength_nm"]), min_value=1.0, step=10.0, format="%.6g",
                                                       key=f"lam_{ver}"))
        sw["mode"] = st.selectbox("Durchlauf (Sweep)", list(SWEEP_MODES), index=list(SWEEP_MODES).index(sw["mode"]), format_func=SWEEP_MODES.get,
                                  key=f"swmode_{ver}", help="Mehrere Rechnungen in einem Lauf: Spektrum, Winkelscan …")
        if sw["mode"] != "none":
            c = st.columns(3)
            sw["start"] = float(c[0].number_input("von", value=float(sw["start"]), format="%.6g", key=f"swa_{sw['mode']}_{ver}"))
            sw["stop"] = float(c[1].number_input("bis", value=float(sw["stop"]), format="%.6g", key=f"swb_{sw['mode']}_{ver}"))
            sw["n"] = int(c[2].number_input("Punkte", 1, 400, int(sw["n"]), key=f"swn_{ver}"))

        # ---------------------------------------------------------------- domain
        st.subheader("Rechengebiet und Ränder")
        lam_lo, lam_hi = fr.wavelength_range(model)
        if st.button("Vorschlag aus der Wellenlänge übernehmen", key=f"suggest_{ver}",
                     help="Setzt Höhen des Einfallsraums, Substrattiefe und PML-Dicke aus Wellenlängenbereich, Substratmaterial, Polynomordnung, PML-Zielfehler und PML-Elementen (Reiter 2 und 3)."):
            model["domain"] = fg.suggest_domain(model, lam_lo, lam_hi, order=p_eff(), pml_target=float(S.solver["pml_target"]),
                                                cap_deg=float(S.solver["pml_angle_cap_deg"]), n_pml=float(S.mesh_settings["pml_cells_per_wavelength"]))
            S.ver += 1
            st.rerun()
        c = st.columns(3)
        dom["cover_nm"] = float(c[0].number_input("Höhe über der Struktur (nm)", value=float(dom["cover_nm"]), min_value=50.0, step=50.0, format="%.6g",
                                                  key=f"cov_{ver}", help="Luftraum zwischen Struktur und PML, in dem die reflektierten Ordnungen gemessen werden."))
        dom["pml_top_nm"] = float(c[1].number_input("PML oben (nm)", value=float(dom["pml_top_nm"]), min_value=50.0, step=50.0, format="%.6g", key=f"pmlt_{ver}",
                                                    help="Absorbierende Schicht; etwa 1 bis 2 Wellenlängen."))
        dom["substrate_nm"] = float(c[2].number_input("Substrattiefe (nm)", value=float(dom["substrate_nm"]), min_value=50.0, step=50.0, format="%.6g",
                                                      key=f"subd_{ver}", help="Tiefe des Substrats im Rechengebiet."))
        c = st.columns(2)
        dom["bottom"] = c[0].selectbox("Unterer Rand", ["pec", "pml"], index=["pec", "pml"].index(dom.get("bottom", "pec")), key=f"bot_{ver}",
                                       format_func=lambda v: {"pec": "Metallwand (nur bei stark absorbierendem Substrat)", "pml": "PML (verlustfreies Substrat)"}[v])
        if dom["bottom"] == "pml":
            dom["pml_bottom_nm"] = float(c[1].number_input("PML unten (nm)", value=float(max(dom.get("pml_bottom_nm", 0.0), 100.0)), min_value=50.0, step=50.0,
                                                           format="%.6g", key=f"pmlb_{ver}"))
        else:
            dom["pml_bottom_nm"] = 0.0

    with right:
        lam_lo, lam_hi = fr.wavelength_range(model)
        msgs = fg.validate(model, lam_lo)
        for lvl, text in msgs:
            (st.error if lvl == "error" else st.warning)(text)
        try:
            show_fig(fg.preview_figure(model), "geometrie.png", f"dl_geo_{ver}")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")
        rows = []
        for n_, spec in model["materials"].items():
            try:
                e = fm.eps_at(spec, inc["wavelength_nm"])
                rows.append({"Material": n_, "n": round(float(np.sqrt(e + 0j).real), 4), "k": round(float(abs(np.sqrt(e + 0j).imag)), 4),
                             "ε": f"{e.real:.3f} {e.imag:+.3f} i", "verwendet": "ja" if n_ in used_materials(model) else "nein"})
            except Exception as exc:
                rows.append({"Material": n_, "n": None, "k": None, "ε": str(exc)[:60], "verwendet": "ja" if n_ in used_materials(model) else "nein"})
        table_show(pd.DataFrame(rows))
        with st.expander("n und k über der Wellenlänge"):
            show_fig(fp.fig_materials(model, lam_lo, lam_hi), "materialien.png", f"dl_mat_{ver}")

# ======================================================================================================================= 2 Netz
with t_mesh:
    ms = S.mesh_settings
    st.markdown("Die Elementarzelle wird mit Gmsh in Dreiecke zerlegt. Die Elementgröße folgt der Wellenlänge im Material, an Grenzflächen wird "
                "verfeinert, Metalle werden nach der Eindringtiefe aufgelöst. Linker und rechter Rand erhalten identische Knoten (periodisch). "
                "Kreise und Ellipsen werden mit gekrümmten Elementen zweiter Ordnung vernetzt.")
    c = st.columns(4)
    p_now = p_eff()
    ms["cells_per_wavelength"] = float(c[0].slider("Elemente pro Wellenlänge", 1.0, 12.0, float(ms["cells_per_wavelength"]), 0.5, key="ms_cpw",
                                                   help="Bei hoher Polynomordnung p genügen etwa 12/p Elemente pro Wellenlänge im Material "
                                                        "(p = 4: 3, p = 3: 4, p = 2: 6). Mehr Elemente sind selten nötig und treiben die Freiheitsgrade hoch."))
    ms["skin_cells"] = float(c[1].slider("Elemente pro Eindringtiefe (Metall)", 0.5, 4.0, float(ms["skin_cells"]), 0.5, key="ms_skin",
                                         help="Feldabklinglänge 1/(k₀·k) im Metall, so viele Elemente darauf. Mit p = 4 genügt etwa 1."))
    ms["interface_factor"] = float(c[2].slider("Verfeinerung an Grenzflächen", 0.2, 1.0, float(ms["interface_factor"]), 0.05, key="ms_if",
                                               help="Faktor auf die Elementgröße an Materialgrenzen und Ecken (kleiner = feiner)."))
    ms["curved"] = c[3].checkbox("Gekrümmte Elemente (Kreise, Ellipsen)", value=bool(ms["curved"]), key="ms_curved")
    ms["pml_cells_per_wavelength"] = float(st.slider("Elemente pro Wellenlänge in der PML", 3.0, 16.0, float(ms.get("pml_cells_per_wavelength", 6.0)), 0.5, key="ms_pml",
                                                    help="In der PML ist das Feld gedehnt (|k·s|·h muss unter der Grenze der Bibliothek bleiben: 3 für p ≥ 4, sonst 0,75·p). "
                                                         "Etwa 6 genügen bei der vorgeschlagenen PML-Dicke. Mit dünnerer PML braucht es feinere Zellen."))
    lam_lo, lam_hi = fr.wavelength_range(model)
    errors = [t for l, t in fg.validate(model, lam_lo) if l == "error"]
    sig_now = geom_signature(model, ms)
    if st.button("Netz erzeugen", type="primary", key="mesh_btn", disabled=bool(errors)):
        with st.spinner("Gmsh vernetzt …"):
            try:
                t0 = time.time()
                mdir = work / "mesh"
                S.mesh_stats = fg.build_mesh(clean(model), dict(ms, lam_min_nm=lam_lo, lam_max_nm=lam_hi), mdir)
                S.mesh_stats["seconds"] = time.time() - t0
                S.mesh_sig, S.mesh_dir, S.mesh_model = sig_now, str(mdir), clean(model)
                S.mesh_err = ""
            except Exception as exc:
                S.mesh_err = str(exc)
                S.mesh_sig = None
    if errors:
        st.error("Das Modell hat Fehler (Reiter 1): " + " | ".join(errors))
    if S.get("mesh_err"):
        st.error(S.mesh_err)
    stats = S.mesh_stats
    if stats and S.mesh_sig is not None:
        if S.mesh_sig != sig_now:
            st.warning("Das Modell oder die Netzeinstellungen wurden seit dem Vernetzen geändert: Netz neu erzeugen.")
        c = st.columns(5)
        c[0].metric("Dreiecke", f"{stats['cells']:,}".replace(",", "."))
        c[1].metric("Knoten", f"{stats['nodes']:,}".replace(",", "."))
        c[2].metric("Elementordnung", stats["order"])
        c[3].metric("Kleinster Winkel", f"{stats['min_angle_deg']:.1f}°")
        c[4].metric("Kante min / max", f"{stats['edge_min_nm']:.1f} / {stats['edge_max_nm']:.0f} nm")
        if not stats["periodic_ok"]:
            st.error("Die Knoten am linken und rechten Rand stimmen nicht überein: für Bloch-Randbedingungen unbrauchbar.")
        if stats["min_angle_deg"] < 10:
            st.info(f"Kleinster Winkel {stats['min_angle_deg']:.1f}°: ein sehr spitzer Winkel in der Geometrie (z. B. eine runde Form nahe an einer Grenzfläche). "
                    "Meist unkritisch; hilft bei Problemen: Form leicht einsinken lassen oder die Verfeinerung an Grenzflächen verringern.")
        p_est = p_eff()
        dofs_est = est_dofs(stats["cells"], p_est)
        st.caption(f"Grobe Schätzung der Freiheitsgrade bei p = {p_est}: etwa {dofs_est:,}".replace(",", ".") +
                   f". Netz in {stats.get('seconds', 0):.1f} s. Empfehlung für p = {p_est}: etwa {12 / p_est:.1f} Elemente pro Wellenlänge.")
        if dofs_est > 400000:
            st.warning("Über 400 000 Freiheitsgrade: die Rechnung braucht viel Zeit und Speicher. Weniger Elemente pro Wellenlänge, eine höhere Ordnung p mit "
                       "gröberem Netz, oder ein kleineres Gebiet (Reiter 1) wählen.")
        c1, c2 = st.columns([2, 3])
        with c1:
            table_show(pd.DataFrame([{"Material": k, "Dreiecke": v, "max. Kante (nm)": stats["sizes_nm"].get(k)} for k, v in stats["cells_per_material"].items()]))
        with c2:
            zoom_mode = st.radio("Ausschnitt", ["ganze Zelle", "Struktur"], horizontal=True, key="mesh_zoom")
            lay = fg.layout(S.mesh_model)
            P = S.mesh_model["period_nm"]
            zoom = None
            if zoom_mode == "Struktur":
                depth = max(300.0, abs(lay["interfaces"][-1]) + 250.0)
                zoom = (0.0, P, lay["interfaces"][-1] - 250.0, lay["y_struct_top"] + 150.0) if depth else None
            show_fig(fg.mesh_figure(Path(S.mesh_dir) / "mesh_plot.npz", S.mesh_model, zoom=zoom), "netz.png", "dl_mesh")
    elif not errors:
        st.info("Noch kein Netz erzeugt.")

# ================================================================================================================== 3 Rechnung
with t_run:
    sv, mp_ = S.solver, S.maps
    mp_.setdefault("triangulate", True)
    mp_.setdefault("subdiv", 3)
    st.markdown("Der Solver (hp-FEM, Löser für TE, TM und konischen Einfall) rechnet die Streuung einer ebenen Welle an der periodischen Struktur: "
                "Bloch-Randbedingung in x, absorbierende Schichten (PML) oben und ggf. unten, analytischer Schichtstapel als Hintergrund.")
    inplane_ok = inc["pol"] == "TM" and abs(inc["phi"]) < 1e-9
    ENGINES = {"conical": "Konischer Löser (TE, TM, konisch), gleichmäßiges Netz",
               "inplane": "In-Ebenen-Löser (nur TM, φ = 0), gleichmäßiges Netz",
               "inplane_hp": "In-Ebenen-Löser (nur TM, φ = 0), hp-adaptiv"}
    avail = list(ENGINES) if inplane_ok else ["conical"]
    if sv.get("engine") not in avail:
        sv["engine"] = "conical"
    sv["engine"] = st.selectbox("Löser", avail, index=avail.index(sv["engine"]), format_func=ENGINES.get, key="sv_engine",
                                help="Konischer Löser: TE, TM und Azimut, Netz und Ordnung bleiben fest. In-Ebenen-Löser: nur TM in der Einfallsebene, dafür mit "
                                     "hp-adaptiver Verfeinerung (Fehlerschätzer, Dörfler-Markierung, h- oder p-Verfeinerung). Beide auf demselben Netz zu rechnen "
                                     "vergleicht die Löser.")
    if not inplane_ok:
        st.caption("Der In-Ebenen-Löser (mit hp-Adaptivität) rechnet nur TM mit φ = 0. Für TE und konischen Einfall steht der konische Löser zur Verfügung.")
    ad = sv["adaptive"]
    c = st.columns(4)
    if sv["engine"] == "inplane_hp":
        ad["p0"] = int(c[0].number_input("Startordnung p₀", 1, 6, int(ad["p0"]), key="ad_p0", help="Anfangsordnung aller Elemente; die Schleife erhöht sie, wo das Feld glatt ist."))
    else:
        sv["order"] = int(c[0].slider("Polynomordnung p", 1, 8, int(sv["order"]), key="sv_p",
                                      help="Höhere Ordnung = genauer bei gleichem Netz, aber mehr Freiheitsgrade pro Element. 3 bis 5 sind üblich."))
    sv["pml_target"] = float(c[1].select_slider("PML-Zielfehler", options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], value=float(sv["pml_target"]), format_func=lambda v: f"{v:g}",
                                                key="sv_pml", help="Zulässiger Reflexionsfehler der PML (auf die Reflektanz). Wird für den größten Winkel der Beugungsordnungen ausgelegt."))
    sv["backend"] = c[2].selectbox("Löser (direkt)", ["AUTO", "SPARSE_LU", "MUMPS"], index=["AUTO", "SPARSE_LU", "MUMPS"].index(sv["backend"]), key="sv_backend",
                                   help="AUTO nimmt MUMPS, wenn verfügbar (schneller und sparsamer), sonst Eigen SparseLU.")
    sv["orders_max"] = int(c[3].number_input("Ordnungen bis ±", 1, 8, int(sv["orders_max"]), key="sv_orders", help="Größte Beugungsordnung, die ausgewertet wird."))
    if sv["engine"] == "inplane_hp":
        st.markdown("**hp-Adaptivität.** Jeder Schritt rechnet, schätzt den Fehler je Dreieck, markiert die schlechtesten (Dörfler) und verfeinert sie in h (teilen, "
                    "wo das Feld singulär ist) oder in p (Ordnung erhöhen, wo es glatt ist). Dreiecke am periodischen Rand und in der PML werden nie markiert.")
        c = st.columns(4)
        ad["steps"] = int(c[0].number_input("Höchstens Schritte", 1, 40, int(ad["steps"]), key="ad_steps"))
        ad["max_dofs"] = int(c[1].number_input("Höchstens Freiheitsgrade", 5000, 1000000, int(ad["max_dofs"]), 5000, key="ad_dofs",
                                               help="Die Schleife hört nach dem Schritt auf, der diese Grenze überschreitet."))
        ad["dorfler"] = float(c[2].slider("Dörfler-Anteil", 0.1, 0.9, float(ad["dorfler"]), 0.05, key="ad_dorfler",
                                          help="Anteil des geschätzten Gesamtfehlers, der in den markierten Dreiecken stecken soll. Größer = mehr Dreiecke je Schritt."))
        ad["tol"] = float(c[3].select_slider("Abbruch bei Änderung von R, T unter", options=[0.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], value=float(ad["tol"]),
                                             format_func=lambda v: "aus" if v == 0 else f"{v:g}", key="ad_tol",
                                             help="Stoppt, wenn sich R und T in zwei aufeinanderfolgenden Schritten um weniger ändern."))
        ad["reuse_mesh"] = st.checkbox("Bei einem Durchlauf nur am ersten Punkt adaptieren, das Endnetz für alle weiteren Punkte benutzen", value=bool(ad["reuse_mesh"]),
                                       key="ad_reuse", help="Spart viel Zeit bei Spektren; das Netz passt dann zur ersten Wellenlänge bzw. zum ersten Winkel.")
        st.info("Für die hp-Adaptivität ein grobes Startnetz wählen (Reiter 2: etwa 2 Elemente pro Wellenlänge, Startordnung 3). Die Verfeinerung setzt die Elemente dann selbst.")
    with st.expander("Weitere Einstellungen"):
        c = st.columns(3)
        sv["pml_angle_cap_deg"] = float(c[0].number_input("PML-Auslegung: größter Winkel (°)", 30.0, 88.0, float(sv["pml_angle_cap_deg"]), 1.0, key="sv_cap",
                                                          help="Obergrenze für den Winkel, für den die PML ausgelegt wird. Streifende Ordnungen brauchen sehr starke PML."))
        sv["order_points"] = int(c[1].number_input("Stützstellen für die Fourier-Zerlegung", 16, 512, int(sv["order_points"]), 16, key="sv_pts"))
        eq = c[2].number_input("Zusätzliche Quadraturordnung", 0, 6, 0, key="sv_eq", help="Erhöht die Integrationsgenauigkeit; selten nötig.")
        sv["extra_quadrature_order"] = int(eq) if eq else None
    st.subheader("Ausgabe")
    n_pts = max(len(fr.sweep_values(model["sweep"])), 1)
    c = st.columns(3)
    mp_["enabled"] = c[0].checkbox("Feldkarten speichern", value=bool(mp_["enabled"]), key="mp_en")
    idx_opts = list(range(min(n_pts, 60)))
    mp_["indices"] = c[1].multiselect("Karten für Punkte Nr.", idx_opts, default=[i for i in mp_["indices"] if i in idx_opts] or [0], key="mp_idx",
                                      help="Nummer im Durchlauf (0 = erster Punkt).") if mp_["enabled"] else []
    mp_["res_nm"] = float(c[2].number_input("Kartenauflösung (nm pro Pixel)", 1.0, 20.0, float(mp_["res_nm"]), 0.5, key="mp_res",
                                           help="Feiner = schönere Karten und genauere Absorption aus dem Feld, aber längere Auswertung.")) if mp_["enabled"] else mp_["res_nm"]
    if mp_["enabled"]:
        c = st.columns(2)
        mp_["triangulate"] = c[0].checkbox("Elementdarstellung speichern (exakt, Materialgrenzen scharf)", value=bool(mp_["triangulate"]), key="mp_tri",
                                           help="Das Feld auf den unterteilten Dreiecken des Netzes (hpfem `triangulate`): Sprünge der Normalkomponente an Materialgrenzen bleiben "
                                                "scharf, das Material ist je Dreieck exakt bekannt. Braucht ein hpfem mit M15 F3, sonst wird nur das Raster gespeichert.")
        if mp_["triangulate"]:
            mp_["subdiv"] = int(c[1].slider("Unterteilung je Dreieck", 1, 6, int(mp_["subdiv"]), key="mp_sub",
                                            help="n-fache Unterteilung jedes Dreiecks; 3 bis 4 genügt für Polynomordnung 4 bis 5."))
    c = st.columns(2)
    do_pscan = c[0].checkbox("Konvergenzstudie in p (statt des Durchlaufs)", key="do_pscan",
                             help="Rechnet den ersten Punkt mit mehreren Polynomordnungen und zeigt Fehler gegen Freiheitsgrade.")
    pscan_txt = c[1].text_input("Ordnungen", "2, 3, 4, 5", key="pscan_txt", disabled=not do_pscan)

    lam_lo, lam_hi = fr.wavelength_range(model)
    problems = [t for l, t in fg.validate(model, lam_lo) if l == "error"]
    for lam in (lam_lo, lam_hi):
        for n_, spec in model["materials"].items():
            if n_ in used_materials(model):
                try:
                    fm.eps_at(spec, lam)
                except Exception as exc:
                    problems.append(str(exc))
                    break
    problems = list(dict.fromkeys(problems))
    sig_now = geom_signature(model, S.mesh_settings)
    mesh_ok = S.mesh_stats is not None and S.mesh_sig == sig_now and S.mesh_stats["periodic_ok"]
    plan = fg.pml_plan(model, p_eff(), float(sv["pml_target"]), float(sv["pml_angle_cap_deg"]), float(S.mesh_settings.get("pml_cells_per_wavelength", 6.0)),
                       S.mesh_stats if (S.mesh_stats and S.mesh_sig == sig_now) else None)
    with st.expander(f"PML-Auslegung (Winkel {plan['psi_deg']:.0f}°, R₀ = {plan['R0']:.1e}, Auflösungsgrenze {plan['limit']:.2f})", expanded=not plan["ok"]):
        table_show(pd.DataFrame([{"Seite": sd["side"], "Dicke (nm)": round(sd["thickness_nm"]), "σ̂max": round(sd["sigma"], 2), "|s|": round(sd["s"], 2),
                                  "nötige Zellgröße ≤ (nm)": round(sd["h_needed_nm"], 1), "Zellgröße im Netz (nm)": round(sd["h_mesh_nm"], 1),
                                  "aufgelöst": "ja" if sd["ok"] else "NEIN"} for sd in plan["sides"]]))
        st.caption("Kriterium der Bibliothek: k₀·n·|s|·h ≤ Grenze(p) am Ende der Schicht, |s| = √(1 + σ̂²), σ̂ = −(m+1)·ln R₀ / (2·k₀·n_ref·d). Eine dünne PML mit strengem Zielfehler "
                   "erzwingt sehr feine Zellen. Abhilfe: PML dicker („Vorschlag aus der Wellenlänge“ im Reiter 1), mehr Elemente in der PML (Reiter 2), "
                   "größerer PML-Zielfehler oder kleinerer größter Winkel der PML-Auslegung.")
    if not plan["ok"]:
        bad = [sd for sd in plan["sides"] if not sd["ok"]]
        st.warning("PML unterauflösend: " + "; ".join(f"{sd['side']}: Zellen {sd['h_mesh_nm']:.0f} nm, nötig ≤ {sd['h_needed_nm']:.1f} nm" for sd in bad) +
                   ". Die Rechnung läuft, die Ergebnisse sind dann nur auf etwa 10⁻³ verlässlich. „Vorschlag aus der Wellenlänge“ (Reiter 1) legt Dicke und Netz passend aus.")
    if sv.get("engine") == "inplane_hp" and S.mesh_stats and S.mesh_stats.get("order") == 2:
        st.warning("Das Netz hat gekrümmte Elemente (zweite Ordnung). hp-Verfeinerung gekrümmter Zellen ist ungetestet: Schlägt die Rechnung fehl, "
                   "gerade Ränder verwenden (Reiter 2: gekrümmte Elemente ausschalten) oder den konischen Löser nehmen.")
    if problems:
        st.error("Vor der Rechnung beheben: " + " | ".join(problems))
    if not mesh_ok:
        st.warning("Es gibt kein aktuelles Netz (Reiter 2: „Netz erzeugen“).")
    if not (Path(repo) / "python").is_dir():
        st.warning("Im hp-FEM-Ordner fehlt python/ (Seitenleiste).")
    n_runs = n_pts if not do_pscan else len([v for v in pscan_txt.split(",") if v.strip()])
    st.caption(f"{n_runs} Rechnung(en)" + (f", etwa {est_dofs(S.mesh_stats['cells'], p_eff()):,} Freiheitsgrade je Rechnung".replace(",", ".") if S.mesh_stats else ""))
    if S.mesh_stats and est_dofs(S.mesh_stats["cells"], p_eff()) > 400000:
        st.warning("Über 400 000 Freiheitsgrade je Rechnung: das dauert lange und braucht viel Speicher.")
    if st.button("Rechnung starten", type="primary", key="run_btn", disabled=bool(problems) or not mesh_ok):
        try:
            run_dir = work / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            for f in ("mesh.msh", "mesh_plot.npz"):
                shutil.copy(Path(S.mesh_dir) / f, run_dir / f)
            P_ = model["period_nm"]
            lay = fg.layout(model)
            nx = int(round(P_ / mp_["res_nm"])) + 1
            ny = min(int(round((lay["y_cover_top"] - lay["y_sub_bottom"]) / mp_["res_nm"])) + 1, 900)
            maps = dict(enabled=bool(mp_["enabled"]), indices=[int(i) for i in mp_["indices"]], nx=nx, ny=ny,
                        triangulate=bool(mp_["enabled"] and mp_["triangulate"]), subdivisions=int(mp_["subdiv"]))
            pscan = [int(v) for v in pscan_txt.split(",") if v.strip()] if do_pscan else None
            job = fr.make_job(clean(model), dict(sv), maps, pscan)
            fr.write_job(run_dir, job)
            S.run_proc = fr.start_worker(py, repo, run_dir, int(threads))
            S.run_dir, S.run_msg = str(run_dir), ""
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("run_msg"):
        (st.success if S.get("run_ok", True) else st.error)(S.run_msg + " Die Ergebnisse stehen im Reiter 4.")
    if S.get("run_dir") and (Path(S.run_dir) / "log.txt").is_file():
        lw = fr.library_warnings(fr.read_log(S.run_dir))
        if lw:
            with st.expander(f"Warnungen des Solvers ({len(lw)} verschiedene)", expanded=True):
                for text_, count in lw:
                    st.warning(f"{text_}" + (f"  (×{count})" if count > 1 else ""))
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(S.run_dir, tail=80))

# ================================================================================================================== 4 Ergebnisse
with t_res:
    arch = work / "archiv"
    c = st.columns([3, 2, 3])
    archives = sorted([p.name for p in arch.iterdir() if (p / "results.json").is_file()]) if arch.is_dir() else []
    pick = c[0].selectbox("Gespeicherte Rechnung öffnen", ["– aktuelle –"] + archives, key="arch_pick")
    if c[1].button("Öffnen", key="arch_open", disabled=pick.startswith("–")):
        S.view_dir = str(arch / pick)
        st.rerun()
    view = Path(S.get("view_dir", work / "run"))
    res = fp.load_results(view) if (view / "results.json").is_file() else None
    if res is None:
        st.info("Noch keine Ergebnisse. Erst Netz erzeugen (Reiter 2), dann rechnen (Reiter 3). "
                "Nach einem Neustart der Seite findest du die letzte Rechnung hier wieder.")
    else:
        job = json.loads((view / "job.json").read_text(encoding="utf-8"))
        rmodel, mode = job["model"], job["sweep"]["mode"]
        name_arch = c[2].text_input("Archivieren unter", value=time.strftime("%Y%m%d_%H%M") + "_" + "".join(ch for ch in rmodel["name"][:20] if ch.isalnum()),
                                    key="arch_name")
        if c[2].button("Rechnung archivieren", key="arch_save"):
            arch.mkdir(parents=True, exist_ok=True)
            shutil.copytree(view, arch / name_arch, dirs_exist_ok=True)
            st.success(f"Gespeichert unter {arch / name_arch}")
        df = fp.results_dataframe(res, mode)
        errs = [p for p in res["points"] if "error" in p] + [p for p in res.get("pscan", []) if "error" in p]
        for p in errs:
            st.error(f"Punkt {p.get('index', p.get('order'))}: {p['error']}")
        meta = res["meta"]
        st.caption(f"{meta['model']} · {ENGINE_SHORT.get(meta.get('engine', 'conical'), '')} · {SWEEP_MODES.get(mode, mode)} · "
                   f"{'Start-' if meta.get('engine') == 'inplane_hp' else ''}p = {meta['order']} · {meta['cells']} Dreiecke (Startnetz) · "
                   f"{len([p for p in res['points'] if 'error' not in p])} von {len(res['points'])} Punkten · Start {meta['started']}")
        sub = st.tabs(["Spektrum / Ordnungen", "Feldkarten", "Schnitte", "Absorption", "Konvergenz in p", "hp-Adaptivität", "Tabelle und Export"])
        with sub[0]:
            if len(df) and "R" in df:
                show_fig(fp.fig_spectrum(df, mode), "spektrum.png", "dl_spec")
                last = df.dropna(subset=["R"]).iloc[0]
                if pd.notna(last["T"]):
                    resid = float(1 - last["R"] - last["T"] - (last["A"] if pd.notna(last["A"]) else 0))
                    st.caption("A = 1 − R − T ist die im Modell absorbierte Leistung (Struktur und Schichten). Negative Werte oder große Sprünge deuten auf zu grobes Netz oder zu schwache PML.")
                else:
                    st.caption("Das Substrat ist verlustbehaftet: T ist nicht messbar, A inkl. Substrat = 1 − R enthält alles, was im Substrat absorbiert wird. "
                               "„Ebener Stapel“ ist die Referenz ohne Struktur (Fresnel).")
            else:
                st.info("Keine auswertbaren Punkte.")
        with sub[1]:
            maps = sorted(view.glob("maps_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
            if not maps:
                st.info("Für diese Rechnung wurden keine Feldkarten gespeichert (Reiter 3: „Feldkarten speichern“).")
            else:
                c = st.columns(4)
                which = c[0].selectbox("Punkt Nr.", [int(p.stem.split("_")[1]) for p in maps], key="map_which")
                qkey = c[1].selectbox("Größe", list(fp.QUANTITIES), key="map_q")
                periods = c[2].selectbox("Perioden", [1, 2, 3], key="map_periods")
                cmap = c[3].selectbox("Farbskala", ["automatisch", "inferno", "viridis", "magma", "turbo", "RdBu_r", "coolwarm"], key="map_cm")
                c = st.columns(4)
                logs = c[0].checkbox("logarithmisch", key="map_log")
                geo = c[1].checkbox("Geometrie einzeichnen", value=True, key="map_geo")
                msh = c[2].checkbox("Netz einzeichnen", key="map_mesh")
                vmax_txt = c[3].text_input("Obergrenze der Skala (leer = automatisch)", "", key="map_vmax")
                pt = next(p for p in res["points"] if p.get("index") == which)
                A_en = pt["A"] if pt.get("A") is not None else pt["A_incl_substrate"]
                try:
                    vmax = float(vmax_txt) if vmax_txt.strip() else None
                except ValueError:
                    vmax = None
                tri_file = view / f"tri_{which}.npz"
                rep = st.radio("Darstellung", ["Elemente (exakt)", "Raster"], horizontal=True, key="map_rep",
                               help="Elemente: Feld auf den unterteilten Dreiecken des Netzes, Sprünge an Materialgrenzen scharf. Raster: reguläres Gitter.") \
                    if tri_file.is_file() else "Raster"
                if rep.startswith("Elemente"):
                    tri = fp.load_tri(tri_file)
                    show_fig(fp.fig_map_tri(tri, fp.derived_tri(tri, rmodel), rmodel, qkey, periods=periods, cmap=None if cmap == "automatisch" else cmap, vmax=vmax,
                                            log=logs, show_geometry=geo), f"karte_{which}_elemente.png", "dl_map")
                    if msh:
                        st.caption("Das Netz lässt sich nur in der Rasterdarstellung einzeichnen.")
                else:
                    mp = fp.load_map(view / f"maps_{which}.npz")
                    d = fp.derived(mp, rmodel, A_en)
                    show_fig(fp.fig_map(mp, d, rmodel, qkey, periods=periods, cmap=None if cmap == "automatisch" else cmap, vmax=vmax, log=logs,
                                        mesh_npz=(view / "mesh_plot.npz") if msh else None, show_geometry=geo), f"karte_{which}.png", "dl_map")
                st.caption("Feld in den Achsen der Rechnung: x entlang der Periode, y vertikal, z entlang der Linien. Einfallende Welle mit Amplitude |E₀| = 1.")
        with sub[2]:
            maps = sorted(view.glob("maps_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
            if not maps:
                st.info("Keine Feldkarten gespeichert.")
            else:
                c = st.columns(3)
                which = c[0].selectbox("Punkt Nr.", [int(p.stem.split("_")[1]) for p in maps], key="cut_which")
                axis = c[1].radio("Schnitt", ["vertikal (bei festem x)", "horizontal (bei festem y)"], key="cut_axis")
                mp = fp.load_map(view / f"maps_{which}.npz")
                d = fp.derived(mp, rmodel)
                if axis.startswith("vertikal"):
                    pos = c[2].slider("x (nm)", float(d["x"][0]), float(d["x"][-1]), float((d["x"][0] + d["x"][-1]) / 2), key="cut_x")
                    show_fig(fp.fig_cut(d, rmodel, "y", pos), "schnitt_vertikal.png", "dl_cut")
                else:
                    pos = c[2].slider("y (nm)", float(d["y"][0]), float(d["y"][-1]), 0.0, key="cut_y")
                    show_fig(fp.fig_cut(d, rmodel, "x", pos), "schnitt_horizontal.png", "dl_cut")
        with sub[3]:
            ok_pts = [p for p in res["points"] if "error" not in p]
            if not ok_pts:
                st.info("Keine auswertbaren Punkte.")
            else:
                if any(c.startswith("A[") for c in df.columns):
                    st.markdown("**Absorption je Material über den Durchlauf** (exakt: Volumenquadratur der Joule-Wärme auf dem Netz)")
                    show_fig(fp.fig_absorption_sweep(df, mode), "absorption_durchlauf.png", "dl_abs_sweep")
                which = st.selectbox("Punkt Nr.", [p["index"] for p in ok_pts], key="abs_which")
                pt = next(p for p in ok_pts if p["index"] == which)
                A_en = pt["A"] if pt.get("A") is not None else pt["A_incl_substrate"]
                ab = pt.get("absorbed")
                map_file = view / f"maps_{which}.npz"
                d = fp.derived(fp.load_map(map_file), rmodel, A_en) if map_file.is_file() else None
                names_ = list(dict.fromkeys(list((ab or {}).get("by_material", {})) + list((d or {}).get("regions", {}))))
                if not names_:
                    st.info("In diesem Modell absorbiert kein Material (Im ε = 0)." if (ab or d) else
                            "Keine Absorptionswerte: dieses hpfem hat kein `absorbed_power_by_tag`, und es wurden keine Feldkarten gespeichert (Reiter 3).")
                else:
                    rows = [{"Material": n_, "exakt (Quadratur auf dem Netz)": (ab or {}).get("by_material", {}).get(n_),
                             "aus dem Kartenraster": (d or {}).get("regions", {}).get(n_),
                             "Raster, normiert auf die Bilanz": ((d or {}).get("regions_norm") or {}).get(n_)} for n_ in names_]
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        fmt_ = {k: "{:.4f}" for k in ("exakt (Quadratur auf dem Netz)", "aus dem Kartenraster", "Raster, normiert auf die Bilanz")}
                        try:
                            st.dataframe(pd.DataFrame(rows).style.format(fmt_, na_rep="–"), width="stretch", hide_index=True)
                        except Exception:
                            table_show(pd.DataFrame(rows))
                        st.metric("Energiebilanz 1 − R − T" if pt.get("A") is not None else "1 − R (inkl. Substrat)", f"{A_en:.4f}")
                        if ab:
                            st.metric("Summe exakt (physikalisches Gebiet)", f"{ab['total']:.4f}",
                                      delta=f"Verhältnis zur Bilanz {ab['total'] / A_en:.3f}" if A_en > 1e-9 else None, delta_color="off")
                        if d is not None:
                            st.metric("Summe aus dem Kartenraster", f"{d['A_map']:.4f}", delta=f"Verhältnis {d['map_quality']:.3f}" if d["map_quality"] else None,
                                      delta_color="off")
                        st.caption("**Exakt:** Q = ½ ω ε₀ Im ε |E|² über die Zellen des Netzes integriert (hpfem `absorbed_power_by_tag`), nur das physikalische Gebiet (PML-Zellen "
                                   "ausgenommen), bezogen auf die einfallende Leistung je Periode. Die Summe weicht von der Energiebilanz um deren Extraktionsfehler ab "
                                   "(und bei verlustbehaftetem Substrat um den Anteil unterhalb des Gebiets). **Raster:** gleiche Größe auf dem Kartenraster, an scharfen "
                                   "Materialgrenzen mit einem Fehler der Größe eines Pixels; die normierte Spalte verteilt die Energiebilanz nach dem Feld.")
                        if ab is None:
                            st.info("Dieses hpfem hat noch kein `absorbed_power_by_tag` (M15 F4): die exakte Spalte fehlt, die Rasterwerte bleiben.")
                    with c2:
                        bars = (ab or {}).get("by_material") or (d or {}).get("regions_norm") or (d or {}).get("regions")
                        show_fig(fp.fig_absorption(bars, A_en), "absorption.png", "dl_abs")
        with sub[4]:
            if res.get("pscan"):
                show_fig(fp.fig_pscan(res["pscan"]), "konvergenz_p.png", "dl_pscan")
                table_show(pd.DataFrame([{"p": p.get("order"), "Freiheitsgrade": p.get("dofs"), "Zeit (s)": p.get("time_s"), "R": p.get("R"), "T": p.get("T"), "A": p.get("A"),
                                          "Fehler": p.get("error")} for p in res["pscan"]]))
            else:
                st.info("Keine Konvergenzstudie in dieser Rechnung (Reiter 3: „Konvergenzstudie in p“).")
        with sub[5]:
            ad_pts = [p for p in res["points"] if p.get("steps")]
            if not ad_pts:
                st.info("Keine hp-adaptive Rechnung in diesem Ergebnis (Reiter 3: Löser „In-Ebenen-Löser, hp-adaptiv“).")
            else:
                wi = st.selectbox("Punkt Nr.", [p["index"] for p in ad_pts], key="ad_which") if len(ad_pts) > 1 else ad_pts[0]["index"]
                pt = next(p for p in ad_pts if p["index"] == wi)
                steps_ = pt["steps"]
                table_show(pd.DataFrame([{"Schritt": s_["step"], "Dreiecke": s_["cells"], "Freiheitsgrade": s_["dofs"], "p max": s_["max_p"], "R": s_["R"], "T": s_["T"],
                                          "η (Schätzer)": s_["eta"], "Änderung": s_["change"], "Zeit (s)": s_["time_s"]} for s_ in steps_]))
                if steps_ and steps_[-1].get("stopped"):
                    st.warning("Die Schleife wurde vorzeitig beendet: Die periodischen Ränder des verfeinerten Netzes ließen sich nicht angleichen "
                               f"({steps_[-1]['stopped']}). Das Ergebnis ist das des letzten vollständigen Schritts.")
                show_fig(fp.fig_adaptive_steps(steps_), "adaptivitaet.png", "dl_adaptive")
                st.caption("Links: R und T je Schritt. Mitte: Abstand zum letzten Schritt und Fehlerschätzer η über den Freiheitsgraden (exponentielle Konvergenz zeigt "
                           "sich als fallende Kurve, die in der Auftragung gegen N^(1/3) gerade wird). Rechts: größte Ordnung und Zahl der Dreiecke.")
                hp_file = view / f"hpmesh_{wi}.npz"
                if hp_file.is_file():
                    zm = st.radio("Ausschnitt", ["ganze Zelle", "Struktur"], horizontal=True, key="hp_zoom")
                    zoom = None
                    if zm == "Struktur":
                        lay_ = fg.layout(rmodel)
                        zoom = (0.0, rmodel["period_nm"], lay_["interfaces"][-1] - 250.0, lay_["y_struct_top"] + 150.0)
                    show_fig(fp.fig_hpmesh(hp_file, rmodel, zoom=zoom), "hp_netz.png", "dl_hpmesh")
                    st.caption("Endnetz, jedes Dreieck nach seiner Polynomordnung gefärbt: kleine Dreiecke mit niedriger Ordnung an Singularitäten (Kanten, Ecken), "
                               "große Dreiecke mit hoher Ordnung im glatten Feld.")
        with sub[6]:
            if len(df):
                table_show(df)
                st.download_button("Ergebnisse (CSV)", csv_bytes(df), file_name="ergebnisse.csv", mime="text/csv", key="dl_csv")
            st.download_button("Modell dieser Rechnung (JSON)", json.dumps(rmodel, indent=1, ensure_ascii=False).encode("utf-8"), file_name="modell_der_rechnung.json",
                               mime="application/json", key="dl_model_res")
            st.caption(f"Ordner mit allen Dateien (Netz, Protokoll, Karten als .npz): {view}")

# ======================================================================================================================= 5 Info
with t_info:
    st.markdown("""
**Ablauf.** 1 Modell eingeben (Materialien, Schichten, Formen, Beleuchtung, Gebiet) → 2 Netz erzeugen → 3 Rechnung starten → 4 Ergebnisse ansehen.

**Koordinaten.** x liegt entlang der Periode, y vertikal nach oben, y = 0 an der Oberkante des Schichtaufbaus. Die dritte Richtung z ist die
Richtung der Linien eines Gitters (translationsinvariant). Bei konischem Einfall (Azimut φ > 0) hat die Welle eine Komponente entlang z.

**Polarisation.** TE: E senkrecht zur Einfallsebene (bei φ = 0 entlang der Linien, E_z). TM: E in der Einfallsebene.
Winkel θ gegen die Normale. Konvention exp(−iωt): Verlust bedeutet Im ε > 0, ε = (n + ik)².

**Was gerechnet wird.** Eine ebene Welle fällt aus dem Einfallsmedium auf die periodische Struktur. Der Schichtaufbau ist der analytische Hintergrund (Streufeld-Formulierung),
die Struktur die Abweichung davon. Links und rechts gilt die Bloch-Randbedingung, oben (und bei verlustfreiem Substrat unten) absorbiert eine PML.
Gemessen werden die reflektierten Beugungsordnungen im Einfallsraum und, bei verlustfreiem Substrat, die transmittierten im Substrat. Daraus: R, T und A = 1 − R − T.

**Löser.** Der konische Löser rechnet TE, TM und beliebigen Azimut auf dem Netz mit fester Ordnung. Der In-Ebenen-Löser rechnet nur TM mit φ = 0, hat aber die
hp-adaptive Verfeinerung: Fehlerschätzer je Dreieck, Dörfler-Markierung, dann h- oder p-Verfeinerung nach der Glattheit des Feldes. Das lohnt sich bei scharfen
Metallecken (TM), wo gleichmäßige Netze nur langsam konvergieren. Beide Löser auf demselben Netz zu rechnen ist ein Test auf Übereinstimmung.

**Ränder und Substrat.** Ein verlustfreies Substrat braucht unten eine PML. Ein stark absorbierendes Substrat (Metall, Silizium im UV) kann unten mit einer Metallwand
abgeschlossen werden, wenn die Tiefe mehrere Eindringtiefen beträgt. „Vorschlag aus der Wellenlänge“ setzt beides.

**Genauigkeit.** Polynomordnung p (Reiter 3) und Netz (Reiter 2) zusammen bestimmen den Fehler. Die Konvergenzstudie in p zeigt, ob das Netz reicht.
Scharfe Metallecken (TM) konvergieren langsam und brauchen hp-Verfeinerung, die dieser Löser noch nicht hat. Dort Verfeinerung an Grenzflächen verkleinern und p erhöhen.

**Grenzen.** Eindimensional periodische Strukturen mit einer Zelle (keine Mehrfach-Formen mit Überlappungsproblemen über mehrere Perioden), Einfallsmedium verlustfrei,
nichtmagnetische Materialien, Strukturen müssen in die Zelle passen (Formen dürfen über den Rand ragen und werden periodisch fortgesetzt).
Tangential an eine Grenzfläche gelegte Kreise erzeugen verzerrte Dreiecke: die Form 1 bis 2 nm einsinken lassen.

**Voraussetzungen.** App: `pip install streamlit numpy pandas matplotlib gmsh`. Rechnung: ein Python mit gebautem hpfem (Seitenleiste „Python mit hpfem“).
Die Datei fem_worker.py liegt neben dieser App und wird vom hpfem-Python gestartet; sie braucht fem_materials.py im selben Ordner.
""")
