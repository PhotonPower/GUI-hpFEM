"""Streamlit building blocks shared by the two modes of the app (periodic structures and bodies of revolution)."""
from __future__ import annotations

import copy
import io

import streamlit as st

import fem_materials as fm

TYPE_LABELS = {"library": "Bibliothek (hpfem)", "index": "n + ik (konstant)", "eps": "ε (konstant)", "drude": "Drude-Metall", "table": "Tabelle λ, n, k"}
LIB_CHOICES = ["Si", "Ag", "Au", "Al", "GaAs", "MAPbI3", "SiO2", "TiO2", "water", "air", "vacuum"]
DEFAULT_SPEC = {"library": {"type": "library", "name": "SiO2"}, "index": {"type": "index", "n": 1.5, "k": 0.0},
                "eps": {"type": "eps", "re": 2.25, "im": 0.0}, "drude": {"type": "drude", "eps_inf": 1.0, "omega_p_eV": 9.0, "gamma_eV": 0.07},
                "table": {"type": "table", "rows": [[400.0, 1.5, 0.0], [800.0, 1.5, 0.0]]}}


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


def unique_name(model, base):
    n, k = base, 2
    while n in model["materials"]:
        n, k = f"{base} {k}", k + 1
    return n


def material_editor(model, ver, lam0, used, rename, changed, prefix=""):
    """Editor of model["materials"]: one expander per material (name, type, parameters, n and k at lam0, rename, delete) and the buttons to
    add a library or an own material. rename(model, old, new) renames everywhere; changed() is called before every st.rerun (bumps the widget
    version)."""
    for name in list(model["materials"]):
        spec = model["materials"][name]
        with st.expander(f"{name}: {fm.describe(spec)}"):
            c1, c2 = st.columns(2)
            newname = c1.text_input("Name", value=name, key=f"{prefix}mn_{ver}_{name}")
            typ = c2.selectbox("Typ", list(TYPE_LABELS), index=list(TYPE_LABELS).index(spec["type"]), format_func=TYPE_LABELS.get,
                               key=f"{prefix}mt_{ver}_{name}")
            if typ != spec["type"]:
                model["materials"][name] = copy.deepcopy(DEFAULT_SPEC[typ])
                changed()
                st.rerun()
            if typ == "library":
                spec["name"] = st.selectbox("Bibliotheksmaterial", list(fm.LIBRARY), index=list(fm.LIBRARY).index(spec["name"]),
                                            key=f"{prefix}ml_{ver}_{name}", help="Gemessene Daten (n, k) bzw. Sellmeier-Formel, mit Gültigkeitsbereich.")
                r = fm.valid_range(spec)
                if r:
                    st.caption(f"Gültig von {r[0]:.0f} bis {r[1]:.0f} nm.")
            elif typ == "index":
                c1, c2 = st.columns(2)
                spec["n"] = float(c1.number_input("n", value=float(spec["n"]), step=0.1, format="%.5g", key=f"{prefix}mi_n_{ver}_{name}"))
                spec["k"] = float(c2.number_input("k (Extinktion, ≥ 0)", value=float(spec["k"]), min_value=0.0, step=0.1, format="%.5g",
                                                   key=f"{prefix}mi_k_{ver}_{name}"))
            elif typ == "eps":
                c1, c2 = st.columns(2)
                spec["re"] = float(c1.number_input("Re ε", value=float(spec["re"]), step=0.1, format="%.6g", key=f"{prefix}me_r_{ver}_{name}"))
                spec["im"] = float(c2.number_input("Im ε (Verlust > 0)", value=float(spec["im"]), step=0.1, format="%.6g", key=f"{prefix}me_i_{ver}_{name}",
                                                    help="Konvention exp(−iωt): Verlust bedeutet Im ε > 0."))
            elif typ == "drude":
                c1, c2, c3 = st.columns(3)
                spec["eps_inf"] = float(c1.number_input("ε∞", value=float(spec["eps_inf"]), step=0.1, format="%.5g", key=f"{prefix}md_e_{ver}_{name}"))
                spec["omega_p_eV"] = float(c2.number_input("ħωp (eV)", value=float(spec["omega_p_eV"]), min_value=0.01, step=0.1, format="%.5g",
                                                            key=f"{prefix}md_p_{ver}_{name}"))
                spec["gamma_eV"] = float(c3.number_input("ħγ (eV)", value=float(spec["gamma_eV"]), min_value=0.0, step=0.01, format="%.5g",
                                                          key=f"{prefix}md_g_{ver}_{name}"))
            else:
                txt = st.text_area("Zeilen: Wellenlänge (nm), n, k", value="\n".join(f"{a:g}, {b:g}, {c:g}" for a, b, c in spec["rows"]),
                                   key=f"{prefix}mtab_{ver}_{name}", height=120, help="Linear interpoliert, keine Extrapolation.")
                try:
                    rows = [[float(v) for v in line.replace(";", ",").split(",")] for line in txt.splitlines() if line.strip()]
                    if len(rows) >= 2 and all(len(r) == 3 for r in rows):
                        spec["rows"] = rows
                    else:
                        st.error("Mindestens zwei Zeilen mit je drei Zahlen.")
                except ValueError:
                    st.error("Zahlen nicht lesbar.")
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
                    rename(model, name, newname.strip())
                    changed()
                    st.rerun()
            in_use = name in used
            if b2.button("Material löschen", key=f"{prefix}mdel_{ver}_{name}", disabled=in_use or len(model["materials"]) <= 1,
                         help="Nur möglich, wenn das Material nirgends verwendet wird."):
                del model["materials"][name]
                changed()
                st.rerun()
    c1, c2, c3 = st.columns([3, 2, 2])
    lib_pick = c1.selectbox("Aus der Bibliothek hinzufügen", ["–"] + LIB_CHOICES, key=f"{prefix}libadd_{ver}")
    if c2.button("Bibliotheksmaterial hinzufügen", key=f"{prefix}libadd_btn_{ver}", disabled=lib_pick == "–"):
        model["materials"][unique_name(model, lib_pick)] = {"type": "library", "name": lib_pick}
        changed()
        st.rerun()
    if c3.button("Eigenes Material hinzufügen", key=f"{prefix}matadd_{ver}"):
        model["materials"][unique_name(model, "Material")] = copy.deepcopy(DEFAULT_SPEC["index"])
        changed()
        st.rerun()
