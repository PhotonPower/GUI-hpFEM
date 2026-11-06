"""Streamlit building blocks shared by the two modes of the app (periodic structures and bodies of revolution)."""
from __future__ import annotations

import copy
import io

import streamlit as st

import fem_materials as fm

TYPE_LABELS = {"library": "Bibliothek (hpfem)", "index": "n + ik (konstant)", "eps": "ε (konstant)", "drude": "Drude-Metall", "table": "Tabelle λ, n, k",
               "crystal": "Kristall (anisotrop)"}
CRYSTAL_LABELS = {"uniaxial": "einachsig (n_o, n_e, Achse)", "biaxial": "zweiachsig (nx, ny, nz, Euler-Winkel)", "tensor": "voller Tensor ε_ij"}
CRYSTAL_DEFAULT = {"uniaxial": {"type": "crystal", "kind": "uniaxial", "n_o": 1.658, "k_o": 0.0, "n_e": 1.486, "k_e": 0.0, "theta_deg": 45.0, "phi_deg": 0.0},
                   "biaxial": {"type": "crystal", "kind": "biaxial", "n": [1.53, 1.54, 1.55], "k": [0.0, 0.0, 0.0], "euler_deg": [0.0, 0.0, 0.0]},
                   "tensor": {"type": "crystal", "kind": "tensor", "re": [[2.25, 0.0, 0.0], [0.0, 2.25, 0.0], [0.0, 0.0, 2.25]],
                              "im": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}}
LIB_CHOICES = ["Si", "Ag", "Au", "Al", "GaAs", "MAPbI3", "SiO2", "TiO2", "water", "air", "vacuum"]
DEFAULT_SPEC = {"library": {"type": "library", "name": "SiO2"}, "index": {"type": "index", "n": 1.5, "k": 0.0},
                "eps": {"type": "eps", "re": 2.25, "im": 0.0}, "drude": {"type": "drude", "eps_inf": 1.0, "omega_p_eV": 9.0, "gamma_eV": 0.07},
                "table": {"type": "table", "rows": [[400.0, 1.5, 0.0], [800.0, 1.5, 0.0]]}, "crystal": CRYSTAL_DEFAULT["uniaxial"]}


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


def crystal_editor(spec, key, changed):
    """Parameters of a crystal spec (fem_materials, type "crystal"): kind, principal indices and orientation in the laboratory frame."""
    kind = st.selectbox("Kristallart", list(CRYSTAL_LABELS), index=list(CRYSTAL_LABELS).index(spec.get("kind", "uniaxial")), format_func=CRYSTAL_LABELS.get,
                        key=f"{key}_kind", help="Laborsystem: x quer zu den Gitterlinien, y längs der Linien, z die Normale (nach oben). "
                                                "Konstante Hauptindizes (keine Dispersion).")
    if kind != spec.get("kind"):
        spec.clear()
        spec.update(copy.deepcopy(CRYSTAL_DEFAULT[kind]))
        changed()
        st.rerun()
    if kind == "uniaxial":
        c = st.columns(4)
        spec["n_o"] = float(c[0].number_input("n_o (ordentlich)", value=float(spec["n_o"]), step=0.01, format="%.5g", key=f"{key}_no"))
        spec["k_o"] = float(c[1].number_input("k_o", value=float(spec["k_o"]), min_value=0.0, step=0.01, format="%.5g", key=f"{key}_ko"))
        spec["n_e"] = float(c[2].number_input("n_e (außerordentlich)", value=float(spec["n_e"]), step=0.01, format="%.5g", key=f"{key}_ne"))
        spec["k_e"] = float(c[3].number_input("k_e", value=float(spec["k_e"]), min_value=0.0, step=0.01, format="%.5g", key=f"{key}_ke"))
        c = st.columns(2)
        spec["theta_deg"] = float(c[0].number_input("Achse θ (°, gegen die Normale z)", value=float(spec["theta_deg"]), min_value=0.0, max_value=180.0,
                                                    step=5.0, key=f"{key}_th", help="θ = 0: c-Schnitt (Achse senkrecht zur Oberfläche)."))
        spec["phi_deg"] = float(c[1].number_input("Achse φ (°, gegen x quer zu den Linien)", value=float(spec["phi_deg"]), step=5.0, key=f"{key}_ph",
                                                  help="Azimut der Achse in der Ebene: φ = 90° legt sie längs der Linien."))
    elif kind == "biaxial":
        c = st.columns(3)
        for i, ax in enumerate("123"):
            spec["n"][i] = float(c[i].number_input(f"n{ax}", value=float(spec["n"][i]), step=0.01, format="%.5g", key=f"{key}_n{ax}"))
            spec["k"][i] = float(c[i].number_input(f"k{ax}", value=float(spec["k"][i]), min_value=0.0, step=0.01, format="%.5g", key=f"{key}_k{ax}"))
        c = st.columns(3)
        for i, lab in enumerate(("α (°, um z)", "β (°, um x')", "γ (°, um z'')")):
            spec["euler_deg"][i] = float(c[i].number_input(lab, value=float(spec["euler_deg"][i]), step=5.0, key=f"{key}_eu{i}"))
        st.caption("Hauptachsen 1, 2, 3 zunächst längs x, y, z; dann gedreht mit R = R_z(α) R_x(β) R_z(γ): ε = R diag(ε1, ε2, ε3) Rᵀ.")
    else:
        st.caption("Symmetrischer Tensor ε_ij (Real- und Imaginärteil, Verlust: positiv semidefiniter Imaginärteil). Eingabe des oberen Dreiecks; "
                   "das untere wird gespiegelt.")
        for part, lab in (("re", "Re"), ("im", "Im")):
            c = st.columns(6)
            for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
                v = float(c[k].number_input(f"{lab} ε_{'xyz'[i]}{'xyz'[j]}", value=float(spec[part][i][j]), step=0.1, format="%.5g",
                                            key=f"{key}_{part}{i}{j}"))
                spec[part][i][j] = spec[part][j][i] = v
    try:
        w = fm.principal_values(spec, 500.0)
        n = [complex(v) ** 0.5 for v in w]
        st.caption("Hauptindizes: " + ", ".join(f"{v.real:.4f}" + (f" + {v.imag:.4f} i" if abs(v.imag) > 1e-12 else "") for v in n))
    except Exception as exc:
        st.error(str(exc))


def material_editor(model, ver, lam0, used, rename, changed, prefix="", crystals=False):
    """Editor of model["materials"]: one expander per material (name, type, parameters, n and k at lam0, rename, delete) and the buttons to
    add a library or an own material. rename(model, old, new) renames everywhere; changed() is called before every st.rerun (bumps the widget
    version). crystals: offer the anisotropic type (only the periodic mode with hpfem.grating supports it)."""
    types = [t for t in TYPE_LABELS if crystals or t != "crystal"]
    for name in list(model["materials"]):
        spec = model["materials"][name]
        with st.expander(f"{name}: {fm.describe(spec)}"):
            c1, c2 = st.columns(2)
            newname = c1.text_input("Name", value=name, key=f"{prefix}mn_{ver}_{name}")
            opts = types if spec["type"] in types else types + [spec["type"]]
            typ = c2.selectbox("Typ", opts, index=opts.index(spec["type"]), format_func=TYPE_LABELS.get,
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
            elif typ == "crystal":
                crystal_editor(spec, f"{prefix}mc_{ver}_{name}", changed)
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
            if not fm.is_crystal(spec):                               # a crystal shows its principal indices in crystal_editor
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
