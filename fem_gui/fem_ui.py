"""Streamlit building blocks shared by the two modes of the app (periodic structures and bodies of revolution)."""
from __future__ import annotations

import copy
import io

import streamlit as st

import fem_materials as fm

TYPE_LABELS = {"library": "Bibliothek (hpfem)", "index": "n + ik (konstant)", "eps": "ε (konstant)", "drude": "Drude-Metall", "drude_lorentz": "Drude-Lorentz (Fit)", "table": "Tabelle λ, n, k",
               "crystal": "Kristall (anisotrop)"}
CRYSTAL_LABELS = {"uniaxial": "einachsig (n_o, n_e, Achse)", "biaxial": "zweiachsig (nx, ny, nz, Euler-Winkel)", "tensor": "voller Tensor ε_ij"}
CRYSTAL_DEFAULT = {"uniaxial": {"type": "crystal", "kind": "uniaxial", "n_o": 1.658, "k_o": 0.0, "n_e": 1.486, "k_e": 0.0, "theta_deg": 45.0, "phi_deg": 0.0},
                   "biaxial": {"type": "crystal", "kind": "biaxial", "n": [1.53, 1.54, 1.55], "k": [0.0, 0.0, 0.0], "euler_deg": [0.0, 0.0, 0.0]},
                   "tensor": {"type": "crystal", "kind": "tensor", "re": [[2.25, 0.0, 0.0], [0.0, 2.25, 0.0], [0.0, 0.0, 2.25]],
                              "im": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}}
LIB_CHOICES = list(fm.LIBRARY)
DEFAULT_SPEC = {"library": {"type": "library", "name": "SiO2"}, "index": {"type": "index", "n": 1.5, "k": 0.0},
                "eps": {"type": "eps", "re": 2.25, "im": 0.0}, "drude": {"type": "drude", "eps_inf": 1.0, "omega_p_eV": 9.0, "gamma_eV": 0.07},
                "drude_lorentz": {"type": "drude_lorentz", "eps_inf": 1.0, "omega_p_eV": 9.0, "gamma_eV": 0.07, "osc": [[0.5, 3.0, 0.5]]},
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


def crystal_editor(spec, key, changed, axi=False):
    """Parameters of a crystal spec (fem_materials, type "crystal"): kind, principal indices and orientation in the laboratory frame."""
    kind = st.selectbox("Kristallart", list(CRYSTAL_LABELS), index=list(CRYSTAL_LABELS).index(spec.get("kind", "uniaxial")), format_func=CRYSTAL_LABELS.get,
                        key=f"{key}_kind", help=(fm.AXI_HELP if axi else "Laborsystem: x quer zu den Gitterlinien, y längs der Linien, z die Normale "
                                                 "(nach oben). ") + " Konstante Hauptindizes (keine Dispersion).")
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
    if axi:
        try:
            fm.axi_diagonal(spec)
            st.caption("In (r, φ, z): ε_r, ε_φ, ε_z = " + ", ".join(f"{complex(v).real:.4f}" for v in fm.axi_diagonal(spec)) + " (x → r, y → φ, z → z).")
        except ValueError as exc:
            st.error(str(exc))
    try:
        w = fm.principal_values(spec, 500.0)
        n = [complex(v) ** 0.5 for v in w]
        st.caption("Hauptindizes: " + ", ".join(f"{v.real:.4f}" + (f" + {v.imag:.4f} i" if abs(v.imag) > 1e-12 else "") for v in n))
    except Exception as exc:
        st.error(str(exc))


def drude_lorentz_editor(spec, key, lam0, fit=None):
    """Parameters of a drude_lorentz spec (ε∞, ħωp, ħγ, oscillators f / ħω_j / ħγ_j in eV) and, if ``fit`` is given, the fit of the model to
    library / table data (see fit_editor)."""
    c1, c2, c3 = st.columns(3)
    spec["eps_inf"] = float(c1.number_input("ε∞", value=float(spec["eps_inf"]), step=0.1, format="%.5g", key=f"{key}_e"))
    spec["omega_p_eV"] = float(c2.number_input("ħωp (eV)", value=float(spec["omega_p_eV"]), min_value=0.0, step=0.1, format="%.5g", key=f"{key}_p",
                                                help="Plasmafrequenz des Drude-Terms; 0 schaltet den Drude-Term und die Oszillatorstärken ab."))
    spec["gamma_eV"] = float(c3.number_input("ħγ (eV)", value=float(spec["gamma_eV"]), min_value=0.0, step=0.01, format="%.5g", key=f"{key}_g"))
    txt = st.text_area("Lorentz-Oszillatoren: Stärke f, ħω_j (eV), ħγ_j (eV) je Zeile", value="\n".join(f"{a:g}, {b:g}, {c:g}" for a, b, c in spec.get("osc", [])),
                       key=f"{key}_o", height=90, help="ε = ε∞ − ωp²/(ω² + iγω) + Σ f_j ωp²/(ω_j² − ω² − iγ_j ω), Konvention exp(−iωt). "
                       "Leer lassen für ein reines Drude-Metall.")
    try:
        rows = [[float(v) for v in line.replace(";", ",").split(",")] for line in txt.splitlines() if line.strip()]
        if all(len(r) == 3 and r[0] >= 0 and r[1] > 0 and r[2] >= 0 for r in rows):
            spec["osc"] = rows
        else:
            st.error("Je Zeile drei Zahlen (f ≥ 0, ħω_j > 0, ħγ_j ≥ 0).")
    except ValueError:
        st.error("Zahlen nicht lesbar.")


def fit_editor(model, name, spec, key, lam0, fit, changed):
    """Expander: Drude-Lorentz-Fit of a library / table / Drude material over a wavelength range; the result can replace the material (the fit is
    analytic in ω, so resonances can be made self-consistent) or be added as a new material."""
    with st.expander("An Messdaten anpassen (Drude-Lorentz)"):
        c1, c2, c3 = st.columns(3)
        lo = float(c1.number_input("λ von (nm)", 100.0, 20000.0, float(max(200.0, round(lam0 * 0.7))), 10.0, key=f"{key}_lo"))
        hi = float(c2.number_input("λ bis (nm)", 100.0, 20000.0, float(round(lam0 * 1.4)), 10.0, key=f"{key}_hi"))
        n_osc = int(c3.number_input("Oszillatoren", 0, 6, 2, key=f"{key}_n", help="Mehr Oszillatoren passen enger an, können aber überanpassen."))
        if lo >= hi:
            st.error("λ von muss kleiner als λ bis sein.")
            return
        if st.button("Anpassen", key=f"{key}_go"):
            with st.spinner("Anpassung läuft …"):
                try:
                    st.session_state[f"{key}_res"] = fit(spec, lo, hi, n_osc)
                except Exception as exc:
                    st.session_state.pop(f"{key}_res", None)
                    st.error(f"Anpassung nicht möglich: {exc}")
        r = st.session_state.get(f"{key}_res")
        if r:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.2, 3.2))
            ax.plot(r["lam_nm"], r["eps_ref_re"], "C0.", label="Re ε Daten")
            ax.plot(r["lam_nm"], r["eps_fit_re"], "C0-", label="Re ε Fit")
            ax.plot(r["lam_nm"], r["eps_ref_im"], "C3.", label="Im ε Daten")
            ax.plot(r["lam_nm"], r["eps_fit_im"], "C3-", label="Im ε Fit")
            ax.set_xlabel("Wellenlänge (nm)")
            ax.set_ylabel("ε")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Größter relativer Fehler von ε auf den Stützstellen: {100 * r['max_rel_error']:.2f} %. {fm.describe(r['spec'])}")
            b1, b2 = st.columns(2)
            if b1.button("Als neues Material hinzufügen", key=f"{key}_add"):
                model["materials"][unique_name(model, f"{name} (Fit)")] = copy.deepcopy(r["spec"])
                st.session_state.pop(f"{key}_res", None)
                changed()
                st.rerun()
            if b2.button("Dieses Material ersetzen", key=f"{key}_rep"):
                model["materials"][name] = copy.deepcopy(r["spec"])
                st.session_state.pop(f"{key}_res", None)
                changed()
                st.rerun()


def material_editor(model, ver, lam0, used, rename, changed, prefix="", crystals=False, axi=False, fit=None):
    """Editor of model["materials"]: one expander per material (name, type, parameters, n and k at lam0, rename, delete) and the buttons to
    add a library or an own material. rename(model, old, new) renames everywhere; changed() is called before every st.rerun (bumps the widget
    version). crystals: offer the anisotropic type (the periodic mode with hpfem.grating and the bodies of revolution take it; axi: the tensor is read in
    (r, φ, z) and must be diagonal there), with templates (constant principal indices at λ0). fit(spec, lo, hi, n_osc): Drude-Lorentz fit (fem_run.fit_material),
    offered for library / table / Drude materials."""
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
            elif typ == "drude_lorentz":
                drude_lorentz_editor(spec, f"{prefix}mdl_{ver}_{name}", lam0)
            elif typ == "crystal":
                crystal_editor(spec, f"{prefix}mc_{ver}_{name}", changed, axi=axi)
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
            if fit is not None and (typ in ("table", "drude", "drude_lorentz") or (typ == "library" and spec["name"] not in ("air", "vacuum"))):
                fit_editor(model, name, spec, f"{prefix}mfit_{ver}_{name}", lam0, fit, changed)
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
    if crystals:
        c1, c2 = st.columns([3, 2])
        cr_pick = c1.selectbox("Kristall-Vorlage hinzufügen (konstante Indizes bei λ0, Achse längs z)", ["–"] + list(fm.CRYSTAL_PRESETS), key=f"{prefix}cradd_{ver}",
                               help="Hauptindizes n_o, n_e bei der in der Vorlage genannten Wellenlänge, ohne Dispersion; die optische Achse liegt längs z "
                                    "(c-Schnitt). Winkel und Indizes lassen sich danach im Material ändern.")
        if c2.button("Kristall hinzufügen", key=f"{prefix}cradd_btn_{ver}", disabled=cr_pick == "–"):
            model["materials"][unique_name(model, cr_pick.split(" (")[0])] = fm.crystal_preset(cr_pick)
            changed()
            st.rerun()
