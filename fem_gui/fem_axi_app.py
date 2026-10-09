"""Mode „Rotationskörper: Resonator, Emitter“ of the FEM model builder (Streamlit). Called from fem_app.py: sidebar_model() in the sidebar,
render() for the page. Model, mesh and figures in fem_axi.py, the solver in fem_axi_worker.py (runs in the Python with hpfem)."""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import fem_axi as fa
import fem_materials as fm
import fem_run as fr
import fem_ui as ui

S = st.session_state


# ------------------------------------------------------------------------------------------------------------------ state
def init():
    if "ax_model" in S:
        return
    S.ax_model = copy.deepcopy(next(iter(fa.presets().values())))
    S.ax_ver = 0
    S.ax_ms = dict(cells_per_wavelength=4.0, skin_cells=1.0, interface_factor=0.7, curved=True, pml_cells_per_wavelength=6.0)
    S.ax_solver = dict(order=2, backend="AUTO", subdivisions=2, fields=True)
    S.ax_task = "resonance"
    S.ax_mesh = None


def bump():
    S.ax_ver += 1


def set_model(m):
    S.ax_model = fa.complete(copy.deepcopy(m))
    bump()
    S.ax_mesh = None
    S.pop("ax_view_dir", None)


def used(model):
    u = {model["background"]} | ({model["substrate"]} if model.get("substrate") else set())
    return u | {p["material"] for p in model["parts"]} | {ly["material"] for ly in model.get("layers", [])}


def rename(model, old, new):
    model["materials"] = {(new if k == old else k): v for k, v in model["materials"].items()}
    for key in ("background", "substrate"):
        if model.get(key) == old:
            model[key] = new
    for p in model["parts"] + model.get("layers", []):
        if p["material"] == old:
            p["material"] = new


def mesh_range(model):
    rs = [fa.wavelength_range(model, t) for t in ("resonance", "emitter", "scattering")]
    return min(r[0] for r in rs), max(r[1] for r in rs)


def mesh_task(model):
    """The mesh carries the emitter box when the emitter settings are valid (then it serves both tasks)."""
    return "emitter" if not [t for lvl, t in fa.validate(model, "emitter") if lvl == "error"] else "resonance"


def signature(model, ms):
    keep = {k: model.get(k) for k in ("materials", "background", "substrate", "layers", "parts", "domain")}
    keep["emitter"] = {k: model["emitter"][k] for k in ("z_nm", "sigma_nm")}
    keep["scattering"] = model["scattering"]["sweep"]
    return json.dumps([keep, ms, mesh_range(model)], sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------------------------------- sidebar
def sidebar_model():
    init()
    presets = fa.presets()
    choice = st.sidebar.selectbox("Vorlage", ["– auswählen –"] + list(presets), key="ax_preset_choice",
                                  help="Beispiele: Mikrosäule mit Quantenpunkt (wie das Beispiel micropillar_qd von hp-FEM), dielektrische Kugel, "
                                       "Gold-Nanokugel mit Emitter, Mikroscheibe.")
    if st.sidebar.button("Vorlage laden", key="ax_load_preset", disabled=choice.startswith("–")):
        set_model(presets[choice])
        st.rerun()
    up = st.sidebar.file_uploader("Modell laden (JSON)", type=["json"], key="ax_model_upload")
    if up is not None and S.get("ax_loaded_upload") != (up.name, up.size):
        try:
            set_model(fa.load_model(up.getvalue().decode("utf-8")))
            S.ax_loaded_upload = (up.name, up.size)
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Datei nicht lesbar: {exc}")
    st.sidebar.download_button("Modell speichern (JSON)", json.dumps(S.ax_model, indent=1, ensure_ascii=False).encode("utf-8"),
                               file_name="rotationskoerper.json", mime="application/json", key="ax_save_model")


# ------------------------------------------------------------------------------------------------------------------ parts
def _move(i, d):
    parts = S.ax_model["parts"]
    j = i + d
    if 0 <= j < len(parts):
        parts[i], parts[j] = parts[j], parts[i]
        bump()


def _delete(i):
    del S.ax_model["parts"][i]
    bump()


def _duplicate(i):
    S.ax_model["parts"].insert(i + 1, copy.deepcopy(S.ax_model["parts"][i]))
    bump()


def _layer_delete(i):
    del S.ax_model["layers"][i]
    bump()


def _layer_move(i, d):
    ls = S.ax_model["layers"]
    j = i + d
    if 0 <= j < len(ls):
        ls[i], ls[j] = ls[j], ls[i]
        bump()


def _layer_add():
    m = S.ax_model
    names = list(m["materials"])
    top = max([ly["z_bottom"] + ly["height"] for ly in m["layers"]], default=0.0)
    m["layers"].append(dict(material=next((n for n in names if n != m["background"]), names[0]), z_bottom=float(top), height=100.0))
    bump()


def _add(kind):
    m = S.ax_model
    names = list(m["materials"])
    mat = next((n for n in names if n != m["background"]), names[0])
    m["parts"].append(fa.new_part(kind, mat))
    bump()


# ------------------------------------------------------------------------------------------------------------------- page
def render(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    init()
    fa.complete(S.ax_model)
    model, ver = S.ax_model, S.ax_ver
    work = Path(ctx["work"]) / "axi"
    st.title("FEM-Modellwerkstatt: Rotationskörper")
    st.caption("Resonatoren, Emitter und Partikel mit Rotationssymmetrie um die z-Achse (Mikrosäulen, Kugeln, Scheiben, Ringe): jede Azimutordnung m "
               "ist ein 2D-Problem in der Meridianebene (r, z). Resonanzen (komplexe Frequenz, Güte Q), Emission eines Dipols auf der Achse "
               "(Purcell-Faktor, Abstrahlung, Modenzerlegung) und Streuung ebener Wellen (Querschnitte, Streudiagramm, Nahfeld).")
    t1, t2, t3, t4, t5 = st.tabs(["1 Modell", "2 Netz", "3 Rechnung", "4 Ergebnisse", "5 Info"])
    with t1:
        tab_model(model, ver)
    with t2:
        tab_mesh(model, work)
    with t3:
        tab_run(model, work, ctx)
    with t4:
        tab_results(work)
    with t5:
        tab_info()


def tab_model(model, ver):
    left, right = st.columns([5, 4])
    em, rs, dom = model["emitter"], model["resonance"], model["domain"]
    with left:
        model["name"] = st.text_input("Name des Modells", value=model["name"], key=f"ax_name_{ver}")
        st.subheader("Materialien")
        ui.material_editor(model, ver, float(rs["wavelength_nm"]), used(model), rename, bump, prefix="ax_")
        names = list(model["materials"])
        st.subheader("Umgebung und Substrat")
        c = st.columns(2)
        model["background"] = c[0].selectbox("Umgebung (verlustfrei)", names, index=names.index(model["background"]) if model["background"] in names else 0,
                                             key=f"ax_bg_{ver}", help="Medium um die Struktur, in das abgestrahlt wird (Luft, Wasser, Glas …).")
        subs = ["– kein Substrat –"] + names
        cur = model.get("substrate") or "– kein Substrat –"
        pick = c[1].selectbox("Substrat (Halbraum z < 0)", subs, index=subs.index(cur) if cur in subs else 0, key=f"ax_sub_{ver}",
                              help="Füllt z < 0 über den ganzen Radius bis in die PML (z. B. der GaAs-Wafer unter einer Mikrosäule).")
        model["substrate"] = None if pick.startswith("–") else pick

        st.subheader("Schichten (radial unendlich, bis in die PML)")
        st.caption("Planare Schichten wie das Substrat: sie laufen über den ganzen Radius durch die PML bis zur Wand (Bragg-Spiegel, Membranen, "
                   "Schichtwellenleiter). In ihnen geführte Leistung wird in der PML absorbiert und zählt beim Emitter zum seitlichen Anteil. Eine spätere "
                   "Schicht überdeckt frühere und das Substrat; die Teile liegen darüber. Nicht für die Streuung ebener Wellen.")
        for i, ly in enumerate(model["layers"]):
            c = st.columns([3, 2, 2, 1, 1, 1])
            ly["material"] = c[0].selectbox(f"Schicht {i + 1}", names, index=names.index(ly["material"]) if ly["material"] in names else 0,
                                            key=f"ax_ly_m_{i}_{ver}")
            ly["z_bottom"] = float(c[1].number_input("Unterkante z (nm)", value=float(ly["z_bottom"]), step=10.0, format="%.6g", key=f"ax_ly_z_{i}_{ver}"))
            ly["height"] = float(c[2].number_input("Dicke (nm)", value=float(ly["height"]), min_value=0.1, step=10.0, format="%.6g", key=f"ax_ly_h_{i}_{ver}"))
            c[3].button("↑", key=f"ax_ly_up_{i}_{ver}", on_click=_layer_move, args=(i, -1), help="früher (wird überdeckt)")
            c[4].button("↓", key=f"ax_ly_dn_{i}_{ver}", on_click=_layer_move, args=(i, 1), help="später (überdeckt)")
            c[5].button("✕", key=f"ax_ly_del_{i}_{ver}", on_click=_layer_delete, args=(i,), help="Schicht entfernen")
        st.button("Schicht hinzufügen", key=f"ax_ly_add_{ver}", on_click=_layer_add)

        st.subheader("Struktur (Teile im Querschnitt r ≥ 0, z)")
        st.caption("Ein späteres Teil überdeckt frühere und das Substrat. Alle Teile sind Rotationskörper um die z-Achse.")
        for i, p in enumerate(model["parts"]):
            with st.expander(f"Teil {i + 1}: {fa.PART_TYPES[p['type']][0]} aus {p['material']}", expanded=False):
                p["material"] = st.selectbox("Material", names, index=names.index(p["material"]) if p["material"] in names else 0, key=f"ax_pm_{i}_{ver}")
                if p["type"] == "polygon":
                    txt = st.text_area("Eckpunkte (eine Zeile pro Punkt: r, z in nm)", value="\n".join(f"{r:g}, {z:g}" for r, z in p["points"]),
                                       key=f"ax_pp_{i}_{ver}", height=130)
                    try:
                        pts = [[float(v) for v in line.replace(";", ",").split(",")] for line in txt.splitlines() if line.strip()]
                        if len(pts) >= 3 and all(len(q) == 2 for q in pts):
                            p["points"] = pts
                        else:
                            st.error("Mindestens drei Punkte mit je zwei Zahlen.")
                    except ValueError:
                        st.error("Zahlen nicht lesbar.")
                else:
                    cols = st.columns(2)
                    for k, (key, label) in enumerate(fa.PART_TYPES[p["type"]][1]):
                        p[key] = float(cols[k % 2].number_input(label, value=float(p[key]), step=10.0, format="%.6g", key=f"ax_p{key}_{i}_{ver}"))
                c = st.columns(4)
                c[0].button("↑ früher", key=f"ax_up_{i}_{ver}", on_click=_move, args=(i, -1))
                c[1].button("↓ später", key=f"ax_dn_{i}_{ver}", on_click=_move, args=(i, 1))
                c[2].button("Duplizieren", key=f"ax_dup_{i}_{ver}", on_click=_duplicate, args=(i,))
                c[3].button("Löschen", key=f"ax_del_{i}_{ver}", on_click=_delete, args=(i,))
        c1, c2 = st.columns([3, 2])
        kind = c1.selectbox("Teil hinzufügen", list(fa.PART_TYPES), format_func=lambda k: fa.PART_TYPES[k][0], key=f"ax_add_{ver}")
        c2.button("Hinzufügen", key=f"ax_add_btn_{ver}", on_click=_add, args=(kind,))
        with st.expander("Generator: Mikrosäule mit Bragg-Spiegeln (DBR)"):
            c = st.columns(3)
            lam = c[0].number_input("Entwurfswellenlänge (nm)", 100.0, 20000.0, float(rs["wavelength_nm"]), 10.0, key=f"ax_g_lam_{ver}")
            radius = c[1].number_input("Radius der Säule (nm)", 50.0, 50000.0, 750.0, 50.0, key=f"ax_g_r_{ver}")
            cav = c[2].number_input("Kavität (Wellenlängen im Material)", 0.5, 10.0, 1.0, 0.5, key=f"ax_g_cav_{ver}")
            c = st.columns(4)
            hi = c[0].selectbox("hochbrechend", names, index=names.index("GaAs") if "GaAs" in names else 0, key=f"ax_g_hi_{ver}")
            lo = c[1].selectbox("niedrigbrechend", names, index=names.index("AlAs") if "AlAs" in names else min(1, len(names) - 1), key=f"ax_g_lo_{ver}")
            top = c[2].number_input("Paare oben", 0, 60, 6, key=f"ax_g_top_{ver}")
            bot = c[3].number_input("Paare unten", 0, 60, 10, key=f"ax_g_bot_{ver}")
            try:
                n_hi, n_lo = fm.nk(model["materials"][hi], lam)[0], fm.nk(model["materials"][lo], lam)[0]
                st.caption(f"n = {n_hi:.3f} / {n_lo:.3f} bei {lam:g} nm: λ/4-Schichten {lam / 4 / n_hi:.1f} / {lam / 4 / n_lo:.1f} nm.")
            except Exception as exc:
                n_hi = n_lo = None
                st.error(str(exc))
            planar = st.checkbox("Unteren Spiegel als planare Schichten (radial unendlich, nur die Kavität und der obere Spiegel sind geätzt)",
                                 key=f"ax_g_planar_{ver}")
            if st.button("Säule erzeugen (ersetzt alle Teile, setzt Emitter in die Kavitätsmitte)", key=f"ax_g_btn_{ver}", disabled=n_hi is None):
                parts, centre, height = fa.dbr_pillar(int(top), int(bot), radius, lam, hi, lo, n_hi, n_lo, cav)
                if planar:
                    k = 2 * int(bot)
                    model["layers"] = [dict(material=q["material"], z_bottom=q["z_bottom"], height=q["height"]) for q in parts[:k]]
                    parts = parts[k:]
                model["parts"] = parts
                em["z_nm"] = round(centre, 4)
                rs["wavelength_nm"] = float(lam)
                bump()
                st.rerun()

        st.subheader("Resonanzsuche")
        c = st.columns(3)
        rs["wavelength_nm"] = float(c[0].number_input("Zielwellenlänge (nm)", 1.0, 100000.0, float(rs["wavelength_nm"]), 10.0, format="%.6g",
                                                      key=f"ax_rl_{ver}", help="Die Moden mit der komplexen Frequenz am nächsten an diesem Ziel."))
        rs["m"] = int(c[1].number_input("Azimutordnung m", 0, 400, int(rs["m"]), key=f"ax_rm_{ver}", help=fa.m_hint(model)))
        rs["num_modes"] = int(c[2].number_input("Anzahl Moden", 1, 40, int(rs["num_modes"]), key=f"ax_rn_{ver}"))
        wm = fa.whispering_m(model, rs["wavelength_nm"])
        if wm and wm > 4:
            st.caption(f"Grobe Schätzung für Flüstergalerie-Moden des äußersten Teils bei dieser Wellenlänge: m ≈ {wm}.")

        st.subheader("Emitter (Dipol auf der Achse, z. B. Quantenpunkt)")
        c = st.columns(3)
        em["z_nm"] = float(c[0].number_input("Position z (nm)", value=float(em["z_nm"]), step=10.0, format="%.6g", key=f"ax_ez_{ver}"))
        em["orientation"] = c[1].selectbox("Dipolrichtung", list(fa.ORIENTATIONS), index=list(fa.ORIENTATIONS).index(em["orientation"]),
                                           format_func=fa.ORIENTATIONS.get, key=f"ax_eo_{ver}",
                                           help="Senkrecht zur Achse: koppelt an Moden mit m = ±1 (Grundmode einer Säule); entlang der Achse: m = 0.")
        em["sigma_nm"] = float(c[2].number_input("Ausdehnung σ (nm)", 0.5, 500.0, float(em["sigma_nm"]), 1.0, format="%.4g", key=f"ax_es_{ver}",
                                                 help="Gauß-verschmierter Punktdipol (Größe eines Quantenpunkts). Der Kasten 4σ um den Emitter muss in "
                                                      "einem verlustfreien Material liegen."))
        sw = em["sweep"]
        SWM = {"resonance": "um die Resonanz (± Linienbreiten)", "fixed": "fester Bereich"}
        c = st.columns(4)
        sw["mode"] = c[0].selectbox("Spektrum", list(SWM), index=list(SWM).index(sw["mode"]), format_func=SWM.get, key=f"ax_swm_{ver}",
                                    help="„Um die Resonanz“: sucht zuerst die Mode der passenden Ordnung (m = 1 bzw. 0) nahe der Zielwellenlänge mit dem "
                                         "größten Q und legt das Spektrum symmetrisch um sie.")
        if sw["mode"] == "fixed":
            sw["start"] = float(c[1].number_input("von (nm)", value=float(sw["start"]), format="%.6g", key=f"ax_swa_{ver}"))
            sw["stop"] = float(c[2].number_input("bis (nm)", value=float(sw["stop"]), format="%.6g", key=f"ax_swb_{ver}"))
        else:
            sw["linewidths"] = float(c[1].number_input("± Linienbreiten λ/Q", 0.1, 50.0, float(sw.get("linewidths", 1.5)), 0.5, key=f"ax_swl_{ver}"))
        sw["n"] = int(c[3].number_input("Punkte", 1, 400, int(sw["n"]), key=f"ax_swn_{ver}"))
        em["modal"] = st.checkbox("Zusätzlich Modenzerlegung (Riesz-Projektion): Anteil der Resonanz und Hintergrund am Purcell-Faktor",
                                  value=bool(em.get("modal")) and sw["mode"] == "resonance", disabled=sw["mode"] != "resonance", key=f"ax_modal_{ver}",
                                  help="Schreibt das Spektrum als Summe über die Quasi-Normalmoden der Resonanzsuche (hpfem.AxisymmetricRieszProjection, "
                                       "PML bei der Zielwellenlänge eingefroren). Braucht das Spektrum „um die Resonanz“. Teuer: etwa 16 Faktorisierungen je Pol "
                                       "und 40 für den Hintergrund (Mikrosäule, schnell, p = 2: etwa 15 min). Die Summe folgt der direkten Rechnung "
                                       "auf wenige Prozent.")

        st.subheader("Ebene Welle (Streuung an Partikeln)")
        sc = model["scattering"]
        c = st.columns(4)
        sc["theta_deg"] = float(c[0].number_input("Einfallswinkel θ gegen +z (°)", 0.0, 180.0, float(sc["theta_deg"]), 5.0, key=f"ax_sc_th_{ver}",
                                                  help="0°: entlang der Achse (nur m = ±1 nötig); schräg: die Welle zerfällt in die Ordnungen m = 0, ±1, ±2 …"))
        sc["pol"] = c[1].selectbox("Polarisation", ["S", "P"], index=["S", "P"].index(sc["pol"]), key=f"ax_sc_pol_{ver}",
                                   format_func={"S": "S (E entlang y, senkrecht zur Einfallsebene)", "P": "P (E in der Einfallsebene x-z)"}.get)
        sc["max_order"] = int(c[2].number_input("höchste Ordnung |m|", 1, 60, int(sc["max_order"]), key=f"ax_sc_m_{ver}",
                                                help="Die Summe stoppt früher, wenn das Paar ±m weniger als 10⁻⁵ der Streuleistung trägt. Faustregel: k·R + 4."))
        c[3].caption("Die Welle läuft in Richtung (sin θ, 0, cos θ) durch die Umgebung; nur ohne Substrat.")
        c = st.columns(3)
        ssw = sc["sweep"]
        ssw["start"] = float(c[0].number_input("λ von (nm)", value=float(ssw["start"]), format="%.6g", key=f"ax_sc_a_{ver}"))
        ssw["stop"] = float(c[1].number_input("λ bis (nm)", value=float(ssw["stop"]), format="%.6g", key=f"ax_sc_b_{ver}"))
        ssw["n"] = int(c[2].number_input("Punkte ", 1, 400, int(ssw["n"]), key=f"ax_sc_n_{ver}"))

        st.subheader("Rechengebiet")
        lam_hi = mesh_range(model)[1]
        if st.button("Vorschlag aus der Wellenlänge übernehmen", key=f"ax_sugg_{ver}", help="Abstand 0,75 λ, PML 1,5 λ (wie das Beispiel micropillar_qd)."):
            dom["margin_nm"], dom["pml_nm"] = round(0.75 * lam_hi, 1), round(1.5 * lam_hi, 1)
            bump()
            st.rerun()
        c = st.columns(2)
        dom["margin_nm"] = float(c[0].number_input("Abstand Struktur → PML (nm)", 10.0, 1e6, float(dom["margin_nm"]), 50.0, format="%.6g",
                                                   key=f"ax_dm_{ver}", help="Rundum: außen, oben und unten. Die Messebenen für β liegen in der Mitte."))
        dom["pml_nm"] = float(c[1].number_input("PML-Dicke (nm)", 10.0, 1e6, float(dom["pml_nm"]), 50.0, format="%.6g", key=f"ax_dp_{ver}",
                                                help="Zylindrische PML außen, oben und unten; dahinter eine Metallwand."))
    with right:
        for lvl, text in fa.validate(model, "emitter"):
            (st.error if lvl == "error" else st.warning)(text)
        try:
            ui.show_fig(fa.preview_figure(model), "rotationskoerper.png", f"ax_dl_prev_{ver}")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")
        lam0 = float(rs["wavelength_nm"])
        rows = []
        for n_, spec in model["materials"].items():
            try:
                e = fm.eps_at(spec, lam0)
                rows.append({"Material": n_, "n": round(float(np.sqrt(e + 0j).real), 4), "k": round(float(abs(np.sqrt(e + 0j).imag)), 4),
                             "verwendet": "ja" if n_ in used(model) else "nein"})
            except Exception as exc:
                rows.append({"Material": n_, "n": None, "k": None, "verwendet": str(exc)[:50]})
        ui.table_show(pd.DataFrame(rows))
        try:
            st.caption(f"Emitter in {fa.material_at(model, 0.0, em['z_nm'])}; Kasten um den Emitter ±{fa.source_box(model):g} nm.")
        except Exception:
            pass


def tab_mesh(model, work):
    ms = S.ax_ms
    st.markdown("Die Meridianebene (r ≥ 0, z) wird mit Gmsh in Dreiecke zerlegt: Elementgröße nach der Wellenlänge im Material, Verfeinerung an "
                "Grenzflächen und um den Emitter, gekrümmte Elemente für Kugeln, Ellipsoide und Tori. Die Achse r = 0 ist ein eigener Rand.")
    c = st.columns(4)
    ms["cells_per_wavelength"] = float(c[0].slider("Elemente pro Wellenlänge", 1.0, 12.0, float(ms["cells_per_wavelength"]), 0.5, key="ax_ms_cpw",
                                                   help="Mit p = 2 etwa 4 bis 6, mit p = 3 etwa 3 bis 4."))
    ms["skin_cells"] = float(c[1].slider("Elemente pro Eindringtiefe (Metall)", 0.5, 4.0, float(ms["skin_cells"]), 0.5, key="ax_ms_skin"))
    ms["interface_factor"] = float(c[2].slider("Verfeinerung an Grenzflächen", 0.2, 1.0, float(ms["interface_factor"]), 0.05, key="ax_ms_if"))
    ms["curved"] = c[3].checkbox("Gekrümmte Elemente", value=bool(ms["curved"]), key="ax_ms_curved")
    lam_lo, lam_hi = mesh_range(model)
    task = mesh_task(model)
    errors = [t for lvl, t in fa.validate(model, task) if lvl == "error"]
    sig = signature(model, ms)
    if task == "resonance":
        st.info("Die Emitter-Einstellungen sind ungültig (Reiter 1): das Netz bekommt keinen Emitterkasten und taugt nur für Resonanzen.")
    if st.button("Netz erzeugen", type="primary", key="ax_mesh_btn", disabled=bool(errors)):
        with st.spinner("Gmsh vernetzt …"):
            try:
                t0 = time.time()
                stats = fa.build_mesh(model, dict(ms, lam_min_nm=lam_lo, lam_max_nm=lam_hi), work / "mesh", task)
                stats["seconds"] = time.time() - t0
                S.ax_mesh = dict(stats=stats, sig=sig, dir=str(work / "mesh"), model=copy.deepcopy(model), err="")
            except Exception as exc:
                S.ax_mesh = dict(err=str(exc))
    if errors:
        st.error("Das Modell hat Fehler (Reiter 1): " + " | ".join(errors))
    mesh = S.ax_mesh
    if mesh and mesh.get("err"):
        st.error(mesh["err"])
    elif mesh:
        stats = mesh["stats"]
        if mesh["sig"] != sig:
            st.warning("Modell oder Netzeinstellungen wurden seit dem Vernetzen geändert: Netz neu erzeugen.")
        c = st.columns(5)
        c[0].metric("Dreiecke", f"{stats['cells']:,}".replace(",", "."))
        c[1].metric("Knoten", f"{stats['nodes']:,}".replace(",", "."))
        c[2].metric("Elementordnung", stats["order"])
        c[3].metric("Kante min / max", f"{stats['edge_min_nm']:.1f} / {stats['edge_max_nm']:.0f} nm")
        c[4].metric("Emitterkasten", "ja" if stats.get("source_box_nm") else "nein")
        p = int(S.ax_solver["order"])
        st.caption(f"Grobe Schätzung der Freiheitsgrade bei p = {p}: etwa {fa.est_dofs(stats['cells'], p):,}".replace(",", ".") +
                   f". Netz in {stats.get('seconds', 0):.1f} s.")
        c1, c2 = st.columns([2, 3])
        with c1:
            ui.table_show(pd.DataFrame([{"Material": k, "Dreiecke": v, "max. Kante (nm)": stats["sizes_nm"].get(k)} for k, v in stats["cells_per_material"].items()]))
        with c2:
            zm = st.radio("Ausschnitt", ["ganzes Gebiet", "Struktur"], horizontal=True, key="ax_mesh_zoom")
            lay = fa.layout(mesh["model"])
            zoom = (0.0, lay["r_struct"] * 1.2 + 50, lay["z_struct_bottom"] - 100, lay["z_struct_top"] + 100) if zm == "Struktur" else None
            ui.show_fig(fa.mesh_figure(Path(mesh["dir"]) / "mesh_plot.npz", mesh["model"], zoom=zoom), "netz_meridian.png", "ax_dl_mesh")
    else:
        st.info("Noch kein Netz erzeugt.")


def tab_run(model, work, ctx):
    sv = S.ax_solver
    TASKS = {"resonance": "Resonanzen (Eigenmoden der Ordnung m, Güte Q, Modenfelder)",
             "emitter": "Emitter: Purcell-Faktor und Abstrahlung über der Wellenlänge",
             "scattering": "Streuung einer ebenen Welle: Querschnitte, Streudiagramm, Nahfeld"}
    S.ax_task = st.radio("Aufgabe", list(TASKS), index=list(TASKS).index(S.ax_task), format_func=TASKS.get, key="ax_task_pick")
    c = st.columns(4)
    sv["order"] = int(c[0].slider("Polynomordnung p", 1, 6, int(sv["order"]), key="ax_p",
                                  help="p = 2 für schnelle Übersichten (Resonanzlage und Q schon gut), p = 3 für Purcell-Faktor und β: an der Mikrosäule liegt "
                                       "F_P mit p = 2 etwa 4 % zu hoch, mit p = 3 auf etwa 1 % am feinsten Vergleichswert."))
    B = ["AUTO", "SPARSE_LU", "MUMPS", "CUDSS"]
    sv["backend"] = c[1].selectbox("Löser (direkt)", B, index=B.index(sv["backend"]) if sv["backend"] in B else 0, key="ax_backend")
    sv["fields"] = c[2].checkbox("Felder speichern", value=bool(sv["fields"]), key="ax_fields",
                                 help="Resonanzen: das Feld jeder Mode; Emitter und Streuung: das Feld in der Mitte des Spektrums.")
    sv["subdivisions"] = int(c[3].slider("Unterteilung je Dreieck", 1, 4, int(sv["subdivisions"]), key="ax_sub",
                                         help="Feldkarten exakt auf den unterteilten Elementen; 2 genügt für p ≤ 3."))
    if S.ax_task == "emitter":
        sv["bulk_fem"] = st.checkbox("Purcell-Normierung numerisch auf demselben Netz (genauer, doppelte Rechenzeit)", value=bool(sv.get("bulk_fem", False)),
                                     key="ax_bulk_fem",
                                     help="P_bulk aus derselben Quelle auf demselben Netz mit dem Emittermaterial überall, statt der Larmor-Formel: der "
                                          "Diskretisierungsfehler der schmalen Gauß-Quelle kürzt sich heraus. An einer GaAs-Membran gegen die exakte "
                                          "Sommerfeld-Lösung: Abweichung 1,4·10⁻³ statt 5,7·10⁻³ (σ = 10 nm, p = 3).")
    errs = [t for lvl, t in fa.validate(model, S.ax_task) if lvl == "error"]
    mesh = S.ax_mesh
    mesh_ok = bool(mesh and not mesh.get("err") and mesh["sig"] == signature(model, S.ax_ms))
    if S.ax_task == "emitter" and mesh_ok and not mesh["stats"].get("source_box_nm"):
        mesh_ok = False
        st.warning("Das Netz hat keinen Emitterkasten: Emitter-Einstellungen korrigieren und das Netz neu erzeugen.")
    if errs:
        st.error("Vor der Rechnung beheben: " + " | ".join(errs))
    if not mesh_ok:
        st.warning("Es gibt kein aktuelles Netz (Reiter 2: „Netz erzeugen“).")
    if ctx["lib"].get("error"):
        st.warning("hpfem lässt sich im eingestellten Python nicht laden (Seitenleiste).")
    if mesh_ok:
        dofs = fa.est_dofs(mesh["stats"]["cells"], sv["order"])
        n = {"resonance": 1, "emitter": int(model["emitter"]["sweep"]["n"]), "scattering": int(model["scattering"]["sweep"]["n"])}[S.ax_task]
        st.caption(f"Etwa {dofs:,} Freiheitsgrade je Lösung".replace(",", ".") + (f", {n} Wellenlängen" if S.ax_task != "resonance" else "") +
                   (", je Wellenlänge eine Lösung pro Azimutordnung" if S.ax_task == "scattering" else "") +
                   (" plus eine Resonanzsuche" if S.ax_task == "emitter" and model["emitter"]["sweep"]["mode"] == "resonance" else "") + ".")
    if st.button("Rechnung starten", type="primary", key="ax_run_btn", disabled=bool(errs) or not mesh_ok):
        try:
            run_dir = work / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            for f in ("mesh.msh", "mesh_plot.npz"):
                shutil.copy(Path(mesh["dir"]) / f, run_dir / f)
            job = dict(model=copy.deepcopy(model), task=S.ax_task, solver=dict(order=sv["order"], backend=sv["backend"], bulk_fem=bool(sv.get("bulk_fem"))), subdivisions=sv["subdivisions"],
                       fields=bool(sv["fields"]), field_at_centre=bool(sv["fields"]))
            (run_dir / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
            S.run_proc = fr.start_worker(ctx["py"], ctx["repo"], run_dir, int(ctx["threads"]), script="fem_axi_worker.py")
            S.run_dir, S.run_msg, S.cancel_requested, S.run_kind = str(run_dir), "", False, "axi"
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("run_msg") and S.get("run_kind") == "axi":
        (st.success if S.get("run_ok", True) else st.error)(S.run_msg + " Die Ergebnisse stehen im Reiter 4.")
    rd = work / "run"
    if (rd / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(rd, tail=80))


def tab_results(work):
    arch = work / "archiv"
    c = st.columns([3, 2, 3])
    archives = sorted([p.name for p in arch.iterdir() if (p / "results.json").is_file()]) if arch.is_dir() else []
    pick = c[0].selectbox("Gespeicherte Rechnung öffnen", ["– aktuelle –"] + archives, key="ax_arch_pick")
    if c[1].button("Öffnen", key="ax_arch_open", disabled=pick.startswith("–")):
        S.ax_view_dir = str(arch / pick)
        st.rerun()
    view = Path(S.get("ax_view_dir", work / "run"))
    if not (view / "results.json").is_file():
        st.info("Noch keine Ergebnisse: Netz erzeugen (Reiter 2), dann rechnen (Reiter 3).")
        return
    res = json.loads((view / "results.json").read_text(encoding="utf-8"))
    job = json.loads((view / "job.json").read_text(encoding="utf-8"))
    rmodel, meta = job["model"], res["meta"]
    name_arch = c[2].text_input("Archivieren unter", value=time.strftime("%Y%m%d_%H%M") + "_" + "".join(ch for ch in rmodel["name"][:20] if ch.isalnum()),
                                key="ax_arch_name")
    if c[2].button("Rechnung archivieren", key="ax_arch_save"):
        arch.mkdir(parents=True, exist_ok=True)
        shutil.copytree(view, arch / name_arch, dirs_exist_ok=True)
        st.success(f"Gespeichert unter {arch / name_arch}")
    vi = meta.get("version_info") or {}
    st.caption(f"{meta['model']} · {dict(resonance='Resonanzen', emitter='Emitter', scattering='Streuung')[meta['task']]} · p = {meta['order']} · {meta['cells']} Dreiecke · "
               f"{meta['dofs']:,} Freiheitsgrade · Start {meta['started']}".replace(",", ".") + (f" · hpfem {vi['hpfem']}" if vi.get("hpfem") else ""))
    if res.get("error"):
        st.error(res["error"])
    if res.get("cancelled"):
        st.warning("Die Rechnung wurde abgebrochen; es fehlen Punkte.")
    rsn = res.get("resonance")
    if meta["task"] == "resonance":
        if not rsn:
            st.info("Keine Moden.")
            return
        modes = rsn["modes"]
        sub = st.tabs(["Moden", "Modenfelder"])
        with sub[0]:
            ui.show_fig(fa.fig_modes(modes, rsn["target_nm"]), "moden.png", "ax_dl_modes")
            df = pd.DataFrame([{"Mode": md["k"], "m": md["m"], "λ_res (nm)": md["lam_nm"], "Q": md["Q"], "Linienbreite λ/Q (nm)": md["lam_nm"] / md["Q"] if md["Q"] > 0 else None,
                                "Re ω (1/s)": md["omega"][0], "Im ω (1/s)": md["omega"][1], "Residuum": md["residual"]} for md in modes])
            ui.table_show(df)
            st.download_button("Moden (CSV)", ui.csv_bytes(df), file_name="moden.csv", mime="text/csv", key="ax_dl_modes_csv")
            if any(md["Q"] < 0 for md in modes):
                st.warning("Moden mit Q < 0 (Im ω > 0) sind nicht physikalisch (PML- oder numerische Moden).")
            st.caption(f"Moden der Azimutordnung m = {rsn['m']} nahe {rsn['target_nm']:g} nm, sortiert nach dem Abstand der komplexen Frequenz zum Ziel. "
                       "λ_res = 2πc / Re ω, Q = Re ω / (−2 Im ω). Eine Mode mit sehr großem Q und kleinem Feld in der Struktur ist meist eine Kasten- "
                       "oder PML-Mode: im Modenfeld prüfen. Rechenzeit " + f"{rsn.get('time_s', 0):.1f} s.")
        with sub[1]:
            files = sorted(view.glob("mode_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
            if not files:
                st.info("Keine Modenfelder gespeichert (Reiter 3: „Felder speichern“).")
            else:
                lab = {f.name: f"Mode {f.stem.split('_')[1]}: λ = {next(md['lam_nm'] for md in modes if md['k'] == int(f.stem.split('_')[1])):.3f} nm, "
                               f"Q = {next(md['Q'] for md in modes if md['k'] == int(f.stem.split('_')[1])):.4g}" for f in files}
                field_view(view, list(lab), lab.get, rmodel, "ax_mode")
    elif meta["task"] == "scattering":
        scattering_results(view, res, rmodel)
    else:
        pts = res.get("points", [])
        ok = [p for p in pts if "error" not in p]
        for p in pts:
            if "error" in p:
                st.error(f"λ = {p['lam_nm']:.3f} nm: {p['error']}")
        chosen = (rsn or {}).get("chosen")
        if chosen:
            st.info(f"Resonanz (m = {rsn['m']}): λ = {chosen['lam_nm']:.3f} nm, Q = {chosen['Q']:.4g}, Linienbreite {chosen['lam_nm'] / chosen['Q']:.3f} nm. "
                    "Das Spektrum liegt symmetrisch darum.")
        has_modal = any(p.get("modal") for p in ok)
        sub = st.tabs(["Purcell und Abstrahlung", "Feld", "Tabelle und Export"] + (["Modenzerlegung"] if has_modal else []))
        if has_modal:
            with sub[3]:
                ui.show_fig(fa.fig_modal(ok, chosen), "modenzerlegung.png", "ax_dl_modal")
                info = (rsn or {}).get("modal") or {}
                st.caption("Riesz-Projektion (hpfem.AxisymmetricRieszProjection): das Spektrum als Summe über die Quasi-Normalmoden der Resonanzsuche; die "
                           "Anteile summieren sich exakt zur Gesamtleistung. Unterschiede zur direkten Rechnung kommen von der bei der Zielwellenlänge "
                           "eingefrorenen PML (wenige Prozent). " + (f"{info.get('poles')} Pole im Hintergrundkontur, Konvergenz (Halbregel) "
                                                                     f"{info.get('convergence', 0):.1e}, {info.get('time_s', 0):.0f} s." if info else ""))
                ui.table_show(pd.DataFrame([{"λ (nm)": p["lam_nm"], "F_P direkt": p["purcell"], "F_P modal": p["modal"]["total"],
                                             "Resonanz": p["modal"]["mode"], "Hintergrund": p["modal"]["background"]} for p in ok if p.get("modal")]))
        with sub[0]:
            if ok:
                ui.show_fig(fa.fig_purcell(ok, chosen), "purcell.png", "ax_dl_purcell")
                best = max(ok, key=lambda p: p["purcell"])
                c = st.columns(4)
                c[0].metric("größter Purcell-Faktor", f"{best['purcell']:.3g}", delta=f"bei {best['lam_nm']:.2f} nm", delta_color="off")
                c[1].metric("β oben dort", f"{best['beta_top']:.3f}" if best.get("beta_top") is not None else "–")
                c[2].metric("nach unten dort", f"{best['beta_bottom']:.3f}" if best.get("beta_bottom") is not None else "–")
                c[3].metric("Brechzahl am Emitter", f"{best['n_emitter']:.3f}")
                st.caption("F_P = P / P_bulk: Leistung, die der Dipol in der Struktur abgibt (Fluss durch die geschlossene Fläche um den Emitterkasten), "
                           "geteilt durch die Leistung desselben verschmierten Dipols im unbegrenzten Material am Emitterort (Larmor, n P₀ e^{−(nkσ)²}). "
                           "Bei einem verlustbehafteten Nachbarn (Metall) enthält P auch die Verluste (Quenching); „abgestrahlt (oben + unten)“ "
                           "zählt nur, was durch die Messebenen geht. β: Anteil durch die obere Messebene (r bis zur PML), entspricht der Sammlung "
                           "nach oben ohne Begrenzung der numerischen Apertur.")
            else:
                st.info("Keine auswertbaren Punkte.")
        with sub[1]:
            files = sorted(view.glob("field_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
            if not files:
                st.info("Kein Feld gespeichert (Reiter 3: „Felder speichern“).")
            else:
                lab = {}
                for f in files:
                    i = int(f.stem.split("_")[1])
                    p = next((q for q in ok if q["index"] == i), None)
                    lab[f.name] = f"Punkt {i}" + (f": λ = {p['lam_nm']:.3f} nm, F_P = {p['purcell']:.3g}" if p else "")
                field_view(view, list(lab), lab.get, rmodel, "ax_field", default_log=True)
        with sub[2]:
            df = pd.DataFrame([{"Nr": p.get("index"), "λ (nm)": p["lam_nm"], "Purcell F_P": p.get("purcell"), "abgestrahlt / P_bulk": p.get("purcell_radiative"),
                                "β oben": p.get("beta_top"), "nach unten": p.get("beta_bottom"), "P gesamt (W)": p.get("P_total"), "P bulk (W)": p.get("P_bulk"),
                                "Zeit (s)": p.get("time_s"), "Fehler": p.get("error")} for p in pts])
            ui.table_show(df)
            st.download_button("Spektrum (CSV)", ui.csv_bytes(df), file_name="purcell.csv", mime="text/csv", key="ax_dl_csv")
            st.caption(f"Ordner: {view}")


def scattering_results(view, res, model):
    pts = res.get("points", [])
    ok = [p for p in pts if "error" not in p]
    for p in pts:
        if "error" in p:
            st.error(f"λ = {p['lam_nm']:.3f} nm: {p['error']}")
    if not ok:
        st.info("Keine auswertbaren Punkte.")
        return
    R = fa.layout(model)["r_struct"]
    sc = model["scattering"]
    sub = st.tabs(["Querschnitte", "Streudiagramm", "Nahfeld", "Tabelle und Export"])
    with sub[0]:
        ui.show_fig(fa.fig_cross_sections(ok, np.pi * R ** 2 if R > 0 else None, sc["theta_deg"], sc["pol"]), "querschnitte.png", "ax_dl_cs")
        if all(p.get("mie") for p in ok):
            dev = max(abs(p["sigma_ext"] - p["mie"]["sigma_ext"]) / max(p["mie"]["sigma_ext"], 1e-30) for p in ok)
            dsc = max(abs(p["sigma_sca"] - p["mie"]["sigma_sca"]) / max(p["mie"]["sigma_sca"], 1e-30) for p in ok)
            (st.success if max(dev, dsc) < 1e-2 else st.warning)(
                f"Einzelne Kugel: Vergleich mit der Mie-Reihe, größte relative Abweichung Streuung {dsc:.1e}, Extinktion {dev:.1e}.")
        st.caption("σ_sca: Leistung des Streufelds durch die geschlossene Messfläche um das Teilchen, geteilt durch die einfallende Intensität "
                   "n|E₀|²/(2Z₀). σ_ext aus dem optischen Theorem (Fernfeld in Vorwärtsrichtung), σ_abs = σ_ext − σ_sca (bei sehr kleiner Absorption "
                   "entsprechend ungenauer). Effizienz rechts bezogen auf π R² mit dem größten Radius der Struktur.")
    with sub[1]:
        idx = st.selectbox("Wellenlänge", [p["index"] for p in ok], format_func=lambda i: f"{next(q['lam_nm'] for q in ok if q['index'] == i):.2f} nm",
                           index=int(np.argmax([p["sigma_sca"] for p in ok])), key="ax_pat_pick")
        pt = next(q for q in ok if q["index"] == idx)
        ui.show_fig(fa.fig_pattern(pt), f"streudiagramm_{idx}.png", "ax_dl_pat")
        st.caption("Differentieller Streuquerschnitt dσ/dΩ = |F|² (|E₀| = 1) aus der Summe der Fernfelder aller Ordnungen, in der Einfallsebene (x-z) und "
                   f"senkrecht dazu (y-z). Gerechnete Ordnungen: {pt['orders']}.")
    with sub[2]:
        files = sorted(view.glob("scatter_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
        if not files:
            st.info("Kein Nahfeld gespeichert (Reiter 3: „Felder speichern“).")
        else:
            c = st.columns(4)
            pick = c[0].selectbox("Feld", [f.name for f in files], key="ax_sc_pick",
                                  format_func=lambda n: f"λ = {next((q['lam_nm'] for q in ok if q['index'] == int(n.split('_')[1].split('.')[0])), 0):.2f} nm")
            q = c[1].selectbox("Größe", list(fa.SCATTER_QUANTITIES), key="ax_sc_q")
            log = c[2].checkbox("logarithmisch", key="ax_sc_log")
            zs = c[3].checkbox("nur Struktur", value=True, key="ax_sc_zoom")
            lay = fa.layout(model)
            zoom = (lay["r_plane"], lay["z_plane_bottom"], lay["z_plane_top"]) if zs else None
            ui.show_fig(fa.fig_scatter_field(fa.load_field(view / pick), model, q, log=log, zoom=zoom), f"{Path(pick).stem}.png", "ax_dl_scf")
            st.caption("Schnitt in der Einfallsebene (y = 0) mit kartesischen Komponenten; rechts x = r (φ = 0), links x = −r (φ = π). Gesamtfeld = "
                       "Streufeld + einfallende Welle (Pfeil), |E₀| = 1.")
    with sub[3]:
        rows = []
        for p in ok:
            row = {"λ (nm)": p["lam_nm"], "σ_sca (µm²)": p["sigma_sca"] * 1e12, "σ_abs (µm²)": p["sigma_abs"] * 1e12, "σ_ext (µm²)": p["sigma_ext"] * 1e12}
            if R > 0:
                row["Q_ext"] = p["sigma_ext"] / (np.pi * (R * 1e-9) ** 2)
            if p.get("mie"):
                row.update({"Mie σ_sca (µm²)": p["mie"]["sigma_sca"] * 1e12, "Mie σ_abs (µm²)": p["mie"]["sigma_abs"] * 1e12,
                            "Mie σ_ext (µm²)": p["mie"]["sigma_ext"] * 1e12})
            row.update({"Ordnungen": len(p["orders"]), "Zeit (s)": p["time_s"]})
            rows.append(row)
        df = pd.DataFrame(rows)
        ui.table_show(df)
        st.download_button("Querschnitte (CSV)", ui.csv_bytes(df), file_name="querschnitte.csv", mime="text/csv", key="ax_dl_cs_csv")
        st.caption(f"Ordner: {view}")


def field_view(view, names, label, model, key, default_log=False):
    c = st.columns(4)
    pick = c[0].selectbox("Feld", names, format_func=label, key=f"{key}_pick")
    q = c[1].selectbox("Größe", list(fa.AXI_QUANTITIES), key=f"{key}_q")
    log = c[2].checkbox("logarithmisch", value=default_log, key=f"{key}_log")
    zoom_struct = c[3].checkbox("nur Struktur", value=True, key=f"{key}_zoom")
    d = fa.load_field(view / pick)
    lay = fa.layout(model)
    zoom = (lay["r_struct"] * 1.15 + 50, lay["z_struct_bottom"] - 0.15 * (lay["z_struct_top"] - lay["z_struct_bottom"]) - 50,
            lay["z_struct_top"] + 0.15 * (lay["z_struct_top"] - lay["z_struct_bottom"]) + 50) if zoom_struct else None
    title = f"{q}, m = {int(d.get('m', 0))}, λ = {float(d['lam_nm']):.3f} nm" + (f", Q = {float(d['Q']):.4g}" if "Q" in d else "")
    ui.show_fig(fa.fig_axi_field(d, model, q, log=log, zoom=zoom, title=title), f"{Path(pick).stem}.png", f"{key}_dl")
    st.caption("Schnitt durch die Achse (rechts r ≥ 0 gerechnet, links gespiegelt; vorzeichenbehaftete Komponenten der Ordnung m tragen links den "
               "Faktor (−1)^m). Feld auf max |E| = 1 normiert. E_φ ist die Komponente um die Achse.")


def tab_info():
    st.markdown("""
**Rotationskörper (2,5D).** Eine Struktur, die sich bei Drehung um die z-Achse nicht ändert, hat Felder der Form E(r, z) e^{imφ}. Jede Azimutordnung m
ist ein eigenes 2D-Problem in der Meridianebene (r ≥ 0, z), gerechnet mit Nédélec-Elementen für (E_r, E_z) und H¹-Elementen für E_φ
(hp-FEM: `AxisymmetricResonance`, `AxisymmetricScattering`). Gegenüber einer 3D-Rechnung spart das zwei bis drei Größenordnungen an Unbekannten.

**Resonanzen.** Quasi-Normalmoden des offenen Resonators: komplexe Frequenz ω mit Im ω < 0, Resonanzwellenlänge λ = 2πc / Re ω, Güte
Q = Re ω / (−2 Im ω). Die PML (zylindrisch, außen, oben, unten) macht das Problem endlich; Moden mit Q < 0 oder fast reinem Feld in der PML sind
numerische Moden. Wahl von m: Grundmode einer Säule (HE₁₁) m = 1; ringförmige Moden ohne Azimutabhängigkeit m = 0; Flüstergalerie-Moden
von Scheiben und Ringen m ≈ 2π n_eff R / λ.

**Emitter.** Ein Gauß-verschmierter Punktdipol auf der Achse (Quantenpunkt): senkrecht zur Achse (koppelt an m = ±1, beide Ordnungen gleich) oder
entlang der Achse (m = 0). Gerechnet wird je Wellenlänge die Abstrahlung (Strom als Quelle). **Purcell-Faktor** F_P = P / P_bulk: abgegebene Leistung
(Fluss durch die geschlossene Fläche um den Emitter) durch die Leistung im unbegrenzten Material am Emitterort. **β**: Anteil durch die obere
Messebene (Sammlung nach oben). Mit „um die Resonanz“ wird zuerst die passende Mode gesucht und das Spektrum um sie gelegt; das Maximum von F_P
muss dort liegen und seine Breite etwa λ/Q betragen.

**Genauigkeit.** Resonanzwellenlänge und Q konvergieren schnell, der Purcell-Faktor langsamer (er hängt am Feld am Emitter und an der genauen
Lage der Resonanz). Immer mit p = 2 und p = 3 (oder feinerem Netz) vergleichen. Messwerte für die schnelle Mikrosäule siehe Anleitung.

**Streuung.** Eine ebene Welle aus der Umgebung unter dem Winkel θ gegen die Achse zerfällt in Azimutordnungen m = 0, ±1, ±2 … (Jacobi-Anger);
jede wird einzeln gelöst (Streufeld-Formulierung), bis das Paar ±m vernachlässigbar wenig streut. σ_sca aus dem Fluss durch eine geschlossene
Messfläche, σ_ext aus dem optischen Theorem, σ_abs als Differenz; Streudiagramm aus der Summe der Fernfelder; Nahfeld in der Einfallsebene. Für eine
einzelne Kugel zeigt die App die Mie-Reihe zum Vergleich. Geht nur in homogener Umgebung (kein Substrat).

**Modenzerlegung.** Mit der Riesz-Projektion wird das Purcell-Spektrum als Summe über die Quasi-Normalmoden geschrieben: der Beitrag der Resonanz
(eine Lorentz-Kurve) und ein glatter Hintergrund (übrige Moden, Kontinuum).

**Grenzen.** Emitter nur auf der Achse; Streuung nur ohne Substrat. Dipole in periodischen Strukturen folgen, wenn hp-FEM sie anbietet.
""")
