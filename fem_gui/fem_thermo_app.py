"""Heat and mechanics (2D) in the FEM model builder (Streamlit), hp-FEM 0.5. Called from fem_app.py: ``render_thermo(ctx)``.

Optics -> absorbed power -> temperature (Thermal2D / ThermoOptical2D) -> thermal stress (Elasticity2D, plane strain or axisymmetric), and mechanics alone on
blocks and cylinders. Models, figures: fem_thermo.py; worker: fem_thermo_worker.py.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import fem_materials as fm
import fem_run as fr
import fem_thermo as ft
import fem_ui as ui

S = st.session_state

INFO = """
**Was gerechnet wird.** Drei Fälle (zweidimensional, Ergebnisse je Meter in der unendlichen Richtung):

1. **Nanodraht: Optik → Wärme → Spannung.** Eine ebene Welle trifft einen Draht (Kreis) in einer verlustfreien Matrix (`square_with_disc`, PML). Aus dem Feld
   folgt die absorbierte Leistung, daraus die Temperatur (`Thermal2D`, Wärmesenke T = T_ref am äußeren Rand des Rechengebiets). Mit **selbstkonsistent** folgt
   die Permittivität der Temperatur, ε(T) = ε + (dε/dT)·(T − T_ref), und die Optik wird neu gerechnet, bis sich die Temperatur nicht mehr ändert
   (`ThermoOptical2D`, Festpunktiteration mit Relaxation). Optional folgt daraus die **Wärmespannung** (`Elasticity2D`, ebene Dehnung).
2. **Block (ebene Dehnung).** Rechteck mit Randbedingungen je Seite (frei, eingespannt, Symmetrie, Zug/Druck), Temperaturänderung (linear entlang x) und Schrumpfung.
3. **Zylinder (rotationssymmetrisch).** Halbebene (r, z) mit der Achse bei r = 0; sonst wie der Block, Spannungen σ_rr, σ_zz, σ_θθ, σ_rz.

**Prüfungen.** Beim Block und beim Zylinder mit den einfachen Randbedingungen (Symmetrie links und unten, Zug oder Temperaturänderung) vergleicht die App
die Verschiebungen mit der geschlossenen Lösung (Reiter 3). Beim Draht ohne Rückkopplung vergleicht sie die Temperatur mit der Linienquelle in einem
quadratischen Wärmesenken-Gebiet, ΔT = P/(2πκ)·ln(1,0787·a/r); das ist eine Näherung (Draht isotherm, κ_Draht ≫ κ_Matrix).

**Grenzen, die man kennen muss.**
- **Der Rand bestimmt die Temperatur.** In 2D wächst die Temperatur einer Linienquelle logarithmisch mit dem Abstand der Wärmesenke. Das Ergebnis gilt für
  genau dieses Gebiet (Senke bei der äußeren Kante, die PML-Schicht gehört mechanisch und thermisch zum Gebiet). Es ist keine Aussage über einen Draht im
  Halbraum oder auf einem Substrat.
- **Nur stationär.** Keine Zeitabhängigkeit der Wärmeleitung, keine Konvektion, keine Wärmestrahlung, Wärmeleitfähigkeit und Elastizität unabhängig von T.
- **Optische Rückwirkung nur über ε(T).** Thermische Ausdehnung ändert die Geometrie der Optik nicht (die Optik rechnet auf dem unverformten Netz).
- **Ebene Dehnung.** Temperatur und Schrumpfung wirken in der Bibliothek als Eigendehnung **in der Ebene** (ε*_zz = 0); die Dehnung in z ist null. Von Mises:
  σ_zz = ν(σ_xx + σ_yy). Der Wert hängt von dieser Annahme ab.
- **Materialwerte.** Die Vorgaben (κ, E, ν, α) sind grobe Richtwerte für Volumenmaterial bei Raumtemperatur (Dünnschichten und Kristallrichtung weichen ab);
  die Wasser/Luft-Werte sind Platzhalter, weil ein Fluid keine Schubsteifigkeit hat. Die Werte für dε/dT in den Vorlagen sind **Beispielwerte, keine Messwerte**.
- **Schrumpfung im Draht** wird an den Quadraturpunkten ausgewertet (Kreisrand nur näherungsweise), nicht über Zellmarken.
- **Nicht validiert** ist die Kette gegen Literaturwerte für konkrete Nanodrähte; geprüft sind die Bausteine (geschlossene Lösungen, Linienquelle) und die Selbstkonsistenz der Schleife in der Bibliothek.
"""


def _init():
    if "th_model" in S:
        return
    S.th_model = copy.deepcopy(ft.default_model())
    S.th_ver, S.th_proc, S.th_dir, S.th_msg = 0, None, None, None


def _bump():
    S.th_ver += 1


def _num(col, label, key, value, step=0.1, lo=-1e9, hi=1e9, fmt="%.5g", help=None):
    return float(col.number_input(label, float(lo), float(hi), float(value), float(step), key=key, format=fmt, help=help))


def render_thermo(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    work = Path(ctx["work"]) / "thermo"
    st.title("FEM-Modellwerkstatt: Wärme und Mechanik")
    st.caption("Absorbierte Leistung → Temperatur (auch selbstkonsistent mit ε(T)) → Wärmespannung, und Mechanik von Block und Zylinder (2D, ebene Dehnung bzw. "
               "rotationssymmetrisch). hp-FEM: Thermal2D, ThermoOptical2D, Elasticity2D.")
    if S.th_proc is not None:
        return _running()
    if not (ctx["FEATS"].get("thermo") or ctx["lib"].get("error")):
        st.warning("Das eingestellte Python hat kein hpfem mit Thermal2D, ThermoOptical2D und Elasticity2D. Anzeigen geht trotzdem.")
    t1, t2, t3, t4 = st.tabs(["1 Modell", "2 Rechnung", "3 Ergebnisse", "4 Info"])
    with t1:
        _tab_model(ctx)
    with t2:
        _tab_run(ctx, work)
    with t3:
        _tab_results(work)
    with t4:
        st.markdown(INFO)


# ===================================================================================================================== model tab
def _props_editor(m, ver, names, wire):
    """Table-like editor of the thermal / elastic data of the materials in use."""
    for n in names:
        p = m["props"].setdefault(n, ft.default_props(m["materials"][n]))
        with st.expander(f"Wärme- und Elastizitätsdaten: {n}", expanded=False):
            c = st.columns(4)
            p["kappa"] = _num(c[0], "κ (W/(m K))", f"th_k_{ver}_{n}", p["kappa"], 1.0, 0.0, 1e4, "%.5g", "Wärmeleitfähigkeit")
            p["E_GPa"] = _num(c[1], "E (GPa)", f"th_E_{ver}_{n}", p["E_GPa"], 5.0, 0.0001, 2000.0, "%.5g", "Elastizitätsmodul")
            p["nu"] = _num(c[2], "ν", f"th_nu_{ver}_{n}", p["nu"], 0.01, -0.99, 0.499, "%.4g", "Querkontraktionszahl")
            p["alpha"] = _num(c[3], "α (10⁻⁶/K)", f"th_al_{ver}_{n}", p["alpha"], 1.0, -100.0, 500.0, "%.5g", "Linearer Wärmeausdehnungskoeffizient")
            if wire:
                c = st.columns(2)
                p["deps_re"] = _num(c[0], "Re dε/dT (1/K)", f"th_der_{ver}_{n}", p["deps_re"], 1e-3, -10.0, 10.0, "%.4g",
                                    "Thermo-optischer Koeffizient: ε(T) = ε + (dε/dT)(T − T_ref). Nur mit „selbstkonsistent“ wirksam.")
                p["deps_im"] = _num(c[1], "Im dε/dT (1/K)", f"th_dei_{ver}_{n}", p["deps_im"], 1e-3, -10.0, 10.0, "%.4g",
                                    "Änderung der Absorption mit der Temperatur (Im ε > 0 ist Verlust).")
            spec = m["materials"][n]
            if spec.get("type") == "library" and spec.get("name") in ft.PROPS and st.button("Richtwerte der Bibliothek laden", key=f"th_def_{ver}_{n}"):
                m["props"][n] = ft.default_props(spec)
                _bump()
                st.rerun()


def _bc_editor(m, ver):
    names = ft.sides(m)
    mech = m["mech"]
    st.markdown("**Randbedingungen je Seite**")
    for s in ft.SIDE_TAGS:
        c = st.columns([3, 3, 2])
        bc = mech["bc"][s]
        c[0].markdown(f"<div style='padding-top:2rem'>{names[s]}</div>", unsafe_allow_html=True)
        kinds = list(ft.BC_KINDS)
        bc["kind"] = c[1].selectbox("Art", kinds, index=kinds.index(bc["kind"]) if bc["kind"] in kinds else 0, format_func=ft.BC_KINDS.get, key=f"th_bck_{ver}_{s}")
        if bc["kind"] == "traction":
            bc["value"] = _num(c[2], "σ (MPa)", f"th_bcv_{ver}_{s}", bc["value"], 10.0, -1e6, 1e6, "%.5g",
                               "Normalspannung auf der Seite; positiv = Zug (nach außen), negativ = Druck.")


def _tab_model(ctx):
    m, ver = S.th_model, S.th_ver
    ft.normalise(m)
    left, right = st.columns([6, 5])
    with left:
        presets = ft.presets()
        c = st.columns([3, 1])
        pick = c[0].selectbox("Vorlage", ["– auswählen –"] + list(presets), key="th_preset")
        if c[1].button("Laden", key="th_preset_load", disabled=pick.startswith("–")):
            S.th_model = copy.deepcopy(presets[pick])
            _bump()
            st.rerun()
        m["name"] = st.text_input("Name", value=m["name"], key=f"th_name_{ver}")
        cases = list(ft.CASES)
        new_case = st.radio("Fall", cases, index=cases.index(m["case"]) if m["case"] in cases else 0, format_func=ft.CASES.get, key=f"th_case_{ver}", horizontal=True)
        if new_case != m["case"]:
            fresh = copy.deepcopy(presets[next(k for k, v in presets.items() if v["case"] == new_case)])
            fresh["name"] = m["name"]
            S.th_model = fresh
            _bump()
            st.rerun()
        wire = m["case"] == "wire"
        lam0 = float(m["optics"]["wavelength_nm"]) if wire else 500.0
        st.subheader("Materialien" + (" (optische Daten)" if wire else ""))
        ui.material_editor(m, ver, lam0, ft.used_materials(m), ft.rename_material, _bump, prefix="th_")
        names = list(m["materials"])
        if wire:
            c = st.columns(2)
            m["inclusion"] = c[0].selectbox("Draht", names, index=names.index(m["inclusion"]) if m["inclusion"] in names else 0, key=f"th_inc_{ver}")
            m["background"] = c[1].selectbox("Matrix", names, index=names.index(m["background"]) if m["background"] in names else 0, key=f"th_bg_{ver}",
                                             help="Muss verlustfrei sein (Im ε = 0): in ihr läuft die einfallende ebene Welle.")
        else:
            m["background"] = m["inclusion"] = st.selectbox("Material", names, index=names.index(m["background"]) if m["background"] in names else 0, key=f"th_bg_{ver}")
        _props_editor(m, ver, sorted(ft.used_materials(m), key=names.index) if all(n in names for n in ft.used_materials(m)) else [], wire)
        if wire:
            st.subheader("Optik und Gebiet")
            c = st.columns(3)
            m["optics"]["wavelength_nm"] = _num(c[0], "Wellenlänge (nm)", f"th_lam_{ver}", m["optics"]["wavelength_nm"], 10.0, 100.0, 20000.0)
            m["optics"]["E0_MVm"] = _num(c[1], "Feldamplitude E₀ (MV/m)", f"th_e0_{ver}", m["optics"]["E0_MVm"], 0.1, 1e-6, 1e4,
                                         help="Amplitude der einfallenden ebenen Welle in der Matrix, E ∥ y, Ausbreitung in +x.")
            inten = ft.intensity_W_m2(m)
            c[2].metric("Intensität", f"{inten * 1e-4 * 1e-6:.3g} MW/cm²" if inten == inten else "–", help="½ n ε₀ c E₀² in der Matrix")
            w = m["wire"]
            c = st.columns(4)
            w["radius_nm"] = _num(c[0], "Radius (nm)", f"th_r_{ver}", w["radius_nm"], 5.0, 1.0, 5000.0)
            w["half_factor"] = int(c[1].number_input("Innenbreite / Radius", 2, 40, int(w["half_factor"]), key=f"th_hf_{ver}",
                                                    help="Halbbreite des physikalischen Gebiets in Radien (ganzzahlig)."))
            w["outer_factor"] = int(c[2].number_input("Außenbreite / Radius", 3, 80, int(w["outer_factor"]), key=f"th_of_{ver}",
                                                     help="Halbbreite des Rechengebiets in Radien. Der Rest bis zur Innenbreite ist die PML-Schicht; hier sitzt auch die Wärmesenke (T = T_ref)."))
            w["cells_per_radius"] = int(c[3].number_input("Zellen je Radius", 1, 8, int(w["cells_per_radius"]), key=f"th_n_{ver}"))
            st.subheader("Wärme")
            th = m["thermal"]
            c = st.columns(2)
            th["T_ref"] = _num(c[0], "Temperatur der Wärmesenke / Referenz T_ref (K)", f"th_tref_{ver}", th["T_ref"], 10.0, 1.0, 5000.0,
                               help="Der äußere Rand des Gebiets wird auf diese Temperatur gesetzt; sie ist auch die Referenz für ε(T) und die Wärmedehnung.")
            th["selfconsistent"] = c[1].checkbox("selbstkonsistent (ε hängt von T ab)", value=bool(th["selfconsistent"]), key=f"th_sc_{ver}",
                                                 help="ThermoOptical2D: Festpunktiteration Optik ↔ Wärme mit den dε/dT oben. Jede Iteration löst die Optik neu (Sekunden bis Minuten).")
            if th["selfconsistent"]:
                c = st.columns(3)
                th["relaxation"] = _num(c[0], "Relaxation", f"th_rel_{ver}", th["relaxation"], 0.1, 0.01, 1.0, "%.3g", "1 = volle Schritte; kleiner dämpft bei starker Kopplung.")
                th["tolerance"] = _num(c[1], "Toleranz (K)", f"th_tol_{ver}", th["tolerance"], 1e-3, 1e-9, 10.0, "%.3g")
                th["max_iter"] = int(c[2].number_input("max. Iterationen", 1, 200, int(th["max_iter"]), key=f"th_mi_{ver}"))
            st.subheader("Mechanik")
            mech = m["mech"]
            mech["on"] = st.checkbox("Wärmespannung berechnen (Elasticity2D, ebene Dehnung)", value=bool(mech["on"]), key=f"th_mon_{ver}")
            if mech["on"]:
                c = st.columns(2)
                walls = list(ft.WALL_BC)
                mech["wall"] = c[0].radio("Rand des Gebiets", walls, index=walls.index(mech["wall"]) if mech["wall"] in walls else 0, format_func=ft.WALL_BC.get,
                                          key=f"th_wall_{ver}", horizontal=False)
                mech["shrink_pct"] = _num(c[1], "Schrumpfung des Drahts (%)", f"th_shr_{ver}", mech["shrink_pct"], 0.1, -50.0, 50.0, "%.4g",
                                          "Isotrope Eigendehnung −s im Draht (positiv = Schrumpfung), zusätzlich zur Wärmedehnung.")
        else:
            b = m["block"]
            axi = m["case"] == "axi"
            st.subheader("Geometrie")
            c = st.columns(4)
            b["width_um"] = _num(c[0], "Radius R (µm)" if axi else "Breite W (µm)", f"th_w_{ver}", b["width_um"], 0.5, 0.001, 1e4)
            b["height_um"] = _num(c[1], "Höhe H (µm)", f"th_h_{ver}", b["height_um"], 0.5, 0.001, 1e4)
            b["nx"] = int(c[2].number_input("Zellen in x (r)", 1, 200, int(b["nx"]), key=f"th_nx_{ver}"))
            b["ny"] = int(c[3].number_input("Zellen in y (z)", 1, 200, int(b["ny"]), key=f"th_ny_{ver}"))
            mech = m["mech"]
            mech["on"] = True
            st.subheader("Mechanik")
            _bc_editor(m, ver)
            c = st.columns(3)
            lab = "z" if axi else "x"
            m["thermal"]["T_ref"] = _num(c[0], "Referenztemperatur T_ref (K)", f"th_tref_{ver}", m["thermal"]["T_ref"], 10.0, 1.0, 5000.0)
            mech["dT_a"] = _num(c[1], f"ΔT bei {lab} = 0 (K)", f"th_dta_{ver}", mech["dT_a"], 10.0, -2000.0, 2000.0, help="Temperaturänderung gegenüber T_ref, linear interpoliert.")
            mech["dT_b"] = _num(c[2], f"ΔT bei {lab} = max (K)", f"th_dtb_{ver}", mech["dT_b"], 10.0, -2000.0, 2000.0)
            mech["shrink_pct"] = _num(st, "Schrumpfung des ganzen Körpers (%)", f"th_shr_{ver}", mech["shrink_pct"], 0.1, -50.0, 50.0, "%.4g",
                                      "Isotrope Eigendehnung −s (positiv = Schrumpfung).")
    with right:
        for lvl, text in ft.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        try:
            ui.show_fig(ft.fig_preview(m), "geometrie.png", f"th_prev_dl_{ver}")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")


# ===================================================================================================================== run tab
def _tab_run(ctx, work):
    m, ver = S.th_model, S.th_ver
    ft.normalise(m)
    st.subheader("Ordnung und Ausgabe")
    c = st.columns(2)
    m["order"] = int(c[0].number_input("Polynomordnung p", 1, 8, int(m["order"]), key=f"th_p_{ver}"))
    m["grid"] = int(c[1].number_input("Auflösung der Feldkarten (Punkte in x)", 40, 400, int(m["grid"]), 20, key=f"th_grid_{ver}",
                                      help="Die Felder werden auf einem regelmäßigen Raster ausgewertet (Zellmitten). Das Raster ändert die Rechnung nicht."))
    errs = [t for lvl, t in ft.validate(m) if lvl == "error"]
    for e in errs:
        st.error(e)
    if m["case"] == "wire":
        n_it = "bis zu " + str(m["thermal"]["max_iter"]) + " Optik-Rechnungen" if m["thermal"]["selfconsistent"] else "eine Optik-Rechnung"
        st.markdown(f"Je Lauf {n_it}, eine Wärme- und (falls gewählt) eine Mechanik-Rechnung. Bei 2048 Zellen und p = 3 etwa 5 s ohne und 20–30 s mit Rückkopplung, "
                    "die Zeit wächst mit Zellen und Ordnung.")
    else:
        st.markdown("Block und Zylinder rechnen in wenigen Sekunden.")
    if st.button("Rechnen", type="primary", key="th_run", disabled=bool(errs)):
        try:
            work.mkdir(parents=True, exist_ok=True)
            (work / "job.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
            S.th_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script="fem_thermo_worker.py")
            S.th_dir, S.th_cancel = str(work), False
            S.th_run_model = copy.deepcopy(m)
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("th_msg"):
        (st.success if S.th_msg[0] == "ok" else st.error)(S.th_msg[1] + (" Die Ergebnisse stehen im Reiter 3." if S.th_msg[0] == "ok" else ""))
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(work, tail=40))


def _running():
    proc, folder = S.th_proc, Path(S.th_dir)
    log = fr.read_log(folder)
    st.subheader("Rechnung läuft …")
    if proc.poll() is None:
        frac, label = fr.progress_info(log)
        st.progress(frac, text=label)
        st.code("\n".join(log.splitlines()[-10:]) or "(noch keine Ausgabe)")
        if S.get("th_cancel"):
            st.info("Abbruch angefordert: die Rechnung hält nach dem laufenden Schritt an (ein einzelner Löserlauf lässt sich nicht unterbrechen).")
        elif st.button("Abbrechen", key="th_cancel_btn"):
            fr.request_cancel(folder)
            S.th_cancel = True
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        S.th_proc = None
        ok = proc.returncode == 0 and not flags["errors"]
        S.th_msg = ("ok", "Abgebrochen." if flags["cancelled"] else "Fertig.") if ok else ("error", f"Beendet mit Code {proc.returncode}: " + "; ".join(flags["errors"][:3]))
        st.rerun()


# ===================================================================================================================== results
def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def _tab_results(work):
    res_file = work / "results.json"
    if not res_file.is_file():
        st.info("Noch keine Ergebnisse: im Reiter 2 rechnen.")
        return
    res = json.loads(res_file.read_text(encoding="utf-8"))
    if res.get("error"):
        st.error(res["error"])
        return
    meta = res["meta"]
    model = S.get("th_run_model") or S.th_model
    st.caption(f"{meta.get('name', '')} · {meta.get('cells')} Zellen, p = {meta.get('order')} · hpfem {meta.get('hpfem', '?')} · Start {meta.get('started', '')}"
               + (f" · {meta['total_s']:.1f} s" if "total_s" in meta else ""))
    if res.get("cancelled"):
        st.warning("Die Rechnung wurde abgebrochen.")
        return
    if not res.get("finished"):
        st.info("Die Rechnung ist noch nicht fertig.")
        return
    npz_path = work / "fields.npz"
    npz = np.load(npz_path) if npz_path.is_file() else None
    if res.get("converged") is False:
        st.warning("Die thermo-optische Schleife hat die Toleranz nicht erreicht: Relaxation verkleinern oder mehr Iterationen zulassen.")
    sub = st.tabs(["Kennzahlen und Prüfung", "Felder", "Schnitt", "Verformung", "Konvergenz"])
    with sub[0]:
        sc = res["scalars"]
        ui.table_show(pd.DataFrame([(k, _fmt(v)) for k, v in sc.items()], columns=["Größe", "Wert"]))
        if model["case"] == "wire" and not model["thermal"]["selfconsistent"] and "Absorbierte Leistung (W/m)" in sc and "Temperaturanstieg max. (K)" in sc:
            est = ft.heat_line_estimate(model, float(sc["Absorbierte Leistung (W/m)"]))
            got = float(sc["Temperaturanstieg max. (K)"])
            st.markdown(f"**Prüfung Linienquelle:** ΔT ≈ P/(2πκ)·ln(1,0787·a/r) = {est:.4g} K gegen {got:.4g} K aus der FEM-Rechnung "
                        f"(Abweichung {100 * (got - est) / est:+.1f} %). Näherung für einen gut leitenden Draht; bei kleinem Verhältnis κ_Draht/κ_Matrix weicht sie ab.")
        checks = ft.analytic_checks(model)
        if checks and "probes" in res:
            rows = []
            for lab, exp, key in checks:
                got = res["probes"][key]
                rows.append((lab, f"{got:.6g}", f"{exp:.6g}", f"{abs(got - exp) / abs(exp):.2e}" if exp else "–"))
            st.markdown("**Prüfung gegen die geschlossene Lösung** (Symmetrieebenen, ebene Dehnung bzw. Rotationssymmetrie):")
            ui.table_show(pd.DataFrame(rows, columns=["Größe", "FEM", "geschlossen", "rel. Abweichung"]))
        elif model["case"] != "wire":
            st.caption("Für diese Randbedingungen gibt es hier keine geschlossene Vergleichslösung (sie gilt für Symmetrie links und unten, freie Seiten sonst, gleichmäßige Temperatur).")
        if model["case"] == "wire":
            st.caption("Leistung und Energie je Meter Drahtlänge. Das Rechengebiet endet an der Wärmesenke: die Temperatur gilt nur für diesen Abstand.")
    if npz is None:
        for t in sub[1:4]:
            with t:
                st.info("Keine Felddaten.")
    else:
        keys = [k for k in ft.FIELDS if k in npz.files]
        with sub[1]:
            k = st.selectbox("Feld", keys, format_func=lambda k: ft.FIELDS[k][0], key="th_field")
            ui.show_fig(ft.fig_field(npz, k), f"{k}.png", "th_dl_field")
            st.caption("Raster der Zellmitten, außerhalb des Netzes leer. Spannungen sind Zellwerte der Lösung (unstetig über Materialgrenzen). "
                       + ("Die PML-Schicht gehört zum Gebiet; das optische Feld dort ist nicht physikalisch." if model["case"] == "wire" else ""))
        with sub[2]:
            c = st.columns(3)
            k2 = c[0].selectbox("Feld", keys, format_func=lambda k: ft.FIELDS[k][0], key="th_line_field")
            axis = c[1].radio("Richtung", ["x", "y"], horizontal=True, key="th_line_axis")
            frac = c[2].slider("Lage (Anteil)", 0.0, 1.0, 0.5, 0.01, key="th_line_frac")
            ui.show_fig(ft.fig_line(npz, k2, axis, frac), "schnitt.png", "th_dl_line")
        with sub[3]:
            if "mesh_disp" in npz.files:
                dmax = float(np.abs(npz["mesh_disp"]).max())
                span = float(max(np.ptp(npz["mesh_vertices"][:, 0]), np.ptp(npz["mesh_vertices"][:, 1])))
                auto = max(1.0, 0.05 * span / dmax) if dmax > 0 else 1.0
                scale = st.number_input("Überhöhung der Verschiebung", 0.0, 1e9, float(f"{auto:.2g}"), format="%.6g", key=f"th_def_scale_{len(npz['mesh_disp'])}",
                                        help="Vorgabe: größte Verschiebung ≈ 5 % der Gebietsgröße. Die Verformung ist winzig (nm) und wird stark überhöht gezeichnet.")
                ui.show_fig(ft.fig_deformed(npz, scale), "verformt.png", "th_dl_def")
            else:
                st.info("Ohne Mechanik keine Verformung.")
    with sub[4]:
        if res.get("history"):
            ui.show_fig(ft.fig_history(res["history"]), "konvergenz.png", "th_dl_hist")
            ui.table_show(ft.history_dataframe(res).round(6))
            st.caption("Größte Temperaturänderung zwischen zwei Iterationen; die erste Zeile enthält den Sprung vom Startfeld (0 K) auf die erste Lösung.")
        else:
            st.info("Keine Iteration (nicht selbstkonsistent oder nur Mechanik).")
    st.download_button("Modell (JSON)", json.dumps(model, indent=1, ensure_ascii=False).encode("utf-8"), file_name="waerme_mechanik_modell.json",
                       mime="application/json", key="th_dl_model")
