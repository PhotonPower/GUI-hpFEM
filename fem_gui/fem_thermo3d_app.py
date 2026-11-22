"""Heat and mechanics in 3D in the FEM model builder (Streamlit), hp-FEM 0.5. Called from fem_thermo_app.py (choice 3D): ``render_thermo3d(ctx)``.

Sphere in a PML box: optics -> absorbed power -> temperature (Thermal3D / ThermoOptical3D) -> thermal stress (Elasticity3D); box: heat conduction with a
volumetric source and mechanics with boundary conditions per face. Models, figures: fem_thermo3d.py; worker: fem_thermo3d_worker.py.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import fem_run as fr
import fem_thermo as ft
import fem_thermo3d as t3
import fem_ui as ui

S = st.session_state
GB = 1024.0**3

INFO = """
**Was gerechnet wird.** Zwei Fälle, dreidimensional (Ergebnisse in W, J, K, nm, MPa):

1. **Kugel: Optik → Wärme → Spannung.** Eine ebene Welle (E längs x, Ausbreitung längs +z) trifft eine Kugel in einer verlustfreien Matrix (`box_with_ball`, PML).
   Aus dem Feld folgt die absorbierte Leistung, daraus die Temperatur (`Thermal3D`, Wärmesenke T = T_ref an den Außenflächen der Box). **Selbstkonsistent** folgt die
   Permittivität der Temperatur, ε(T) = ε + (dε/dT)(T − T_ref), und die Optik wird neu gerechnet, bis sich die Temperatur nicht mehr ändert (`ThermoOptical3D`). Optional
   folgt die **Wärmespannung** (`Elasticity3D`; Wärmeausdehnung α je Material, Schrumpfung der Kugel; Außenflächen eingespannt oder frei). Mit dem **Viertelmodell**
   (x > 0, y > 0; Spiegelsymmetrie der Welle) rechnet die App ein Viertel des Gebiets, Leistung und Energie werden mit 4 multipliziert und die Felder gespiegelt.
2. **Quader: Wärmeleitung und Mechanik.** Quader [0, W] × [0, D] × [0, H]: Randbedingungen je Fläche (frei, eingespannt, Symmetrie, Zug/Druck), Temperaturänderung und
   Schrumpfung (linear in z) oder eine **Wärmequelle** (gleichmäßig, mW/µm³) mit Wärmesenken auf gewählten Flächen (`Thermal3D`, die übrigen Flächen adiabatisch);
   das berechnete Temperaturfeld treibt dann die Wärmeausdehnung.

**Prüfungen.** Quader mit Symmetrieebenen x = 0, y = 0, z = 0 und Zug auf x = W (oder frei) bei gleichmäßiger Temperatur und Schrumpfung: die Verschiebungen werden mit der
geschlossenen Lösung verglichen (die Dehnungen sind gleichmäßig, die Lösung ist in 3D exakt; gemessen bis 1e-13 relativ). Wärmequelle zwischen den Senken x = 0 und x = W:
T_max = T_ref + qW²/(8κ), nur bei x = 0: qW²/(2κ) (eindimensional, exakt; gemessen bis 1e-12 relativ). Kugel ohne Rückkopplung: der Temperaturanstieg im Mittelpunkt wird mit der
Schätzung ΔT ≈ P/(4πκ)·(1/R − 1/R_s) verglichen (R_s = 0,6607·2a, die Kapazität des Würfels der Halbbreite a); das ist eine Näherung (gut leitende Kugel, Würfel
als Kugel ersetzt) und stimmt auf wenige Prozent.

**Grenzen, die man kennen muss.**
- **3D kostet.** Die Bibliothek im Wheel hat nur den direkten Löser SparseLU (ein Thread). Die App zeigt vor jedem Lauf die Schätzung (Zellen, Unbekannte, Speicher, Zeit) und der Worker
  bricht ab, wenn die Schätzung die eingestellte Speichergrenze überschreitet. Die Zeit der Mechanik ist eine grobe Faustformel aus Messungen am Quader (die Kugel war etwa dreimal
  schneller als geschätzt). Selbstkonsistent löst jede Iteration die Optik neu: bei 10 790 Unbekannten rund 65 s für 5 Iterationen.
- **Der Rand bestimmt die Temperatur.** Die Senke sitzt an den Außenflächen der Box (die PML-Schicht gehört thermisch und mechanisch zum Gebiet); die Temperatur gilt für genau dieses
  Gebiet und keine Aussage über eine Kugel im unendlichen Raum oder auf einem Substrat. (In 3D wächst der Anstieg einer Punktquelle nicht über alle Grenzen, hängt aber vom Senkenabstand ab.)
- **Netz und PML.** Das Standardnetz (1 Zelle je Radius, p = 2) ist grob; die Bibliothek meldet dann eine unteraufgelöste PML (Protokoll). Leistung und Temperatur ändern sich mit dem Netz;
  die Konvergenz muss der Nutzer prüfen (Zellen je Radius oder p erhöhen).
- **Nur stationär.** Keine Zeitabhängigkeit, keine Konvektion, keine Strahlung; Wärmeleitfähigkeit und Elastizität hängen nicht von T ab. Die Optik rechnet auf dem unverformten Netz.
- **Wärmedehnung in 3D** wirkt in allen drei Richtungen (α(T − T_ref) I); Von Mises aus dem vollen Spannungstensor.
- **Felder** werden auf einem regelmäßigen Raster der Zellmitten ausgewertet (außerhalb des Netzes leer); Maximalwerte sind Rasterwerte und können Spitzen verfehlen. Spannungen sind Zellwerte der
  Lösung (unstetig über Materialgrenzen). Das optische Feld in der PML ist nicht physikalisch.
- **Materialwerte.** κ, E, ν, α sind grobe Richtwerte; dε/dT der Vorlage ist ein Beispielwert, kein Messwert. Fluide haben keine Schubsteifigkeit (Platzhalterwerte).
- **Schrumpfung der Kugel** wird an den Quadraturpunkten ausgewertet (die Kugeloberfläche nur näherungsweise).
- **Nicht validiert** ist die Kette gegen Literaturwerte für konkrete Partikel (geprüft sind die Bausteine gegen geschlossene Lösungen und die Schätzung oben); in der Bibliothek weicht die absorbierte
  Leistung zwischen `absorbed_power_by_tag` und der Lastbilanz bei grobem Netz um etwa 2·10⁻³ ab.
"""


def _init():
    if "th3_model" in S:
        return
    S.th3_model = copy.deepcopy(t3.default_model())
    S.th3_ver, S.th3_proc, S.th3_dir, S.th3_msg, S.th3_est = 0, None, None, None, None


def _bump():
    S.th3_ver += 1
    S.th3_est = None


def _num(col, label, key, value, step=0.1, lo=-1e9, hi=1e9, fmt="%.5g", help=None):
    return float(col.number_input(label, float(lo), float(hi), float(value), float(step), key=key, format=fmt, help=help))


def run_estimate(py, repo, folder, jb, threads=0, timeout=300):
    """Mesh and memory estimate from the library (no solve): dict as printed by the worker, or {'error': ...}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(jb), encoding="utf-8")
    try:
        r = subprocess.run([str(py), str(fr.HERE / "fem_thermo3d_worker.py"), str(folder), "estimate"], cwd=fr._cwd(repo, folder),
                           env=fr.worker_env(repo, threads), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "Zeitüberschreitung beim Vernetzen."}
    for line in r.stdout.splitlines():
        if line.startswith("ESTIMATE"):
            return json.loads(line[len("ESTIMATE"):])
    txt = (r.stdout + r.stderr).strip()
    return {"error": txt.splitlines()[-1] if txt else "keine Ausgabe"}


def render_thermo3d(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    work = Path(ctx["work"]) / "thermo3d"
    st.title("FEM-Modellwerkstatt: Wärme und Mechanik (3D)")
    st.caption("Kugel: absorbierte Leistung → Temperatur (auch selbstkonsistent mit ε(T)) → Wärmespannung; Quader: Wärmeleitung und Mechanik. "
               "hp-FEM: Scattering3D, Thermal3D, ThermoOptical3D, Elasticity3D. 3D ist speicher- und zeitintensiv: vor dem Rechnen die Schätzung ansehen.")
    if S.th3_proc is not None:
        return _running()
    if not (ctx["FEATS"].get("thermo3d") or ctx["lib"].get("error")):
        st.warning("Das eingestellte Python hat kein hpfem mit Thermal3D, ThermoOptical3D und Elasticity3D. Anzeigen geht trotzdem.")
    t1, t2, t3_, t4 = st.tabs(["1 Modell", "2 Rechnung", "3 Ergebnisse", "4 Info"])
    with t1:
        _tab_model()
    with t2:
        _tab_run(ctx, work)
    with t3_:
        _tab_results(work)
    with t4:
        st.markdown(INFO)


# ===================================================================================================================== model tab
def _props_editor(m, ver, names, particle):
    for n in names:
        p = m["props"].setdefault(n, ft.default_props(m["materials"][n]))
        with st.expander(f"Wärme- und Elastizitätsdaten: {n}", expanded=False):
            c = st.columns(4)
            p["kappa"] = _num(c[0], "κ (W/(m K))", f"t3_k_{ver}_{n}", p["kappa"], 1.0, 0.0, 1e4, "%.5g", "Wärmeleitfähigkeit")
            p["E_GPa"] = _num(c[1], "E (GPa)", f"t3_E_{ver}_{n}", p["E_GPa"], 5.0, 0.0001, 2000.0, "%.5g", "Elastizitätsmodul")
            p["nu"] = _num(c[2], "ν", f"t3_nu_{ver}_{n}", p["nu"], 0.01, -0.99, 0.499, "%.4g", "Querkontraktionszahl")
            p["alpha"] = _num(c[3], "α (10⁻⁶/K)", f"t3_al_{ver}_{n}", p["alpha"], 1.0, -100.0, 500.0, "%.5g", "Linearer Wärmeausdehnungskoeffizient")
            if particle:
                c = st.columns(2)
                p["deps_re"] = _num(c[0], "Re dε/dT (1/K)", f"t3_der_{ver}_{n}", p["deps_re"], 1e-3, -10.0, 10.0, "%.4g",
                                    "Thermo-optischer Koeffizient: ε(T) = ε + (dε/dT)(T − T_ref). Nur mit „selbstkonsistent“ wirksam.")
                p["deps_im"] = _num(c[1], "Im dε/dT (1/K)", f"t3_dei_{ver}_{n}", p["deps_im"], 1e-3, -10.0, 10.0, "%.4g",
                                    "Änderung der Absorption mit der Temperatur (Im ε > 0 ist Verlust).")
            spec = m["materials"][n]
            if spec.get("type") == "library" and spec.get("name") in ft.PROPS and st.button("Richtwerte der Bibliothek laden", key=f"t3_def_{ver}_{n}"):
                m["props"][n] = ft.default_props(spec)
                _bump()
                st.rerun()


def _bc_editor(m, ver):
    mech, th = m["mech"], m["thermal"]
    st.markdown("**Randbedingungen je Fläche**")
    for s in t3.SIDE_TAGS:
        c = st.columns([2, 3, 2, 2]) if th["heat"] else st.columns([2, 3, 2])
        bc = mech["bc"][s]
        c[0].markdown(f"<div style='padding-top:2rem'>{t3.SIDES[s]}</div>", unsafe_allow_html=True)
        kinds = list(t3.BC_KINDS)
        bc["kind"] = c[1].selectbox("Art", kinds, index=kinds.index(bc["kind"]) if bc["kind"] in kinds else 0, format_func=t3.BC_KINDS.get, key=f"t3_bck_{ver}_{s}")
        if bc["kind"] == "traction":
            bc["value"] = _num(c[2], "σ (MPa)", f"t3_bcv_{ver}_{s}", bc["value"], 10.0, -1e6, 1e6, "%.5g", "Normalspannung auf der Fläche; positiv = Zug (nach außen), negativ = Druck.")
        if th["heat"]:
            th["sinks"][s] = c[-1].checkbox("Wärmesenke (T_ref)", value=bool(th["sinks"][s]), key=f"t3_sink_{ver}_{s}",
                                            help="Feste Temperatur T_ref auf dieser Fläche; nicht gewählte Flächen sind adiabatisch.")


def _tab_model():
    m, ver = S.th3_model, S.th3_ver
    t3.normalise(m)
    left, right = st.columns([6, 5])
    with left:
        presets = t3.presets()
        c = st.columns([3, 1])
        pick = c[0].selectbox("Vorlage", ["– auswählen –"] + list(presets), key="t3_preset")
        if c[1].button("Laden", key="t3_preset_load", disabled=pick.startswith("–")):
            S.th3_model = copy.deepcopy(presets[pick])
            _bump()
            st.rerun()
        m["name"] = st.text_input("Name", value=m["name"], key=f"t3_name_{ver}")
        cases = list(t3.CASES)
        new_case = st.radio("Fall", cases, index=cases.index(m["case"]) if m["case"] in cases else 0, format_func=t3.CASES.get, key=f"t3_case_{ver}", horizontal=True)
        if new_case != m["case"]:
            fresh = copy.deepcopy(presets[next(k for k, v in presets.items() if v["case"] == new_case)])
            fresh["name"] = m["name"]
            S.th3_model = fresh
            _bump()
            st.rerun()
        particle = m["case"] == "particle"
        lam0 = float(m["optics"]["wavelength_nm"]) if particle else 500.0
        st.subheader("Materialien" + (" (optische Daten)" if particle else ""))
        ui.material_editor(m, ver, lam0, t3.used_materials(m), t3.rename_material, _bump, prefix="t3_")
        names = list(m["materials"])
        if particle:
            c = st.columns(2)
            m["inclusion"] = c[0].selectbox("Kugel", names, index=names.index(m["inclusion"]) if m["inclusion"] in names else 0, key=f"t3_inc_{ver}")
            m["background"] = c[1].selectbox("Matrix", names, index=names.index(m["background"]) if m["background"] in names else 0, key=f"t3_bg_{ver}",
                                             help="Muss verlustfrei sein (Im ε = 0): in ihr läuft die einfallende ebene Welle.")
        else:
            m["background"] = m["inclusion"] = st.selectbox("Material", names, index=names.index(m["background"]) if m["background"] in names else 0, key=f"t3_bg_{ver}")
        used = t3.used_materials(m)
        _props_editor(m, ver, sorted(used, key=names.index) if all(n in names for n in used) else [], particle)
        th, mech = m["thermal"], m["mech"]
        if particle:
            st.subheader("Optik und Gebiet")
            c = st.columns(3)
            m["optics"]["wavelength_nm"] = _num(c[0], "Wellenlänge (nm)", f"t3_lam_{ver}", m["optics"]["wavelength_nm"], 10.0, 100.0, 20000.0)
            m["optics"]["E0_MVm"] = _num(c[1], "Feldamplitude E₀ (MV/m)", f"t3_e0_{ver}", m["optics"]["E0_MVm"], 0.1, 1e-6, 1e4,
                                         help="Amplitude der einfallenden ebenen Welle in der Matrix, E ∥ x, Ausbreitung in +z.")
            inten = ft.intensity_W_m2(m)
            c[2].metric("Intensität", f"{inten * 1e-4 * 1e-6:.3g} MW/cm²" if inten == inten else "–", help="½ n ε₀ c E₀² in der Matrix")
            p = m["particle"]
            c = st.columns(4)
            p["radius_nm"] = _num(c[0], "Radius (nm)", f"t3_r_{ver}", p["radius_nm"], 5.0, 1.0, 5000.0)
            p["half_factor"] = int(c[1].number_input("Innenbreite / Radius", 2, 20, int(p["half_factor"]), key=f"t3_hf_{ver}", help="Halbbreite des physikalischen Gebiets in Radien (ganzzahlig)."))
            p["outer_factor"] = int(c[2].number_input("Außenbreite / Radius", 3, 40, int(p["outer_factor"]), key=f"t3_of_{ver}",
                                                     help="Halbbreite der Box in Radien. Der Rest bis zur Innenbreite ist die PML-Schicht; hier sitzt auch die Wärmesenke (T = T_ref)."))
            p["n"] = int(c[3].number_input("Zellen je Radius", 1, 6, int(p["n"]), key=f"t3_n_{ver}", help="Mehr Zellen: genauer, aber der Speicher wächst steil (siehe Schätzung im Reiter 2)."))
            p["quarter"] = st.checkbox("Viertelmodell (x > 0, y > 0, Spiegelsymmetrie)", value=bool(p["quarter"]), key=f"t3_q_{ver}",
                                       help="Rechnet ein Viertel des Gebiets (x = 0 ideal leitend für die Optik, y = 0 magnetisch; Symmetrieebenen für Wärme und Mechanik); Leistung und Energie ×4.")
            st.subheader("Wärme")
            c = st.columns(2)
            th["T_ref"] = _num(c[0], "Temperatur der Wärmesenke / Referenz T_ref (K)", f"t3_tref_{ver}", th["T_ref"], 10.0, 1.0, 5000.0,
                               help="Die Außenflächen der Box werden auf diese Temperatur gesetzt; sie ist auch die Referenz für ε(T) und die Wärmedehnung.")
            th["selfconsistent"] = c[1].checkbox("selbstkonsistent (ε hängt von T ab)", value=bool(th["selfconsistent"]), key=f"t3_sc_{ver}",
                                                 help="ThermoOptical3D: Festpunktiteration Optik ↔ Wärme mit den dε/dT oben. Jede Iteration löst die Optik neu (Minuten).")
            if th["selfconsistent"]:
                c = st.columns(3)
                th["relaxation"] = _num(c[0], "Relaxation", f"t3_rel_{ver}", th["relaxation"], 0.1, 0.01, 1.0, "%.3g", "1 = volle Schritte; kleiner dämpft bei starker Kopplung.")
                th["tolerance"] = _num(c[1], "Toleranz (K)", f"t3_tol_{ver}", th["tolerance"], 1e-3, 1e-9, 10.0, "%.3g")
                th["max_iter"] = int(c[2].number_input("max. Iterationen", 1, 100, int(th["max_iter"]), key=f"t3_mi_{ver}"))
            st.subheader("Mechanik")
            mech["on"] = st.checkbox("Wärmespannung berechnen (Elasticity3D)", value=bool(mech["on"]), key=f"t3_mon_{ver}")
            if mech["on"]:
                c = st.columns(2)
                walls = list(t3.WALL_BC)
                mech["wall"] = c[0].radio("Außenflächen der Box", walls, index=walls.index(mech["wall"]) if mech["wall"] in walls else 0, format_func=t3.WALL_BC.get, key=f"t3_wall_{ver}")
                mech["shrink_a_pct"] = _num(c[1], "Schrumpfung der Kugel (%)", f"t3_shr_{ver}", mech["shrink_a_pct"], 0.1, -50.0, 50.0, "%.4g",
                                            "Isotrope Eigendehnung −s in der Kugel (positiv = Schrumpfung), zusätzlich zur Wärmedehnung.")
        else:
            b = m["block"]
            st.subheader("Geometrie")
            c = st.columns(3)
            b["W_um"] = _num(c[0], "Breite W (µm)", f"t3_w_{ver}", b["W_um"], 0.5, 0.001, 1e4)
            b["D_um"] = _num(c[1], "Tiefe D (µm)", f"t3_d_{ver}", b["D_um"], 0.5, 0.001, 1e4)
            b["H_um"] = _num(c[2], "Höhe H (µm)", f"t3_h_{ver}", b["H_um"], 0.5, 0.001, 1e4)
            c = st.columns(3)
            b["nx"] = int(c[0].number_input("Zellen in x", 1, 40, int(b["nx"]), key=f"t3_nx_{ver}"))
            b["ny"] = int(c[1].number_input("Zellen in y", 1, 40, int(b["ny"]), key=f"t3_ny_{ver}"))
            b["nz"] = int(c[2].number_input("Zellen in z", 1, 40, int(b["nz"]), key=f"t3_nz_{ver}"))
            mech["on"] = True
            st.subheader("Wärme")
            c = st.columns(3)
            th["T_ref"] = _num(c[0], "Referenztemperatur T_ref (K)", f"t3_tref_{ver}", th["T_ref"], 10.0, 1.0, 5000.0)
            th["heat"] = c[1].checkbox("Wärmequelle und Senken (Thermal3D)", value=bool(th["heat"]), key=f"t3_heat_{ver}",
                                       help="Gleichmäßige Wärmequelle q im Quader; die gewählten Flächen (Reiter Mechanik) sind Senken bei T_ref, die übrigen adiabatisch. Das berechnete Temperaturfeld treibt die Wärmedehnung.")
            if th["heat"]:
                th["q_mWum3"] = _num(c[2], "Wärmequelle q (mW/µm³)", f"t3_q_{ver}", th["q_mWum3"], 0.1, -1e6, 1e6, "%.5g", "1 mW/µm³ = 10¹⁵ W/m³.")
            st.subheader("Mechanik")
            _bc_editor(m, ver)
            if not th["heat"]:
                c = st.columns(2)
                mech["dT_a"] = _num(c[0], "ΔT bei z = 0 (K)", f"t3_dta_{ver}", mech["dT_a"], 10.0, -2000.0, 2000.0, help="Temperaturänderung gegenüber T_ref, linear in z interpoliert.")
                mech["dT_b"] = _num(c[1], "ΔT bei z = H (K)", f"t3_dtb_{ver}", mech["dT_b"], 10.0, -2000.0, 2000.0)
            c = st.columns(2)
            mech["shrink_a_pct"] = _num(c[0], "Schrumpfung bei z = 0 (%)", f"t3_sha_{ver}", mech["shrink_a_pct"], 0.1, -50.0, 50.0, "%.4g", "Isotrope Eigendehnung −s (positiv = Schrumpfung), linear in z.")
            mech["shrink_b_pct"] = _num(c[1], "Schrumpfung bei z = H (%)", f"t3_shb_{ver}", mech["shrink_b_pct"], 0.1, -50.0, 50.0, "%.4g")
    with right:
        for lvl, text in t3.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        try:
            ui.show_fig(t3.fig_preview(m), "geometrie3d.png", f"t3_prev_dl_{ver}")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")


# ===================================================================================================================== run tab
def _estimate_box(est, m):
    if est is None:
        return
    if "error" in est:
        st.error(f"Netz oder Schätzung nicht möglich: {est['error']}")
        return
    parts = [f"**{est['cells']} Zellen, {est['h1_dofs']} H1-Unbekannte**"]
    if "nd_dofs" in est:
        parts.append(f"Optik {est['nd_dofs']} Unbekannte, {est['optics_text']}, Zeit je Optik-Lauf grob {est['optics_seconds']:.0f} s")
    if m["case"] == "block" or m["mech"]["on"]:
        parts.append(f"Mechanik ≈ {est['mech_dofs']} Unbekannte, grob {est['mech_seconds']:.0f} s und {est['mech_bytes'] / GB:.2f} GB (Faustformel, eher zu hoch)")
    st.info(" · ".join(parts) + f" · Spitzenspeicher ≈ {est['human']}")
    gb = est["bytes"] / GB
    avail = t3.available_gb()
    if gb > float(m["max_gb"]):
        st.error(f"Die Schätzung ({gb:.1f} GB) liegt über der eingestellten Grenze von {m['max_gb']:g} GB: der Worker würde abbrechen. Netz oder Ordnung verkleinern oder die Grenze anheben.")
    elif avail is not None and gb > 0.6 * avail:
        st.warning(f"Die Schätzung ({gb:.1f} GB) beansprucht mehr als 60 % des freien Hauptspeichers ({avail:.0f} GB); andere Prozesse könnten gestört werden.")


def _tab_run(ctx, work):
    m, ver = S.th3_model, S.th3_ver
    t3.normalise(m)
    st.subheader("Ordnung, Ausgabe und Speichergrenze")
    c = st.columns(3)
    m["order"] = int(c[0].number_input("Polynomordnung p", 1, 6, int(m["order"]), key=f"t3_p_{ver}"))
    m["grid"] = int(c[1].number_input("Auflösung der Feldschnitte (Punkte entlang der längsten Achse)", 12, 120, int(m["grid"]), 6, key=f"t3_grid_{ver}",
                                      help="Die Felder werden auf einem regelmäßigen 3D-Raster ausgewertet (Zellmitten). Das Raster ändert die Rechnung nicht."))
    m["max_gb"] = float(c[2].number_input("Speichergrenze (GB)", 0.1, 4096.0, float(m["max_gb"]), 1.0, key=f"t3_maxgb_{ver}",
                                          help="Der Worker bricht vor dem Rechnen ab, wenn die Schätzung darüber liegt."))
    errs = [t for lvl, t in t3.validate(m) if lvl == "error"]
    for e in errs:
        st.error(e)
    est_work = work / "estimate"
    if st.button("Netz prüfen und Speicher schätzen", key="t3_est", disabled=bool(errs)):
        with st.spinner("Vernetze …"):
            S.th3_est = run_estimate(ctx["py"], ctx["repo"], est_work, t3.job(m), int(ctx["threads"]))
    _estimate_box(S.get("th3_est"), m)
    if st.button("Rechnen", type="primary", key="t3_run", disabled=bool(errs)):
        try:
            work.mkdir(parents=True, exist_ok=True)
            (work / "job.json").write_text(json.dumps(t3.job(m), ensure_ascii=False), encoding="utf-8")
            S.th3_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script="fem_thermo3d_worker.py")
            S.th3_dir, S.th3_cancel = str(work), False
            S.th3_run_model = copy.deepcopy(m)
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("th3_msg"):
        (st.success if S.th3_msg[0] == "ok" else st.error)(S.th3_msg[1] + (" Die Ergebnisse stehen im Reiter 3." if S.th3_msg[0] == "ok" else ""))
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(work, tail=40))


def _running():
    proc, folder = S.th3_proc, Path(S.th3_dir)
    log = fr.read_log(folder)
    st.subheader("Rechnung läuft …")
    if proc.poll() is None:
        frac, label = fr.progress_info(log)
        st.progress(frac, text=label)
        st.code("\n".join(log.splitlines()[-10:]) or "(noch keine Ausgabe)")
        if S.get("th3_cancel"):
            st.info("Abbruch angefordert: die Rechnung hält nach dem laufenden Schritt an (ein einzelner Löserlauf lässt sich nicht unterbrechen).")
        elif st.button("Abbrechen", key="t3_cancel_btn"):
            fr.request_cancel(folder)
            S.th3_cancel = True
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        S.th3_proc = None
        ok = proc.returncode == 0 and not flags["errors"]
        S.th3_msg = ("ok", "Abgebrochen." if flags["cancelled"] else "Fertig.") if ok else ("error", f"Beendet mit Code {proc.returncode}: " + "; ".join(flags["errors"][:3]))
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
    model = S.get("th3_run_model") or S.th3_model
    st.caption(f"{meta.get('name', '')} · {meta.get('cells')} Zellen, p = {meta.get('order')} · hpfem {meta.get('hpfem', '?')} · Start {meta.get('started', '')}"
               + (f" · {meta['total_s']:.1f} s" if "total_s" in meta else ""))
    if res.get("cancelled"):
        st.warning("Die Rechnung wurde abgebrochen.")
        return
    if not res.get("finished"):
        st.info("Die Rechnung ist noch nicht fertig.")
        return
    npz_path = work / "fields3d.npz"
    npz = np.load(npz_path) if npz_path.is_file() else None
    if res.get("converged") is False:
        st.warning("Die thermo-optische Schleife hat die Toleranz nicht erreicht: Relaxation verkleinern oder mehr Iterationen zulassen.")
    sub = st.tabs(["Kennzahlen und Prüfung", "Schnitte", "Linien", "Verformung", "Konvergenz"])
    with sub[0]:
        sc, probes = res["scalars"], res.get("probes", {})
        ui.table_show(pd.DataFrame([(k, _fmt(v)) for k, v in sc.items()], columns=["Größe", "Wert"]))
        if model["case"] == "particle" and not model["thermal"]["selfconsistent"] and "Absorbierte Leistung (W)" in sc and "T_center" in probes:
            est = t3.particle_heat_estimate(model, float(sc["Absorbierte Leistung (W)"]))
            got = probes["T_center"] - float(model["thermal"]["T_ref"])
            st.markdown(f"**Prüfung Schätzung:** ΔT ≈ P/(4πκ)·(1/R − 1/R_s) = {est:.4g} K gegen {got:.4g} K im Kugelmittelpunkt aus der FEM-Rechnung "
                        f"(Abweichung {100 * (got - est) / est:+.1f} %). Näherung für eine gut leitende Kugel; das Netz ({meta.get('cells')} Zellen) beeinflusst beide Werte.")
        rows = []
        for lab, exp, key in t3.analytic_checks(model) + t3.heat_checks(model):
            if key in probes:
                got = probes[key]
                rows.append((lab, f"{got:.6g}", f"{exp:.6g}", f"{abs(got - exp) / abs(exp):.2e}" if exp else "–"))
        if rows:
            st.markdown("**Prüfung gegen die geschlossene Lösung:**")
            ui.table_show(pd.DataFrame(rows, columns=["Größe", "FEM", "geschlossen", "rel. Abweichung"]))
        elif model["case"] == "block":
            st.caption("Für diese Randbedingungen gibt es hier keine geschlossene Vergleichslösung (sie gilt für Symmetrieebenen x = 0, y = 0, z = 0, Zug nur auf x = W, gleichmäßige Temperatur; "
                       "Wärmequelle: Senken auf x = 0 und x = W oder nur auf x = 0).")
        if model["case"] == "particle":
            st.caption("Leistung und Energie für das ganze Gebiet (beim Viertelmodell ×4). Die Box endet an der Wärmesenke: die Temperatur gilt nur für diesen Abstand.")
    if npz is None:
        for t in sub[1:4]:
            with t:
                st.info("Keine Felddaten.")
    else:
        keys = t3.field_keys(npz.files)
        with sub[1]:
            c = st.columns(3)
            k = c[0].selectbox("Feld", keys, format_func=lambda k: t3.FIELDS[k][0], key="t3_field")
            axis = c[1].radio("Schnitt senkrecht zu", ["x", "y", "z"], index=1, horizontal=True, key="t3_axis")
            frac = c[2].slider("Lage (Anteil der Ausdehnung)", 0.0, 1.0, 0.5, 0.01, key="t3_frac")
            ui.show_fig(t3.fig_slice(npz, k, axis, frac), f"{k}_{axis}.png", "t3_dl_field")
            st.caption("Raster der Zellmitten, außerhalb des Netzes leer. Spannungen sind Zellwerte der Lösung (unstetig über Materialgrenzen). "
                       + ("Die PML-Schicht gehört zum Gebiet; das optische Feld dort ist nicht physikalisch." if model["case"] == "particle" else ""))
        with sub[2]:
            c = st.columns(4)
            k2 = c[0].selectbox("Feld", keys, format_func=lambda k: t3.FIELDS[k][0], key="t3_line_field")
            axis2 = c[1].radio("Richtung", ["x", "y", "z"], index=2, horizontal=True, key="t3_line_axis")
            fa = c[2].slider("Lage 1", 0.0, 1.0, 0.5, 0.01, key="t3_line_fa", help="Anteil der Ausdehnung der ersten übrigen Achse (in der Reihenfolge x, y, z)")
            fb = c[3].slider("Lage 2", 0.0, 1.0, 0.5, 0.01, key="t3_line_fb", help="Anteil der Ausdehnung der zweiten übrigen Achse")
            ui.show_fig(t3.fig_line(npz, k2, axis2, fa, fb), "linie3d.png", "t3_dl_line")
        with sub[3]:
            if "umag" in npz.files:
                c = st.columns(3)
                axis3 = c[0].radio("Schnitt senkrecht zu", ["x", "y", "z"], index=1, horizontal=True, key="t3_def_axis")
                frac3 = c[1].slider("Lage (Anteil der Ausdehnung)", 0.0, 1.0, 0.5, 0.01, key="t3_def_frac")
                span = float(max(np.ptp(npz["x"]), np.ptp(npz["y"]), np.ptp(npz["z"])))
                conv = 1.0 if str(npz["unit"]) == "nm" else 1e-3
                dmax = float(np.nanmax(npz["umag"])) * conv
                auto = max(1.0, 0.05 * span / dmax) if dmax > 0 else 1.0
                scale = c[2].number_input("Überhöhung der Verschiebung", 0.0, 1e9, float(f"{auto:.2g}"), format="%.6g", key=f"t3_def_scale_{auto:.3g}",
                                          help="Vorgabe: größte Verschiebung ≈ 5 % der Gebietsgröße. Die Verformung ist winzig (nm) und wird stark überhöht gezeichnet.")
                ui.show_fig(t3.fig_deformed(npz, axis3, frac3, scale), "verformt3d.png", "t3_dl_def")
            else:
                st.info("Ohne Mechanik keine Verformung.")
    with sub[4]:
        if res.get("history"):
            ui.show_fig(t3.fig_history(res["history"]), "konvergenz3d.png", "t3_dl_hist")
            ui.table_show(pd.DataFrame({"Iteration": list(range(1, len(res["history"]) + 1)), "größte Temperaturänderung (K)": res["history"]}).round(6))
            st.caption("Größte Temperaturänderung zwischen zwei Iterationen; die erste Zeile enthält den Sprung vom Startfeld (0 K) auf die erste Lösung.")
        else:
            st.info("Keine Iteration (nicht selbstkonsistent oder nur Mechanik).")
    st.download_button("Modell (JSON)", json.dumps(model, indent=1, ensure_ascii=False).encode("utf-8"), file_name="waerme_mechanik_3d_modell.json",
                       mime="application/json", key="t3_dl_model")
