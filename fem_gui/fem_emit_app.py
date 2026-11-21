"""Emitters in the FEM model builder (Streamlit), hp-FEM 0.5. Called from fem_app.py:

* ``grating_model_panel`` / ``grating_run_panel`` / ``grating_results``: the emitter in the cell of the periodic mode (task "emitter",
  hpfem.grating.emit / dipole_emission / emission_pattern, M17), embedded in the tabs 1, 3 and 4 of fem_app.py.

Models, jobs and figures: fem_emit.py; worker: fem_emit_worker.py (inside fem_worker.py).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

import fem_emit as fe
import fem_materials as fm
import fem_run as fr
import fem_ui as ui

S = st.session_state


# ===================================================================================================================== periodic mode
def grating_model_panel(model, ver, lam_nm, supported):
    """Expander of tab 1: the emitter of the cell (position, width). Sets / removes model["emitter"]."""
    with st.expander("Emitter in der Zelle (hp-FEM 0.5, optional)", expanded="emitter" in model):
        if not supported:
            st.info("Emitter in Gittern brauchen hp-FEM ab 0.5 (hpfem.grating.dipole_emission).")
        on = st.checkbox("Gauß-Dipol in die Zelle setzen (Aufgabe „Emitter“ im Reiter 3)", value="emitter" in model, key=f"em_on_{ver}",
                         help="Ein einzelner Dipol in der periodischen Zelle: Purcell-Faktor, Anteile nach oben, unten und nicht abgestrahlt, Leistung in "
                              "Sammelaperturen und Abstrahlmuster. Das Netz wird um den Emitter verfeinert (Reiter 2 neu erzeugen). "
                              "Die Quelle liegt in einem verlustfreien Medium, 6σ von Grenzflächen und Rändern entfernt.")
        if not on:
            model.pop("emitter", None)
            return
        em = model["emitter"] = fe.emitter_complete(model.get("emitter", {}))
        c = st.columns(3)
        em["x_nm"] = float(c[0].number_input("x (nm)", value=float(em["x_nm"]), step=10.0, format="%.6g", key=f"em_x_{ver}", help="Entlang der Periode."))
        em["y_nm"] = float(c[1].number_input("y (nm)", value=float(em["y_nm"]), step=10.0, format="%.6g", key=f"em_y_{ver}",
                                             help="Senkrecht zu den Schichten; y = 0 ist die Oberkante des Schichtstapels."))
        em["sigma_nm"] = float(c[2].number_input("σ (nm)", 0.5, 500.0, float(em["sigma_nm"]), 1.0, key=f"em_s_{ver}", format="%.6g",
                                                 help="Breite der Gauß-Quelle (Stromdichte). Klein gegen die Wellenlänge im Medium (λ/40 ist üblich); "
                                                      "das Netz ist innerhalb 6σ etwa 2σ fein."))
        for lvl, text in fe.emitter_validate(model, em, lam_nm):
            (st.error if lvl == "error" else st.warning)(text)


def grating_run_panel(model, feats, lib):
    """Settings of the task "emitter" in tab 3 (stage, orientation, scan, apertures, directions)."""
    em = model["emitter"] = fe.emitter_complete(model["emitter"])
    em["stage"] = st.radio("Stufe", list(fe.STAGES), index=list(fe.STAGES).index(em["stage"]), format_func=fe.STAGES.get, key="em_stage")
    c = st.columns(2)
    em["moment"] = c[0].selectbox("Orientierung des Dipols", list(fe.MOMENTS), index=list(fe.MOMENTS).index(em["moment"]), format_func=fe.MOMENTS.get, key="em_moment",
                                  help="Achsen des Lösersystems: x entlang der Periode, y die Schichtnormale, z entlang der Linien.")
    if em["stage"] == "B":
        c[1].caption("Der Array-Scan löst viele Zellenprobleme (komplexe Bloch-Phasen); die Kosten stehen vorab im Protokoll.")
        sc = em["scan"]
        c = st.columns(4)
        sc["nodes"] = int(c[0].number_input("Stützstellen in β (je Feld)", 2, 24, int(sc["nodes"]), key="em_nodes",
                                            help="Die β-Richtung braucht mehr als kx: 6 reicht meist auf einige 10⁻⁴, 8 genauer."))
        sc["kx_nodes"] = int(c[1].number_input("Stützstellen in kx", 2, 16, int(sc["kx_nodes"]), key="em_kxn"))
        sc["depth"] = float(c[2].number_input("Tiefe der Kontur", 0.05, 2.0, float(sc["depth"]), 0.05, key="em_depth"))
        sc["workers"] = int(c[3].number_input("Parallele Lösungen", 1, 32, int(sc["workers"]), key="em_workers",
                                              help="Zellenprobleme gleichzeitig; der Speicher wächst mit der Zahl."))
        sc["guided"] = st.radio("Geführte Moden und Oberflächenplasmonen (Pole)", ["auto", "on", "off"], index=["auto", "on", "off"].index(sc.get("guided", "auto")),
                                horizontal=True, key="em_guided",
                                format_func={"auto": "automatisch (bei höherem Index als die Halbräume oder Metall)", "on": "immer", "off": "aus (schnell, nur ohne geführte Moden richtig)"}.get,
                                help="Der Array-Scan sucht Pole geführter Moden dicht an der reellen Achse und umgeht sie mit einer Beule der Kontur: das kostet "
                                     "Tausende Zellenprobleme (die Kostenschätzung im Protokoll zeigt es vorher). Ohne geführte Moden (z. B. Dielektrikum mit "
                                     "kleinerem Index als die Halbräume) „aus“ wählen; gibt es welche, bleibt „aus“ falsch.")
        c = st.columns(3)
        an = sc["angle_nodes"]
        sc["angle_nodes"] = [int(c[0].number_input("Winkelpunkte θ (Kanäle)", 4, 64, int(an[0]), 4, key="em_ant")),
                             int(c[1].number_input("Azimute φ (Vielfaches von 4)", 4, 128, int(an[1]), 4, key="em_anp"))]
        sc["symmetric"] = c[2].checkbox("Zelle spiegelsymmetrisch um den Dipol", value=bool(sc["symmetric"]), key="em_sym",
                                        help="Halbiert den Aufwand, wenn die Zelle um die x-Position des Emitters spiegelsymmetrisch ist (nicht bei Kristallen).")
    if em["stage"] in ("B", "C"):
        c = st.columns(2)
        txt = c[0].text_input("Sammelaperturen (NA, durch Komma)", ", ".join(f"{v:g}" for v in em["apertures"]), key="em_ap",
                              help="Leistung in den Kegel θ ≤ arcsin(NA / n) um die Normale; als Anteil der abgegebenen Leistung (Stufe B) bzw. von P_bulk (Stufe C).")
        try:
            em["apertures"] = [float(v) for v in txt.replace(";", ",").split(",") if v.strip()]
        except ValueError:
            st.error("Aperturen: Zahlen durch Komma.")
        if em["stage"] == "B":
            em["aperture_side"] = c[1].selectbox("Seite der Aperturen", ["up", "down"], index=["up", "down"].index(em["aperture_side"]), key="em_apside",
                                                 format_func={"up": "oben (Einfallsmedium)", "down": "unten (verlustfreies Substrat)"}.get)
    if em["stage"] == "A":
        c = st.columns(2)
        em["kx_over_k0"] = float(c[0].number_input("kx / k₀", -3.0, 3.0, float(em["kx_over_k0"]), 0.05, key="em_kx", help="Bloch-Wellenzahl der Dipolreihe."))
        em["beta_over_k0"] = float(c[1].number_input("β / k₀", -3.0, 3.0, float(em["beta_over_k0"]), 0.05, key="em_beta", help="Abhängigkeit e^{iβz} entlang der Linien."))
    if em["stage"] == "C":
        d = em["directions"]
        c = st.columns(3)
        t = c[0].text_input("Polarwinkel θ (°)", ", ".join(f"{v:g}" for v in d["theta_deg"]), key="em_dth")
        p = c[1].text_input("Azimute φ (°)", ", ".join(f"{v:g}" for v in d["phi_deg"]), key="em_dph")
        d["sides"] = c[2].multiselect("Seiten", ["up", "down"], default=d["sides"] or ["up"], key="em_dside", format_func={"up": "oben", "down": "unten"}.get)
        try:
            d["theta_deg"] = [float(v) for v in t.split(",") if v.strip()]
            d["phi_deg"] = [float(v) for v in p.split(",") if v.strip()]
        except ValueError:
            st.error("Winkel: Zahlen durch Komma.")
        st.caption("Eine ebene-Wellen-Lösung je Richtung und Polarisation (Reziprozität). Richtungen ins Substrat brauchen gerade Netzränder (Reiter 2: gekrümmte Elemente aus). "
                   "Für eine Apertur muss das θ-Raster mindestens bis zu ihrem Rand reichen (NA 0,5 in Luft: 30°); der Rand wird interpoliert, was dahinter fehlt, fehlt der Leistung.")
        if d["theta_deg"] and em["apertures"]:
            short = [na for na in em["apertures"] if na <= 1.0 and max(d["theta_deg"]) < float(np.degrees(np.arcsin(na))) - 1e-9]
            if short:
                st.warning("Das θ-Raster endet vor dem Rand der Apertur(en) NA " + ", ".join(f"{v:g}" for v in short) + " (in Luft): ihre Leistung wird unterschätzt.")
    c = st.columns(2)
    em["calibrate"] = c[0].checkbox("Rechenzeit vorab messen", value=bool(em.get("calibrate", True)), key="em_cal", disabled=em["stage"] != "B",
                                    help="Löst eine Probe (Zellenproblem und ebene Welle) und schätzt die Gesamtzeit; kostet etwa so viel wie drei Lösungen.")
    em["dry_run"] = c[1].checkbox("Nur die Kosten schätzen (nicht rechnen)", value=bool(em.get("dry_run", False)), key="em_dry")


def grating_results(view, res):
    """Tab 4 for an emitter result of the periodic mode."""
    em = res.get("emitter", {})
    cost = res.get("cost")
    if cost:
        st.info("Kosten: " + fe.cost_text(cost))
    if res.get("dry_run"):
        st.success("Nur die Kosten wurden geschätzt (Reiter 3: „Nur die Kosten schätzen“ ausschalten, um zu rechnen).")
        return
    for p in res.get("points", []):
        if "error" in p:
            st.error(f"λ = {p.get('wavelength', '?')} nm: {p['error']}")
    if res.get("cancelled"):
        st.warning("Die Rechnung wurde abgebrochen; es fehlen Punkte.")
    pts = fe.emission_points(res)
    if not pts:
        st.info("Keine auswertbaren Punkte.")
        return
    stage = em.get("stage", "B")
    st.caption(f"{res['meta']['model']} · Emitter bei x = {em['x_nm']:g} nm, y = {em['y_nm']:g} nm, σ = {em['sigma_nm']:g} nm · Orientierung {em['moment']} · Stufe {stage} · "
               f"p = {res['meta']['order']} · {res['meta']['cells']} Dreiecke · {len(pts)} Wellenlänge(n)")
    if stage == "B":
        if any(p.get("guided") for p in pts):
            st.caption("Der „nicht abgestrahlte“ Anteil ist absorbiert (verlustbehaftete Struktur) oder in geführte Moden gegangen (verlustfrei).")
        sub = st.tabs(["Purcell-Faktor", "Kanäle und Aperturen", "Tabelle und Export"])
        with sub[0]:
            ui.show_fig(fe.fig_emit_purcell(res), "purcell_zelle.png", "em_dl_p")
        with sub[1]:
            ui.show_fig(fe.fig_emit_fractions(res), "kanaele_zelle.png", "em_dl_c")
    elif stage == "A":
        sub = st.tabs(["Ordnungen", "Tabelle und Export"])
        with sub[0]:
            k = st.selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: f"{pts[i]['wavelength']:g} nm", key="em_a_lam")
            ui.show_fig(fe.fig_emit_orders(res, k), "ordnungen_emitter.png", "em_dl_o")
    else:
        sub = st.tabs(["Abstrahlmuster", "Tabelle und Export"])
        with sub[0]:
            c = st.columns(2)
            k = c[0].selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: f"{pts[i]['wavelength']:g} nm", key="em_c_lam")
            lg = c[1].checkbox("logarithmisch", key="em_c_log")
            ui.show_fig(fe.fig_emit_pattern(res, k, lg), "muster_zelle.png", "em_dl_pat")
    with sub[-1]:
        df = fe.emitter_dataframe(res)
        ui.table_show(df)
        st.download_button("Ergebnisse (CSV)", ui.csv_bytes(df), file_name="ergebnisse_emitter.csv", mime="text/csv", key="em_dl_csv")
        st.caption(f"Ordner mit allen Dateien: {view}")
