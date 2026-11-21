"""Oberfläche der Photovoltaik-Auswertung (hpfem.pv): Einstellungen im Reiter „Rechnung“ und Unterreiter „Photovoltaik“ unter den Ergebnissen.

Die Rechnung selbst liegt im Worker (``fem_worker.pv_profile``, reine Nachbereitung eines Streupunkts), die Spektren und die Stromdichte in
``fem_pv.py``.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

import fem_materials as fm
import fem_pv as pv
from fem_ui import csv_bytes, show_fig, table_show

S = st.session_state


def default_absorber(model, names, lam_nm):
    """Das Material mit der größten Absorption (Im ε) bei lam_nm unter ``names``, sonst das erste."""
    best, best_im = names[0], -1.0
    for n in names:
        try:
            im = float(fm.eps_at(model["materials"][n], lam_nm).imag)
        except Exception:
            continue
        if im > best_im:
            best, best_im = n, im
    return best


def run_panel(sv, model, used, lam_nm, available=True):
    """Expander „Photovoltaik“ im Reiter Rechnung: schreibt ``sv["pv"]`` (None = aus oder {absorber, bins})."""
    names = [n for n in model["materials"] if n in used]
    with st.expander("Photovoltaik: Generationsrate und Stromdichte", expanded=bool(sv.get("pv"))):
        if not available:
            sv["pv"] = None
            st.caption("Braucht hpfem mit `absorbed_power_by_tag` (exakte Absorption) und `hpfem.pv`.")
            return
        on = st.checkbox("Generationsrate im Absorber berechnen", value=bool(sv.get("pv")), key="pv_on",
                         help="Nachbereitung jedes Punkts, kein zusätzlicher Lösungsschritt: G = absorbierte Leistung je Zelle / (ħω · Zellfläche) im Absorbermaterial "
                              "(hpfem.pv), seitlich gemittelt zu einem Tiefenprofil G(y). Im Reiter Ergebnisse gewichtet die App die Punkte mit einem Sonnenspektrum "
                              "und schätzt die Kurzschlussstromdichte. Sinnvoll bei einem Wellenlängendurchlauf.")
        if not on:
            sv["pv"] = None
            return
        old = sv.get("pv") or {}
        absorber = old.get("absorber") if old.get("absorber") in names else default_absorber(model, names, lam_nm)
        c = st.columns(2)
        absorber = c[0].selectbox("Absorber (Material)", names, index=names.index(absorber), key="pv_abs",
                                  help="Nur in Zellen dieses Materials zählt die Absorption als Generation (Ladungsträger werden gesammelt). Die PML und Zellen anderer "
                                       "Materialien tragen nichts bei.")
        bins = int(c[1].number_input("Stützstellen des Tiefenprofils", 10, 400, int(old.get("bins", 40)), 5, key="pv_bins",
                                     help="Streifen entlang y über das physikalische Gebiet; das Profil mittelt seitlich über die ganze Periode."))
        sv["pv"] = dict(absorber=absorber, bins=bins)


def _spectrum_choice(lam):
    """Gewähltes Spektrum (λ [nm], I [W/(m² nm)]) und Name; None bei Fehlern (mit Meldung)."""
    kind = st.radio("Spektrum", ["Schwarzkörper 5778 K (Näherung)", "Eigenes Spektrum (CSV)"], horizontal=True, key="pv_spec_kind")
    if kind.startswith("Schwarz"):
        total = st.number_input("Gesamtbestrahlungsstärke (W/m²)", 100.0, 2000.0, 1000.0, 50.0, key="pv_total",
                                help="Der Schwarzkörper wird so skaliert, dass sein Integral über 280–4000 nm diesen Wert hat. Er ist eine Näherung ohne "
                                     "Absorptionslinien der Sonne und der Atmosphäre: für belastbare Zahlen ein Normspektrum (AM1.5) als CSV laden.")
        grid = np.linspace(280.0, 4000.0, 1500)
        return (grid, pv.blackbody_irradiance(grid, total=total)), "Schwarzkörper 5778 K"
    up = st.file_uploader("CSV: λ [nm]; Irradianz [W/(m² nm)]", type=["csv", "txt", "dat"], key="pv_spec_file")
    if up is None:
        st.info("Spektrum als CSV hochladen (zwei Spalten, Semikolon, Komma oder Leerraum; Kommentarzeilen mit #).")
        return None, ""
    try:
        return pv.parse_spectrum_csv(up.getvalue().decode("utf-8-sig", errors="replace")), up.name
    except ValueError as exc:
        st.error(str(exc))
        return None, ""


def result_view(res):
    """Unterreiter „Photovoltaik“: Absorption im Absorber, J_sc und G(z) (spektral gewichtet oder je Punkt)."""
    got = pv.collect(res["points"])
    if got is None:
        st.info("Keine Photovoltaik-Daten in dieser Rechnung (Reiter 3: Expander „Photovoltaik“ einschalten, Wellenlängendurchlauf rechnen).")
        return
    lam, a, y, g = got
    absorber = next(p["pv"]["absorber"] for p in res["points"] if p.get("pv"))
    ratios = [p["pv"]["check_ratio"] for p in res["points"] if p.get("pv") and p["pv"].get("check_ratio")]
    if len(lam) < 2:
        st.warning("Nur ein Wellenlängenpunkt: ein Spektrum lässt sich nur mit einem Durchlauf über mehrere Wellenlängen gewichten. Gezeigt sind die Werte des Punkts.")
    spectrum, sname = _spectrum_choice(lam)
    if spectrum is None:
        return
    cov = pv.coverage(lam, spectrum)
    if cov < 1.0:
        st.warning(f"{(1 - cov) * 100:.0f} % der Wellenlängenpunkte liegen außerhalb des Spektrums und tragen nichts bei.")
    j, jmax = pv.jsc(lam, a, spectrum), pv.jsc_limit(lam, spectrum)
    c = st.columns(3)
    c[0].metric("J_sc (mA/cm²)", f"{j:.2f}")
    c[1].metric("bei A = 1 im Bereich (mA/cm²)", f"{jmax:.2f}")
    c[2].metric("Anteil", f"{(j / jmax * 100) if jmax > 0 else 0:.1f} %")
    show_fig(pv.fig_absorption(lam, a, spectrum, absorber), "pv_absorption.png", "dl_pv_abs")
    gsum = pv.spectral_profile(lam, g, spectrum)
    tab = pv.spectrum_table(lam, a, spectrum)
    c = st.columns([1, 1])
    with c[0]:
        st.markdown("**Generationsrate G(y), spektral gewichtet**")
        show_fig(pv.fig_profile(y, gsum), "pv_generation.png", "dl_pv_g")
    with c[1]:
        st.markdown("**Tabelle je Wellenlänge**")
        table_show(tab)
        st.download_button("Absorption und Stromdichte (CSV)", csv_bytes(tab), file_name="pv_spektrum.csv", mime="text/csv", key="dl_pv_tab")
        st.download_button("Generationsprofil G(y) (CSV)", csv_bytes(pv.profile_table(y, gsum)), file_name="pv_generation.csv", mime="text/csv", key="dl_pv_prof")
    st.caption(f"Spektrum: {sname}. J_sc = q Σ A_Absorber(λ) Φ(λ) mit Φ = I λ/(h c) und den Bandbreiten aus den Mittelpunkten der Punkte; jedes absorbierte Photon "
               f"erzeugt ein Paar, alle werden gesammelt (obere Grenze, keine Ladungsträgersimulation). G(y) in 1/(m³ s), seitlich über die Periode gemittelt, "
               f"nur im Material {absorber}. ∫G dy ħω stimmt mit der Absorption überein (Verhältnis {np.mean(ratios):.3f}, Abweichung durch die Streifen).")
