"""Emitters in the FEM model builder (Streamlit), hp-FEM 0.5. Called from fem_app.py:

* ``render_planar(ctx)``: the mode "Emitter über Schichtstapel" (no mesh: hpfem.dataports.planar_emitter_response): Purcell factor, radiation
  channels, detector regions (numerical apertures) and far field of a dipole above a stack of layers; the emitter response table can be
  downloaded as the BindOptiX envelope (JSON).
* ``grating_model_panel`` / ``grating_run_panel`` / ``grating_results``: the emitter in the cell of the periodic mode (task "emitter",
  hpfem.grating.emit / dipole_emission / emission_pattern, M17), embedded in the tabs 1, 3 and 4 of fem_app.py.

Models, jobs and figures: fem_emit.py; workers: fem_planar_worker.py and fem_emit_worker.py (inside fem_worker.py).
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


# ===================================================================================================================== planar mode
def _init():
    if "pl_model" in S:
        return
    S.pl_model = copy.deepcopy(fe.default_planar())
    S.pl_ver, S.pl_proc, S.pl_dir, S.pl_msg = 0, None, None, None


def _bump():
    S.pl_ver += 1


def _rename(model, old, new):
    model["materials"] = {(new if k == old else k): v for k, v in model["materials"].items()}
    for key in ("cover", "substrate"):
        if model.get(key) == old:
            model[key] = new
    for ly in model["layers"]:
        if ly["material"] == old:
            ly["material"] = new


def _used(m):
    return {m["cover"], m["substrate"]} | {ly["material"] for ly in m["layers"]}


def render_planar(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    work = Path(ctx["work"]) / "planar"
    st.title("FEM-Modellwerkstatt: Emitter über Schichtstapel")
    st.caption("Ein Dipol in einem verlustfreien Medium über einem Stapel aus Schichten und Substrat: Purcell-Faktor (Auslöschung nahe Metall), "
               "Abstrahlung nach oben und ins Substrat, Anteile in Detektorbereiche (numerische Aperturen) und Fernfeld, halbanalytisch "
               "(Sommerfeld/Fresnel, hpfem.dataports.planar_emitter_response, ohne Netz). Gilt auch als Referenz für die Emitter in Gittern und "
               "Rotationskörpern.")
    if S.pl_proc is not None:
        return _running(work)
    feats = ctx["FEATS"]
    if not (feats.get("dataports") or ctx["lib"].get("error")):
        st.warning("Das eingestellte Python hat kein hpfem mit hpfem.dataports (hp-FEM 0.5 oder neuer). Anzeigen geht trotzdem.")
    t1, t2, t3, t4 = st.tabs(["1 Modell", "2 Rechnung", "3 Ergebnisse", "4 Info"])
    with t1:
        _tab_model()
    with t2:
        _tab_run(ctx, work)
    with t3:
        _tab_results(work)
    with t4:
        st.markdown(INFO_PLANAR)


def _tab_model():
    m, ver = S.pl_model, S.pl_ver
    left, right = st.columns([5, 4])
    with left:
        presets = fe.planar_presets()
        c = st.columns([3, 1])
        pick = c[0].selectbox("Vorlage", ["– auswählen –"] + list(presets), key="pl_preset")
        if c[1].button("Laden", key="pl_preset_load", disabled=pick.startswith("–")):
            S.pl_model = copy.deepcopy(presets[pick])
            _bump()
            st.rerun()
        m["name"] = st.text_input("Name", value=m["name"], key=f"pl_name_{ver}")
        st.subheader("Materialien")
        lam0 = fe.sweep_list(m["sweep"])[0]
        ui.material_editor(m, ver, lam0, _used(m), _rename, _bump, prefix="pl_")
        names = list(m["materials"])
        st.subheader("Stapel (von oben nach unten)")
        c = st.columns(2)
        m["cover"] = c[0].selectbox("Einfallsmedium = Wirt des Emitters (verlustfrei)", names, index=names.index(m["cover"]) if m["cover"] in names else 0,
                                    key=f"pl_cov_{ver}")
        m["substrate"] = c[1].selectbox("Substrat", names, index=names.index(m["substrate"]) if m["substrate"] in names else 0, key=f"pl_sub_{ver}",
                                        help="Der Halbraum unter den Schichten. Ist er verlustfrei, rechnet die Bibliothek auch die Abstrahlung hinein "
                                             "(Detektoren auf der Substratseite, überkritische Winkel).")
        for i, ly in enumerate(list(m["layers"])):
            c = st.columns([3, 2, 1])
            ly["material"] = c[0].selectbox(f"Schicht {i + 1}", names, index=names.index(ly["material"]) if ly["material"] in names else 0, key=f"pl_ly_m_{ver}_{i}")
            ly["thickness_nm"] = float(c[1].number_input("Dicke (nm)", 0.1, 1e5, float(ly["thickness_nm"]), 1.0, key=f"pl_ly_t_{ver}_{i}", format="%.6g"))
            if c[2].button("Entfernen", key=f"pl_ly_del_{ver}_{i}"):
                m["layers"].pop(i)
                _bump()
                st.rerun()
        if st.button("Schicht hinzufügen", key=f"pl_ly_add_{ver}"):
            m["layers"].append({"material": names[0], "thickness_nm": 50.0})
            _bump()
            st.rerun()
        st.subheader("Wellenlänge und Höhen")
        sw = m["sweep"]
        sw["mode"] = st.radio("Wellenlänge", ["single", "wavelength"], index=["single", "wavelength"].index(sw["mode"]), horizontal=True, key=f"pl_swm_{ver}",
                              format_func={"single": "ein Wert", "wavelength": "Durchlauf"}.get)
        c = st.columns(3)
        sw["start"] = float(c[0].number_input("λ (von) nm" if sw["mode"] != "single" else "λ (nm)", 100.0, 20000.0, float(sw["start"]), 10.0, key=f"pl_swa_{ver}", format="%.6g"))
        if sw["mode"] != "single":
            sw["stop"] = float(c[1].number_input("λ bis (nm)", 100.0, 20000.0, float(sw["stop"]), 10.0, key=f"pl_swb_{ver}", format="%.6g"))
            sw["n"] = int(c[2].number_input("Punkte", 2, 200, int(sw["n"]), key=f"pl_swn_{ver}"))
        c = st.columns(3)
        hh = m["heights"]
        hh["start"] = float(c[0].number_input("Höhe von (nm)", 0.5, 1e5, float(hh["start"]), 5.0, key=f"pl_ha_{ver}", format="%.6g",
                                              help="Abstand des Dipols über der Oberkante des Stapels (> 0)."))
        hh["stop"] = float(c[1].number_input("Höhe bis (nm)", 0.5, 1e5, float(hh["stop"]), 10.0, key=f"pl_hb_{ver}", format="%.6g"))
        hh["n"] = int(c[2].number_input("Höhen", 1, 200, int(hh["n"]), key=f"pl_hn_{ver}"))
        st.subheader("Detektorbereiche")
        st.caption("Numerische Apertur NA = n sin θ im Medium der Seite (θ vom Lot): oben im Einfallsmedium, unten im Substrat. Bereiche einer Seite dürfen sich nicht überlappen.")
        for i, d in enumerate(list(m["detectors"])):
            c = st.columns([3, 2, 1, 1, 1])
            d["name"] = c[0].text_input("Name", value=d["name"], key=f"pl_d_n_{ver}_{i}")
            d["side"] = c[1].selectbox("Seite", ["cover", "substrate"], index=["cover", "substrate"].index(d["side"]), key=f"pl_d_s_{ver}_{i}",
                                       format_func={"cover": "oben (Einfallsmedium)", "substrate": "unten (Substrat)"}.get)
            d["na_min"] = float(c[2].number_input("NA min", 0.0, 3.0, float(d["na_min"]), 0.05, key=f"pl_d_a_{ver}_{i}"))
            d["na_max"] = float(c[3].number_input("NA max", 0.0, 3.0, float(d["na_max"]), 0.05, key=f"pl_d_b_{ver}_{i}"))
            if c[4].button("Entfernen", key=f"pl_d_del_{ver}_{i}"):
                m["detectors"].pop(i)
                _bump()
                st.rerun()
        if st.button("Detektor hinzufügen", key=f"pl_d_add_{ver}"):
            m["detectors"].append({"name": f"Detektor {len(m['detectors']) + 1}", "side": "cover", "na_min": 0.0, "na_max": 1.0})
            _bump()
            st.rerun()
        with st.expander("Genauigkeit der Quadratur"):
            c = st.columns(2)
            m["theta_points"] = int(c[0].number_input("Stützstellen in θ (je Intervall)", 16, 400, int(m.get("theta_points", 96)), 8, key=f"pl_tp_{ver}",
                                                      help="Fernfeldgitter. Mehr Punkte, wenn die Rechnung A_tot < A_rad meldet."))
            m["phi_points"] = int(c[1].number_input("Azimute", 4, 64, int(m.get("phi_points", 16)), 4, key=f"pl_pp_{ver}"))
    with right:
        for lvl, text in fe.planar_validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        try:
            ui.show_fig(_stack_figure(m), "stapel.png", f"pl_stack_dl_{ver}")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")
        rows = []
        for n_ in _used(m):
            try:
                e = fm.eps_at(m["materials"][n_], fe.sweep_list(m["sweep"])[0])
                rows.append({"Material": n_, "n": round(float(np.sqrt(e + 0j).real), 4), "k": round(float(abs(np.sqrt(e + 0j).imag)), 4)})
            except Exception as exc:
                rows.append({"Material": n_, "n": None, "k": str(exc)[:50]})
        ui.table_show(rows)


def _stack_figure(m):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    hs = fe.heights_list(m["heights"])
    top = max(max(hs) * 1.15, 50.0)
    thick = [float(ly["thickness_nm"]) for ly in m["layers"]]
    bottom = -(sum(thick) + max(sum(thick) * 0.5, 80.0))
    palette = plt.get_cmap("tab10")
    color = {n: palette(i % 10) for i, n in enumerate(m["materials"])}
    ax.add_patch(Rectangle((0, 0), 1, top, facecolor=color[m["cover"]], alpha=0.25))
    ax.text(0.5, top * 0.95, f"{m['cover']} (Wirt)", ha="center", va="top", fontsize=9)
    y = 0.0
    for ly in m["layers"]:
        t = float(ly["thickness_nm"])
        ax.add_patch(Rectangle((0, y - t), 1, t, facecolor=color[ly["material"]], alpha=0.6, edgecolor="k"))
        ax.text(0.5, y - t / 2, f"{ly['material']}, {t:g} nm", ha="center", va="center", fontsize=8)
        y -= t
    ax.add_patch(Rectangle((0, bottom), 1, y - bottom, facecolor=color[m["substrate"]], alpha=0.4))
    ax.text(0.5, (y + bottom) / 2, f"{m['substrate']} (Substrat)", ha="center", va="center", fontsize=9)
    ax.plot([0.5] * len(hs), hs, "r*", ms=7, label="Dipolhöhen")
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_ylabel("z (nm), 0 = Oberkante des Stapels")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _tab_run(ctx, work):
    m = S.pl_model
    errs = [t for lvl, t in fe.planar_validate(m) if lvl == "error"]
    lams, hs = fe.sweep_list(m["sweep"]), fe.heights_list(m["heights"])
    st.markdown(f"**{len(lams)} Wellenlänge(n) × {len(hs)} Höhen.** Etwa 5 s je Wellenlänge bei 96 Stützstellen in θ (die Zeit hängt kaum von der Zahl der Höhen ab).")
    for e in errs:
        st.error(e)
    lib_ok = ctx["FEATS"].get("dataports") or bool(ctx["lib"].get("error"))
    if st.button("Rechnung starten", type="primary", key="pl_run", disabled=bool(errs) or not lib_ok):
        try:
            work.mkdir(parents=True, exist_ok=True)
            fr.write_job(work, fe.planar_job(m))
            for f in work.glob("table_*.json"):
                f.unlink()
            S.pl_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script="fem_planar_worker.py")
            S.pl_dir, S.pl_cancel = str(work), False
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("pl_msg"):
        (st.success if S.pl_msg[0] == "ok" else st.error)(S.pl_msg[1] + " Die Ergebnisse stehen im Reiter 3.")
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(work, tail=40))


def _running(work):
    proc, folder = S.pl_proc, Path(S.pl_dir)
    log = fr.read_log(folder)
    st.subheader("Rechnung läuft …")
    if proc.poll() is None:
        frac, label = fr.progress_info(log)
        st.progress(frac, text=label)
        st.code("\n".join(log.splitlines()[-10:]) or "(noch keine Ausgabe)")
        if S.get("pl_cancel"):
            st.info("Abbruch angefordert: die Rechnung hält nach der laufenden Wellenlänge an.")
        elif st.button("Abbrechen", key="pl_cancel_btn"):
            fr.request_cancel(folder)
            S.pl_cancel = True
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        S.pl_proc = None
        ok = proc.returncode == 0 and not flags["errors"]
        S.pl_msg = ("ok", "Abgebrochen." if flags["cancelled"] else "Fertig.") if ok else ("error", f"Beendet mit Code {proc.returncode}: " + "; ".join(flags["errors"][:3]))
        st.rerun()


def _tab_results(work):
    res_file = work / "results.json"
    if not res_file.is_file():
        st.info("Noch keine Ergebnisse: im Reiter 2 rechnen.")
        return
    res = json.loads(res_file.read_text(encoding="utf-8"))
    pts = [p for p in res["points"] if "error" not in p]
    for p in res["points"]:
        if "error" in p:
            st.error(f"λ = {p['lam_nm']:g} nm: {p['error']}")
    if not pts:
        return
    st.caption(f"{res['meta'].get('name', '')} · {len(pts)} Wellenlänge(n) · hpfem {res['meta'].get('hpfem', '?')} · Start {res['meta'].get('started', '')}")
    sub = st.tabs(["Purcell-Faktor", "Kanäle und Detektoren", "Abstrahlmuster", "Tabelle und Export"])
    with sub[0]:
        ui.show_fig(fe.fig_planar_purcell(pts), "purcell.png", "pl_dl_p")
        st.caption("Auf das freie Dipol im selben Medium bezogen. F_∥: Dipol parallel zur Oberfläche, F_⊥: senkrecht. Nahe Metall fällt die "
                   "Strahlung, die Auslöschung (Energietransfer in Plasmonen und Verluste) lässt F wachsen, ohne dass Licht herauskommt: dazu die Kanäle.")
    lam_labels = [f"{p['lam_nm']:g} nm" for p in pts]
    with sub[1]:
        k = st.selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: lam_labels[i], key="pl_res_lam")
        ui.show_fig(fe.fig_planar_channels(pts[k]), "kanaele.png", "pl_dl_c")
        if not pts[k].get("substrate_lossless", True):
            st.info("Das Substrat ist verlustbehaftet: es strahlt nichts aus, seine Leistung zählt zu „absorbiert“.")
        conv = pts[k]["convergence"]
        st.caption(f"Prüfung der Bibliothek: min. Eigenwert von A_tot − A_rad = {conv['energy']:.2e} der Spur (≈ 0 verlustfrei ohne geführte Moden, > 0 bei "
                   f"Absorption oder geführten Moden); Fernfeld gegen adaptive Integration: {conv['radiated_check']:.1e}.")
    with sub[2]:
        c = st.columns(3)
        k = c[0].selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: lam_labels[i], key="pl_pat_lam")
        hlist = pts[k]["heights_nm"]
        hi = c[1].selectbox("Höhe", range(len(hlist)), format_func=lambda i: f"{hlist[i]:g} nm", key="pl_pat_h")
        ori = c[2].selectbox("Orientierung", ["isotropic", "parallel", "perp"], key="pl_pat_o",
                             format_func={"isotropic": "isotrop", "parallel": "parallel zur Oberfläche", "perp": "senkrecht"}.get)
        if pts[k].get("far"):
            ui.show_fig(fe.fig_planar_pattern(pts[k], hi, ori), "muster.png", "pl_dl_pat")
            st.caption("dP/dΩ in Einheiten der Leistung des freien Dipols je Raumwinkel, über den Azimut gemittelt; oben: Einfallsmedium, unten: Substrat. "
                       "Der Winkel läuft vom Lot der jeweiligen Seite.")
    with sub[3]:
        k = st.selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: lam_labels[i], key="pl_tab_lam")
        df = fe.planar_dataframe(pts, k)
        ui.table_show(df)
        st.download_button("Tabelle (CSV)", ui.csv_bytes(df), file_name=f"emitter_planar_{pts[k]['lam_nm']:g}nm.csv", mime="text/csv", key="pl_dl_csv")
        env = fe.planar_table_json(work, pts[k]["index"])
        if env:
            st.download_button("Emitter-Antworttabelle (JSON-Umschlag raytatouille-result, EmitterResponseTable)", env.encode("utf-8"),
                               file_name=f"emitter_response_{pts[k]['lam_nm']:g}nm.json", mime="application/json", key="pl_dl_env",
                               help="A_tot, A_rad und A_det auf dem Gitter (0, 0, Höhen in µm), mit Herkunftsangaben; das Datenport-Format für BindOptiX "
                                    "(hpfem.dataports).")
        st.download_button("Modell (JSON)", json.dumps(S.pl_model, indent=1, ensure_ascii=False).encode("utf-8"), file_name="emitter_planar_modell.json",
                           mime="application/json", key="pl_dl_model")


INFO_PLANAR = """
**Was gerechnet wird.** Ein Punktdipol (frei in einem verlustfreien Medium, z. B. Wasser oder Luft) über einem Stapel aus Schichten und einem Substrat.
Die Bibliothek (`hpfem.dataports.planar_emitter_response`) löst die Sommerfeld-Integrale halbanalytisch (Chance–Prock–Silbey) und liefert drei Matrizen je Höhe:

* **A_tot** = diag(F_∥, F_∥, F_⊥): die gesamte abgegebene Leistung, auf den freien Dipol bezogen (Purcell-Faktor).
* **A_rad**: die ins Fernfeld abgestrahlte Leistung (oben und, bei verlustfreiem Substrat, unten, auch überkritische Winkel). A_tot − A_rad ist die
  absorbierte oder in geführte Moden (Plasmonen, Wellenleiter) abgegebene Leistung.
* **A_det**: der Teil von A_rad, der in einen Detektorbereich fällt (numerische Apertur), z. B. ein Objektiv.

**Koordinaten.** z = 0 an der Oberkante des Stapels, der Dipol liegt bei den gewählten Höhen darüber. x und y sind gleichberechtigt (Schicht-Invarianz).
Der isotrope Dipol ist der Mittelwert (2 F_∥ + F_⊥) / 3.

**Zweck.** Referenz für die Emitter in Gittern (Reiter „Rechnung“ der periodischen Struktur) und Rotationskörpern: flache Schicht ohne Struktur
gegen die Rechnung mit Netz. Der JSON-Umschlag im Export ist das Datenport-Format der Bibliothek für BindOptiX.

**Grenzen.** Isotrope Schichten (keine Kristalle); der Wirt muss verlustfrei sein; ein verlustbehaftetes Substrat strahlt nichts aus.
"""


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
