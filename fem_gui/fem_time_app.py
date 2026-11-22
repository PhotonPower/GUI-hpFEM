"""Time domain in the FEM model builder (Streamlit), hp-FEM 0.5. Called from fem_app.py: ``render_time(ctx)``.

hpfem.TimeDomain2D / TimeDomain3D (Newmark-beta, implicit): a pulse or continuous wave from a current source through a structure with PEC /
absorbing / natural boundaries; probe signals E(t), their spectra, the energy, and field frames with a slider and an animation. Models, meshing and
figures: fem_time.py; worker: fem_time_worker.py.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

import fem_run as fr
import fem_time as ft
import fem_ui as ui
import fem_wg as fw
import fem_wg_app as fwa

S = st.session_state

INFO = """
**Was gerechnet wird.** Die Wellengleichung für das elektrische Feld im Zeitbereich, ε E″ + σ E′ + rot(μ⁻¹ rot E) = −J g′(t), mit hp-Elementen im
Raum (Nédélec) und der impliziten Newmark-Regel in der Zeit (`hpfem.TimeDomain2D/3D`). Mit β = ¼, γ = ½ ist das die Trapezregel: bedingungslos stabil,
ohne numerische Dämpfung (die Energie bleibt ohne Verlust und Absorber erhalten), aber mit einem Phasenfehler von etwa (ω Δt)²/12 je Bogenmaß.
Der Zeitschritt ist deshalb eine Frage der **Genauigkeit**, nicht der Stabilität: 20–40 Schritte je Periode der höchsten Frequenz (Voreinstellung 40).
γ > ½ (mit β ≥ (γ + ½)²/4) dämpft hohe Frequenzen numerisch.

**Quelle.** Eine Stromdichte J(x) = J₀ exp(−|x − x_s|²/(2 s²)) p mit dem Zeitsignal g(t); eingespeist wird −J g′(t). Signale: modulierter Gauß-Puls
sin(ω(t − t₀)) exp(−(t − t₀)²/(2 τ²)) mit Trägerwelle bei λ₀; Gauß-Puls exp(−(t − t₀)²/(2 τ²)) (breitbandig, bis etwa 4/(2π τ)); Dauerstrich
sin(ωt) mit weichem (logistischem) Einschalten. Die Amplitude J₀ ist in A/m² angegeben; die Felder skalieren linear damit. Das Verhältnis
„Sonde zu Quelle“ im Spektrum ist davon unabhängig.

**Materialien.** Der Löser rechnet nur konstante, reelle Permittivitäten (μ = 1) und eine Leitfähigkeit σ (S/m) je Material. Alle Materialien werden
einmal bei λ₀ ausgewertet; Dispersion gibt es nicht (Metalle mit ε < 0 sind nicht möglich). Verlust Im ε wird nur als Leitfähigkeit akzeptiert; die
Umrechnung σ = ε₀ ω₀ Im ε ist nur für schmalbandige Pulse bei λ₀ richtig.

**Ränder.** PEC (tangentiales E = 0), absorbierend (Silver-Müller erster Ordnung: nur für senkrecht auftreffende Wellen vollständig, schräge Anteile werden
teilweise zurückgeworfen; keine PML) und PMC (natürlicher Rand, magnetische Wand). 2D: ein Schnitt, E in der Ebene (H_z senkrecht), Ergebnisse je Meter in z.
Die Energie ist (vᵀ M v + uᵀ S u)/2 der Bibliothek.

**Netz.** „Zellen je Wellenlänge“ bezieht sich auf die kürzeste Wellenlänge des Pulses (c / f_max, f_max aus Träger und Pulsbreite; im Material durch n geteilt).
Ein breitbandiger oder sehr kurzer Puls verlangt deshalb ein feines Netz, auch wenn die Trägerwelle grob wäre. 3D: Würfel mit kugelförmigem Einschluss
(`box_with_ball`, gekrümmte Zellen); das Netz ist sehr grob (die Faktorisierung wächst schnell, Grenze 30 000 Freiheitsgrade) und es gibt keine
freie Geometrie.

**Spektrum.** Aus dem Signal der Sonde per FFT (Nullauffüllung, optional Hann-Fenster). Die Frequenzauflösung ist 1/T mit der Rechendauer T. Wenn das Signal
bei Rechenende nicht abgeklungen ist (Resonator, Dauerstrich), stört das Abschneiden: Hann-Fenster oder längere Rechnung. „Übertragung“ teilt durch das
Spektrum der Quelle und zeigt nur Frequenzen, bei denen die Quelle mindestens 2 % ihres Maximums hat. Das Spektrum ist **nicht** gegen eine Rechnung im
Frequenzbereich geprüft; getestet sind die Spitzenfrequenz der Quelle und die Laufzeit eines Pulses im Streifen.
"""


def _init():
    if "tm_model" in S:
        return
    S.tm_model = copy.deepcopy(ft.default_model())
    S.tm_ver, S.tm_proc, S.tm_dir, S.tm_msg = 0, None, None, None


def _bump():
    S.tm_ver += 1


def _rename(model, old, new):
    model["materials"] = {(new if k == old else k): v for k, v in model["materials"].items()}
    model["sigma"] = {(new if k == old else k): v for k, v in model.get("sigma", {}).items()}
    if model["background"] == old:
        model["background"] = new
    for s in model["shapes"]:
        if s["material"] == old:
            s["material"] = new
    if model["sphere"]["material"] == old:
        model["sphere"]["material"] = new


def render_time(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    work = Path(ctx["work"]) / "time"
    st.title("FEM-Modellwerkstatt: Zeitbereich")
    st.caption("Pulse und Dauerstrich im Zeitbereich (hp-FEM TimeDomain2D/3D): Quelle, Ränder, Sonden, Spektren, Energie und Feldfilme.")
    if S.tm_proc is not None:
        return _running()
    if not (ctx["FEATS"].get("time") or ctx["lib"].get("error")):
        st.warning("Das eingestellte Python hat kein hpfem mit TimeDomain2D. Anzeigen geht trotzdem.")
    t1, t2, t3, t4, t5 = st.tabs(["1 Modell", "2 Quelle und Zeit", "3 Rechnung", "4 Ergebnisse", "5 Info"])
    with t1:
        _tab_model()
    with t2:
        _tab_source()
    with t3:
        _tab_run(ctx, work)
    with t4:
        _tab_results(work)
    with t5:
        st.markdown(INFO)


# ===================================================================================================================== model tab
def _num(col, label, key, value, step=10.0, lo=-1e6, hi=1e6, help=None):
    return float(col.number_input(label, lo, hi, float(value), step, key=key, format="%.6g", help=help))


def _boundaries(m, ver):
    sides = ft.SIDES_2D if m["dim"] == "2d" else ft.SIDES_3D
    bd = m["boundary"] if m["dim"] == "2d" else m["boundary3"]
    cols = st.columns(len(sides) if m["dim"] == "2d" else 3)
    for i, (k, (label, _)) in enumerate(sides.items()):
        c = cols[i % len(cols)]
        bd[k] = c.selectbox(f"Rand {label}", list(ft.BOUNDARY), index=list(ft.BOUNDARY).index(bd[k]) if bd[k] in ft.BOUNDARY else 0,
                            format_func=lambda v: ft.BOUNDARY[v].split(" (")[0], key=f"tm_bd_{m['dim']}_{k}_{ver}",
                            help="PEC: ideal leitend (tangentiales E = 0). Absorbierend: Silver-Müller erster Ordnung (nur senkrechter Einfall vollständig). "
                                 "PMC: natürlicher Rand der Formulierung (magnetische Wand).")


def _tab_model():
    m, ver = S.tm_model, S.tm_ver
    left, right = st.columns([6, 5])
    with left:
        presets = ft.presets()
        c = st.columns([3, 1])
        pick = c[0].selectbox("Vorlage", ["– auswählen –"] + list(presets), key="tm_preset")
        if c[1].button("Laden", key="tm_preset_load", disabled=pick.startswith("–")):
            S.tm_model = copy.deepcopy(presets[pick])
            _bump()
            st.rerun()
        m["name"] = st.text_input("Name", value=m["name"], key=f"tm_name_{ver}")
        m["dim"] = st.radio("Dimension", list(ft.DIMS), index=list(ft.DIMS).index(m["dim"]), format_func=ft.DIMS.get, key=f"tm_dim_{ver}", horizontal=True,
                            help="2D: ein z-invarianter Schnitt (E in der Ebene). 3D: Würfel mit Kugel und sehr grobem Netz; die Faktorisierung wächst schnell.")
        st.subheader("Materialien (konstant, reell; ausgewertet bei λ₀)")
        lam0 = float(m["source"]["lambda_nm"])
        ui.material_editor(m, ver, lam0, ft.used_materials(m), _rename, _bump, prefix="tm_")
        names = list(m["materials"])
        st.markdown("**Leitfähigkeit σ (S/m)** je verwendetem Material: dämpft das Feld (σ E′ im Löser), z. B. 10⁵ S/m für ein schlechtes Metall.")
        cs = st.columns(3)
        for i, n in enumerate(sorted(ft.used_materials(m))):
            if n in m["materials"]:
                m["sigma"][n] = float(cs[i % 3].number_input(f"σ {n}", 0.0, 1e9, float(m["sigma"].get(n, 0.0)), 1000.0, format="%.6g", key=f"tm_sg_{ver}_{n}"))
        m["loss_as_sigma"] = st.checkbox("Verlust Im ε als Leitfähigkeit σ = ε₀ ω₀ Im ε abbilden (nur schmalbandig bei λ₀)", value=bool(m.get("loss_as_sigma")),
                                         key=f"tm_las_{ver}")
        st.subheader("Gebiet")
        c = st.columns(3)
        m["background"] = c[0].selectbox("Hintergrund", names, index=names.index(m["background"]) if m["background"] in names else 0, key=f"tm_bg_{ver}")
        if m["dim"] == "2d":
            m["length_nm"] = _num(c[1], "Länge L (nm), x von 0 bis L", f"tm_L_{ver}", m["length_nm"], 100.0, 50.0)
            m["width_nm"] = _num(c[2], "Breite B (nm), y von −B/2 bis B/2", f"tm_W_{ver}", m["width_nm"], 100.0, 50.0)
            st.subheader("Formen (spätere überdecken frühere)")
            for i, s in enumerate(list(m["shapes"])):
                with st.container(border=True):
                    fwa._shape_editor(m, i, s, names, ver, bump=_bump, prefix="tm_s")
            c = st.columns([2, 1])
            new_type = c[0].selectbox("Neue Form", list(fw.SHAPES), format_func=fw.SHAPES.get, key=f"tm_newtype_{ver}", label_visibility="collapsed")
            if c[1].button("Form hinzufügen", key=f"tm_add_{ver}"):
                m["shapes"].append({"type": new_type, "material": names[-1] if names else ""})
                _bump()
                st.rerun()
        else:
            sp = m["sphere"]
            st.subheader("Kugel im Kasten (hpfem.box_with_ball)")
            c = st.columns(3)
            sp["material"] = c[0].selectbox("Material der Kugel", names, index=names.index(sp["material"]) if sp["material"] in names else 0, key=f"tm_spm_{ver}")
            sp["radius_nm"] = _num(c[1], "Radius (nm)", f"tm_spr_{ver}", sp["radius_nm"], 10.0, 1.0)
            sp["outer_factor"] = int(c[2].number_input("Kasten: Halbkante in Radien (ganzzahlig)", 2, 8, int(sp["outer_factor"]), key=f"tm_spo_{ver}",
                                                        help="Der Kasten ist [−R, R]³ mit R = Zahl · Radius."))
            c = st.columns(2)
            sp["half_factor"] = _num(c[0], "Gleichmäßiges Netz außerhalb von (Radien)", f"tm_sph_{ver}", sp["half_factor"], 0.1, 1.0, 8.0)
            sp["cells"] = int(c[1].number_input("Zellen je Radius", 1, 4, int(sp["cells"]), key=f"tm_spc_{ver}",
                                                help="1 genügt für ein grobes Bild mit höherer Ordnung; mehr Zellen vergrößern das 3D-Problem stark."))
        st.subheader("Ränder")
        _boundaries(m, ver)
    with right:
        for lvl, text in ft.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        try:
            ui.show_fig(ft.fig_preview(m), "zeitbereich_modell.png", f"tm_prev_dl_{ver}")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")
        rows = []
        for n_ in sorted(ft.used_materials(m)):
            if n_ in m["materials"]:
                try:
                    e, sg = ft.eps_sigma(m, n_)
                    rows.append({"Material": n_, "ε (Re)": round(e, 4), "n": round(e**0.5, 4) if e > 0 else None, "σ (S/m)": sg})
                except Exception as exc:
                    rows.append({"Material": n_, "ε (Re)": str(exc)})
        if rows:
            ui.table_show(rows)


# ===================================================================================================================== source and time tab
def _tab_source():
    m, ver = S.tm_model, S.tm_ver
    s, tm, me = m["source"], m["time"], m["mesh"]
    three = m["dim"] == "3d"
    st.subheader("Zeitsignal")
    c = st.columns(4)
    s["kind"] = c[0].selectbox("Signal", list(ft.SIGNALS), index=list(ft.SIGNALS).index(s["kind"]), format_func=ft.SIGNALS.get, key=f"tm_kind_{ver}")
    s["lambda_nm"] = _num(c[1], "Referenzwellenlänge λ₀ (nm)", f"tm_lam_{ver}", s["lambda_nm"], 10.0, 1.0,
                          help="Träger des Pulses und Wellenlänge, bei der die Materialien ausgewertet werden.")
    s["width_fs"] = _num(c[2], "Breite τ (fs)" if s["kind"] != "cw" else "Einschaltzeit τ (fs)", f"tm_w_{ver}", s["width_fs"], 0.5, 0.01, 1e5,
                         "Die Breite der Hüllkurve; das Spektrum ist etwa 1/(2π τ) breit.")
    s["delay"] = _num(c[3], "Verzögerung t₀ (in τ)", f"tm_d_{ver}", s["delay"], 0.5, 1.0, 50.0, "Der Puls ist bei t = 0 bis auf exp(−t₀²/2τ²) abgeklungen; mindestens 4 τ.")
    s["amplitude"] = float(st.number_input("Amplitude J₀ (A/m²)", 1e-12, 1e18, float(s["amplitude"]), format="%.4g", key=f"tm_amp_{ver}",
                                           help="Die Felder skalieren linear. In 2D bezogen auf einen Meter Länge in z."))
    st.subheader("Quelle (Stromdichte)")
    c = st.columns(5 if three else 4)
    s["x_nm"] = _num(c[0], "x (nm)", f"tm_sx_{ver}", s["x_nm"])
    s["y_nm"] = _num(c[1], "y (nm)", f"tm_sy_{ver}", s["y_nm"])
    if three:
        s["z_nm"] = _num(c[2], "z (nm)", f"tm_sz_{ver}", s["z_nm"])
        s["axis"] = c[3].selectbox("Polarisation", list(ft.AXES_3D), index=list(ft.AXES_3D).index(s["axis"]), key=f"tm_ax_{ver}")
        s["size_nm"] = _num(c[4], "Gauß-Radius s (nm)", f"tm_ss_{ver}", s["size_nm"], 5.0, 0.1)
    else:
        s["angle_deg"] = _num(c[2], "Polarisation (° gegen x)", f"tm_an_{ver}", s["angle_deg"], 5.0, -360.0, 360.0, "90° = E_y (TEM-Mode eines Streifens mit PEC oben und unten).")
        s["size_nm"] = _num(c[3], "Gauß-Radius s (nm)", f"tm_ss_{ver}", s["size_nm"], 5.0, 0.1)
    st.subheader("Sonden")
    need = 3 if three else 2
    text = "\n".join(", ".join(f"{float(v):g}" for v in p[:need]) for p in m["probes"])
    new = st.text_area(f"Sonden ({'x, y, z' if three else 'x, y'} in nm, eine Zeile je Sonde)", value=text, key=f"tm_probes_{ver}_{m['dim']}", height=90)
    try:
        pts = [[float(v) for v in ln.replace(";", ",").split(",")] for ln in new.splitlines() if ln.strip()]
        if all(len(p) == need for p in pts):
            m["probes"] = pts
        else:
            st.error(f"Je Zeile {need} Zahlen.")
    except ValueError:
        st.error("Zahlen nicht lesbar.")
    st.subheader("Zeit")
    c = st.columns(4)
    tm["steps_per_period"] = int(c[0].number_input("Schritte je Periode (höchste Frequenz)", 4, 400, int(tm["steps_per_period"]), key=f"tm_spp_{ver}",
                                                   help="Bestimmt Δt, wenn Δt unten 0 ist. Die Trapezregel ist stabil; der Phasenfehler fällt mit dem Quadrat."))
    tm["dt_fs"] = _num(c[1], "Δt (fs, 0 = automatisch)", f"tm_dt_{ver}", tm["dt_fs"], 0.01, 0.0, 1e4)
    tm["t_end_fs"] = _num(c[2], "Endzeit (fs, 0 = automatisch)", f"tm_te_{ver}", tm["t_end_fs"], 5.0, 0.0, 1e6,
                          "Automatisch: Verzögerung + 3 τ + Laufzeit einmal über die Diagonale des Gebiets. Für Resonatoren länger wählen.")
    tm["f_max_THz"] = _num(c[3], "höchste Frequenz (THz, 0 = automatisch)", f"tm_fm_{ver}", tm["f_max_THz"], 10.0, 0.0, 1e5,
                           "Bestimmt Netzweite und Δt; automatisch aus Träger und Pulsbreite.")
    c = st.columns(4)
    tm["frames"] = int(c[0].number_input("Feldbilder (0 = keine)", 0, 400, int(tm["frames"]), key=f"tm_fr_{ver}"))
    if three:
        m["frame_z_nm"] = _num(c[1], "Schnittebene z (nm)", f"tm_fz_{ver}", m["frame_z_nm"])
        m["grid"] = int(c[2].number_input("Gitterpunkte je Richtung", 11, 201, int(m["grid"]), key=f"tm_grid_{ver}"))
    with st.expander("Newmark-Parameter"):
        c = st.columns(2)
        tm["gamma"] = _num(c[0], "γ", f"tm_gm_{ver}", tm["gamma"], 0.05, 0.5, 1.0, "½: ohne numerische Dämpfung.")
        tm["beta"] = _num(c[1], "β", f"tm_bt_{ver}", tm["beta"], 0.01, 0.01, 1.0, "¼ mit γ = ½: Trapezregel. Stabil für β ≥ (γ + ½)²/4.")
    st.subheader("Netz")
    c = st.columns(3)
    me["cells_per_wavelength"] = _num(c[0], "Zellen je kürzester Wellenlänge", f"tm_cpw_{ver}", me["cells_per_wavelength"], 0.5, 0.5, 40.0,
                                      "Bezogen auf λ_min = c / f_max, im Material geteilt durch n (nur 2D; in 3D: Zellen je Radius im Reiter Modell).")
    me["order"] = int(c[1].number_input("Ordnung p", 1, 8, int(me["order"]), key=f"tm_p_{ver}"))
    me["interface_factor"] = _num(c[2], "Faktor an Grenzflächen (2D)", f"tm_if_{ver}", me.get("interface_factor", 0.8), 0.1, 0.2, 2.0)
    est = ft.estimate(m)
    c = st.columns(4)
    c[0].metric("Träger f₀", f"{est['f0_THz']:.1f} THz")
    c[1].metric("höchste Frequenz", f"{est['f_max_THz']:.1f} THz", f"λ_min = {est['lam_min_nm']:.0f} nm", delta_color="off")
    c[2].metric("Δt", f"{est['dt_fs']:.4g} fs", f"{est['period_fs'] / est['dt_fs']:.0f} Schritte/Periode" if est["dt_fs"] > 0 else "", delta_color="off")
    c[3].metric("Schritte", f"{est['steps']}", f"bis {est['t_end_fs']:.1f} fs", delta_color="off")
    st.caption(f"Frequenzauflösung des Spektrums 1/T = {est['resolution_f_THz']:.1f} THz (Rechendauer T = {est['t_end_fs']:.1f} fs), Pulsbandbreite etwa {est['bandwidth_THz']:.0f} THz.")
    for lvl, t_ in ft.validate(m):
        (st.error if lvl == "error" else st.warning)(t_)
    try:
        ui.show_fig(ft.fig_source(m), "zeitbereich_quelle.png", f"tm_src_dl_{ver}")
    except Exception as exc:
        st.error(f"Vorschau nicht möglich: {exc}")


# ===================================================================================================================== run tab
def _tab_run(ctx, work):
    m = S.tm_model
    errs = [t for lvl, t in ft.validate(m) if lvl == "error"]
    for e in errs:
        st.error(e)
    est = ft.estimate(m)
    if m["dim"] == "2d":
        st.markdown(f"**{est['steps']} Schritte.** Ein Schritt ist ein Rücksubstitutions-Durchgang mit der einmal faktorisierten Matrix: bei einigen tausend Unbekannten "
                    "wenige Millisekunden, bei 10⁴ und mehr deutlich mehr (die Energie wird dann nur an einem Teil der Schritte berechnet).")
    else:
        st.markdown(f"**{est['steps']} Schritte, 3D.** Die Faktorisierung dauert bei etwa 3000 Unbekannten einige Sekunden und wächst schnell mit der Ordnung; über "
                    f"{ft.MAX_DOFS_3D} Freiheitsgraden bricht die Rechnung ab, bevor faktorisiert wird.")
    if st.button("Netz erzeugen und rechnen", type="primary", key="tm_run", disabled=bool(errs)):
        try:
            work.mkdir(parents=True, exist_ok=True)
            with st.spinner("Netz wird erzeugt …"):
                stats = ft.build_mesh(m, work)
            (work / "job.json").write_text(json.dumps(ft.job(m), ensure_ascii=False), encoding="utf-8")
            S.tm_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script="fem_time_worker.py")
            S.tm_dir, S.tm_cancel, S.tm_stats = str(work), False, stats
            S.tm_run_model = copy.deepcopy(m)
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("tm_msg"):
        (st.success if S.tm_msg[0] == "ok" else st.error)(S.tm_msg[1] + (" Die Ergebnisse stehen im Reiter 4." if S.tm_msg[0] == "ok" else ""))
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(work, tail=40))


def _running():
    proc, folder = S.tm_proc, Path(S.tm_dir)
    log = fr.read_log(folder)
    st.subheader("Rechnung läuft …")
    if proc.poll() is None:
        done, total = fr.progress(log)
        if total:
            st.progress(min(done / total, 1.0), text=f"Schrittblock {done} von {total}")
        else:
            st.progress(0.0, text="Netz und Faktorisierung …")
        st.code("\n".join(log.splitlines()[-8:]) or "(noch keine Ausgabe)")
        if S.get("tm_cancel"):
            st.info("Abbruch angefordert: die Rechnung hält nach dem laufenden Schrittblock an.")
        elif st.button("Abbrechen", key="tm_cancel_btn"):
            fr.request_cancel(folder)
            S.tm_cancel = True
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        S.tm_proc = None
        ok = proc.returncode == 0 and not flags["errors"]
        S.tm_msg = ("ok", "Abgebrochen." if flags["cancelled"] else "Fertig.") if ok else ("error", f"Beendet mit Code {proc.returncode}: " + "; ".join(flags["errors"][:3]))
        st.rerun()


# ===================================================================================================================== results
def _tab_results(work):
    res_file = work / "results.json"
    if not res_file.is_file():
        st.info("Noch keine Ergebnisse: im Reiter 3 rechnen.")
        return
    res = json.loads(res_file.read_text(encoding="utf-8"))
    meta = res["meta"]
    model = S.get("tm_run_model") or S.tm_model
    e = np.asarray(res["energy"])
    st.caption(f"{meta.get('name', '')} · {meta['dim'].upper()} · {meta.get('cells')} Zellen, p = {meta.get('order')}, {meta.get('dofs')} Freiheitsgrade · "
               f"{len(res['t_fs']) - 1} von {meta.get('steps')} Schritten, Δt = {meta['dt_fs']:.4g} fs · hpfem {meta.get('hpfem', '?')} · Start {meta.get('started', '')}")
    if res.get("cancelled"):
        st.warning("Die Rechnung wurde abgebrochen: nur ein Teil der Zeit liegt vor.")
    if res.get("error"):
        st.error(res["error"])
    sub = st.tabs(["Feldfilm", "Sonden E(t)", "Spektrum", "Energie", "Tabellen"])
    with sub[0]:
        _frames_tab(work, res, model)
    with sub[1]:
        if not res["probes"]:
            st.info("Keine Sonden im Modell.")
        else:
            comps = [c for c in ("ex", "ey", "ez") if c in res["probes"][0]]
            c = st.columns(2)
            which = c[0].multiselect("Sonden", range(len(res["probes"])), default=list(range(len(res["probes"]))), format_func=lambda i: f"Sonde {i + 1}", key="tm_pr_w")
            cs = c[1].multiselect("Komponenten", comps, default=comps[:2] if len(comps) > 2 else comps, format_func=lambda v: "E_" + v[1], key="tm_pr_c")
            ui.show_fig(ft.fig_probes(res, which, cs), "sonden.png", "tm_dl_probes")
            if len(res["probes"]) >= 2 and comps:
                c0 = cs[0] if cs else comps[0]
                tt = [ft.arrival_time(res["t_fs"], p[c0]) for p in res["probes"]]
                st.caption(f"Zeit des Maximums der Hüllkurve von E_{c0[1]} je Sonde (fs): " + ", ".join(f"Sonde {i + 1}: {t:.2f}" for i, t in enumerate(tt))
                           + ". Aus der Differenz und dem Abstand folgt die Gruppengeschwindigkeit (nur für getrennte Pulse aussagekräftig).")
    with sub[2]:
        _spectrum_tab(res)
    with sub[3]:
        ui.show_fig(ft.fig_energy(res, log=st.checkbox("logarithmisch", value=False, key="tm_en_log")), "energie.png", "tm_dl_energy")
        st.caption(f"Größte Energie {e.max():.3e} J, am Ende {e[-1]:.3e} J ({e[-1] / max(e.max(), 1e-300) * 100:.2f} % der größten). Ohne Absorber, Leitfähigkeit und "
                   "Quelle bleibt die Energie mit β = ¼, γ = ½ erhalten; während die Quelle läuft, nimmt sie zu.")
    with sub[4]:
        df = ft.probe_dataframe(res)
        ui.table_show(df.head(2000))
        c = st.columns(3)
        c[0].download_button("Sonden und Quelle (CSV)", ui.csv_bytes(df), file_name="zeitbereich_sonden.csv", mime="text/csv", key="tm_dl_csv")
        c[1].download_button("Energie (CSV)", ui.csv_bytes(ft.energy_dataframe(res)), file_name="zeitbereich_energie.csv", mime="text/csv", key="tm_dl_csv_e")
        c[2].download_button("Modell (JSON)", json.dumps(model, indent=1, ensure_ascii=False).encode("utf-8"), file_name="zeitbereich_modell.json",
                             mime="application/json", key="tm_dl_model")


def _spectrum_tab(res):
    if not res["probes"]:
        st.info("Keine Sonden im Modell.")
        return
    comps = [c for c in ("ex", "ey", "ez") if c in res["probes"][0]]
    c = st.columns(4)
    which = c[0].multiselect("Sonden", range(len(res["probes"])), default=[0], format_func=lambda i: f"Sonde {i + 1}", key="tm_sp_w")
    comp = c[1].selectbox("Komponente", comps, index=min(1, len(comps) - 1), format_func=lambda v: "E_" + v[1], key="tm_sp_c")
    win = c[2].selectbox("Fenster", list(ft.WINDOWS), format_func=ft.WINDOWS.get, key="tm_sp_win", help="Hann: unterdrückt das Abschneiden, wenn das Signal bei Rechenende nicht abgeklungen ist.")
    mode = c[3].radio("Größe", ["spectrum", "transfer"], format_func={"spectrum": "Spektrum |E(f)|", "transfer": "Übertragung |E|/|G′|"}.get, key="tm_sp_m", horizontal=True)
    axis = st.radio("Achse", ["f", "lam"], format_func={"f": "Frequenz (THz)", "lam": "Wellenlänge (nm)"}.get, horizontal=True, key="tm_sp_ax")
    if not which:
        return
    ui.show_fig(ft.fig_spectrum(res, which, comp, win, mode, axis), "spektrum.png", "tm_dl_spec")
    est = res["meta"]["estimate"]
    t = res["t_fs"]
    rows = []
    for i in which:
        f, a = (ft.transfer(res, i, comp, win) if mode == "transfer" else ft.spectrum(t, res["probes"][i][comp], win))
        ok = (f > 0.1 * est["f0_THz"]) & (f < 1.6 * est["f_max_THz"]) & np.isfinite(a)
        if ok.any():
            fp = float(f[ok][np.argmax(a[ok])])
            rows.append({"Sonde": i + 1, "Spitze f (THz)": round(fp, 2), "Spitze λ (nm)": round(ft.C0 / (fp * 1e12) / ft.NM, 1)})
    if rows:
        ui.table_show(rows)
    st.caption(f"Frequenzauflösung 1/T = {1e3 / max(t[-1], 1e-9):.2f} THz. Das Spektrum ist nicht gegen eine Rechnung im Frequenzbereich geprüft; es gilt nur im Band der Quelle "
               "(gestrichelt) und nur, wenn das Signal bis Rechenende abgeklungen ist.")


def _frames_tab(work, res, model):
    f = work / "field_frames.npz"
    if not f.is_file():
        st.info("Keine Feldbilder: im Reiter 2 „Feldbilder“ größer als 0 wählen.")
        return
    frames = dict(np.load(f))
    n = len(frames["times_fs"])
    comps = [c for c in ("ex", "ey", "ez") if c in frames]
    c = st.columns(3)
    quantity = c[0].selectbox("Größe", ["abs"] + comps, index=min(2, len(comps)), format_func=ft.QUANTITIES.get, key="tm_fq")
    scale_mode = c[1].radio("Farbskala", ["global", "frame"], format_func={"global": "fest (größter Wert aller Bilder)", "frame": "je Bild"}.get, key="tm_fs", horizontal=True)
    boost = c[2].number_input("Skala = Anteil des Maximums", 0.01, 1.0, 0.6, 0.05, key="tm_fb", help="Kleiner als 1 macht schwache Felder nach dem Puls sichtbar.")
    scale = None
    if scale_mode == "global":
        scale = max(float(np.nanmax(np.abs(ft.frame_values(frames, k, quantity)))) for k in range(n)) * float(boost) or None
    k = st.slider("Bild", 0, n - 1, min(n - 1, n // 3), key="tm_fk") if n > 1 else 0
    meta = res["meta"]
    model_ = model if model.get("dim") == "2d" else None
    ui.show_fig(ft.fig_frame(frames, k, quantity, meta, scale, model_), "feld.png", "tm_dl_frame")
    if st.button("Als Film abspielen", key="tm_play"):
        import matplotlib.pyplot as plt
        ph = st.empty()
        for j in range(n):
            fig = ft.fig_frame(frames, j, quantity, meta, scale, model_)
            ph.pyplot(fig)
            plt.close(fig)
            time.sleep(0.05)
    st.caption(f"{n} Bilder von t = {frames['times_fs'][0]:.1f} bis {frames['times_fs'][-1]:.1f} fs. 2D: Feld auf dem unterteilten Netz; 3D: Schnitt in der Ebene z = "
               f"{meta.get('frame_z_nm', 0):g} nm auf einem regelmäßigen Gitter. Farbskala fest: Felder nach dem Durchgang des Pulses sind klein.")
