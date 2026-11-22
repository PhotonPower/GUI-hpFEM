"""3D structures in the FEM model builder (Streamlit), hp-FEM 0.5. Called from fem_app.py: ``render_threed(ctx)``.

Scattering3D (sphere / cube, cross sections, Mie reference), Resonance3D (quasi-normal modes, Q), BandStructure3D (cubic photonic crystal), cut planes of the
fields and of the mesh, memory estimate before every run. Models, job and figures: fem_threed.py; worker: fem_threed_worker.py.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

import fem_bands as fb
import fem_run as fr
import fem_threed as ft
import fem_ui as ui

S = st.session_state

INFO = """
**Was gerechnet wird.** Die dreidimensionalen Löser der Bibliothek (Nédélec-Elemente, Vektorwellengleichung) für ein Objekt in Luft:

- **Streuung.** Ebene Welle (x-polarisiert, Ausbreitung längs +z, Amplitude 1 V/m) auf eine Kugel oder einen Würfel in einer Box mit PML (Streufeld-Formulierung).
  Berechnet werden Streu-, Absorptions- und Extinktionsquerschnitt aus den Poynting-Flüssen durch eine geschlossene Fläche um das Objekt und |E|² auf zwei Schnittebenen
  (x–z bei y = 0, x–y bei z = 0). Effizienz Q = σ / Querschnittsfläche (Kugel π r², Würfel (2 r)² mit halber Kante r). Für die Kugel zeigt die Oberfläche die
  **Mie-Reihe** als Referenz (`hpfem.mie_sphere`), für den Würfel gibt es keine Referenz.
- **Resonanzen.** Komplexe Eigenfrequenzen ω nahe der Zielwellenlänge (`Resonance3D`, PML): λ = 2πc/Re ω und Q = Re ω / (2 |Im ω|). Das Material wird bei der
  Zielwellenlänge eingefroren (keine selbstkonsistente Dispersion in diesem Modus); entartete Moden erscheinen als Paare oder Tripel gleicher λ.
- **Bandstruktur.** Floquet-Bloch-Bänder f = ω a / (2π c) eines einfach kubischen Gitters (Kugel oder Würfel mit ε in einem Hintergrund-ε, verlustfrei) entlang Γ–X–M–R–Γ
  (`BandStructure3D`, Gradientenkern ausprojiziert). Die statischen Nullmoden bei Γ werden entfernt.

**Netze.** Kugel: `box_with_ball` (gekrümmte Elemente, n Zellen je Radius; (Außenrand / r) · n muss ganzzahlig sein). Würfel: gleichmäßiges Tetraedernetz aus `box`,
das Objekt besteht aus den Zellen im Würfel (an das Gitter angepasst, also nur sinnvoll, wenn die Kante ein Vielfaches der Zelle ist). Kein Gmsh nötig.
Der Innenraum (halbe Breite) liegt um das Objekt, zwischen Innenraum und Außenrand liegt das PML.

**Viertelmodell.** Für die ebene x-polarisierte Welle längs z sind die Ebenen x = 0 (PEC) und y = 0 (natürliche Randbedingung) Spiegelebenen: es genügt der Quadrant x > 0, y > 0, die
Flüsse werden mit 4 multipliziert (so rechnet die Validierung C der Bibliothek). |E|² wird für die Darstellung gespiegelt. Das spart den Faktor 4 an Speicher und rund 8 an Zeit.

**Speicher und Zeit.** 3D-Rechnungen sind speichergebunden. Vor jedem Lauf schätzt die Bibliothek den Speicher (`hpfem.estimate_memory`); die App zeigt die Schätzung, warnt, wenn sie
einen großen Teil des freien Hauptspeichers beansprucht, und der Worker bricht ab, wenn sie die eingestellte Grenze überschreitet. Der installierte Wheel hat nur den direkten Löser
SparseLU (ein Thread): etwa 70 s bei 18 000 Unbekannten (Kugel, n = 2, p = 2, Viertelmodell), grob proportional zu N^1,7. Ein einzelner Löserlauf ist nicht unterbrechbar,
„Abbrechen“ wirkt zwischen Wellenlängen bzw. k-Punkten.

**Genauigkeit und Grenzen.**
- Grobe Netze liefern nur einige Prozent Genauigkeit: n = 2, p = 2 ergibt für die Kugel n = 2, ka = 2 etwa 3 % Abweichung in Q_sca; die Absorption (hier physikalisch 0) kann um
  mehrere Prozent von Q_sca abweichen. Die Bibliothek warnt bei diesem Netz, dass das PML unteraufgelöst ist (steht im Protokoll); bei höherem p und feinerem Netz verschwindet
  die Warnung, aber der Speicher wächst steil (n = 2, p = 3: 5,5 GB, über 2 Minuten).
- Die Abweichung zur Mie-Reihe ist das Maß für die Netzgüte der Streuung; für Würfel, Resonanzen und Bänder gibt es dieses Maß hier nicht. Die Resonanz- und
  Bandstruktur-Modi sind in der Oberfläche **nicht gegen analytische Lösungen validiert**.
- Material: Streuung mit beliebigem Material der App bei jeder Wellenlänge (Dispersion wird je Wellenlänge ausgewertet); Resonanzen und Bänder mit konstantem ε.
- Nicht in dieser Oberfläche: Substrate/Schichtstapel in 3D, Ports und Wellenleiter in 3D, adaptive hp-Verfeinerung, Gmsh-Netze, 3D-Wärme und -Mechanik.
"""


def _init():
    if "td_model" in S:
        return
    S.td_model = copy.deepcopy(ft.default_model())
    S.td_ver, S.td_proc, S.td_dir, S.td_msg, S.td_est = 0, None, None, None, None


def _bump():
    S.td_ver += 1


def _rename(m, old, new):
    m["materials"] = {(new if k == old else k): v for k, v in m["materials"].items()}
    if m["object"] == old:
        m["object"] = new


def render_threed(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    work = Path(ctx["work"]) / "threed"
    st.title("FEM-Modellwerkstatt: 3D-Strukturen")
    st.caption("Kugel oder Würfel in Luft: Streuung mit Mie-Referenz, Resonanzen, Bandstruktur eines kubischen Kristalls. hp-FEM: Scattering3D, Resonance3D, BandStructure3D.")
    if S.td_proc is not None:
        return _running()
    if not (ctx["FEATS"].get("threed") or ctx["lib"].get("error")):
        st.warning("Das eingestellte Python hat kein hpfem mit den 3D-Lösern. Anzeigen geht trotzdem.")
    t1, t2, t3, t4 = st.tabs(["1 Modell", "2 Rechnung", "3 Ergebnisse", "4 Info"])
    with t1:
        _tab_model()
    with t2:
        _tab_run(ctx, work)
    with t3:
        _tab_results(work)
    with t4:
        st.markdown(INFO)


def _num(col, label, key, value, step=0.1, lo=0.0, hi=1e6, help=None, fmt="%.5g"):
    return float(col.number_input(label, lo, hi, float(value), step, key=key, format=fmt, help=help))


# ===================================================================================================================== model
def _tab_model():
    m, ver = S.td_model, S.td_ver
    left, right = st.columns([6, 5])
    with left:
        presets = ft.presets()
        c = st.columns([3, 1])
        pick = c[0].selectbox("Vorlage", ["– auswählen –"] + list(presets), key="td_preset")
        if c[1].button("Laden", key="td_preset_load", disabled=pick.startswith("–")):
            S.td_model = copy.deepcopy(presets[pick])
            S.td_est = None
            _bump()
            st.rerun()
        m["name"] = st.text_input("Name", value=m["name"], key=f"td_name_{ver}")
        m["task"] = st.radio("Aufgabe", list(ft.TASKS), index=list(ft.TASKS).index(m["task"]), format_func=ft.TASKS.get, key=f"td_task_{ver}")
        m["shape"] = st.radio("Form", list(ft.SHAPES), index=list(ft.SHAPES).index(m["shape"]), format_func=ft.SHAPES.get, key=f"td_shape_{ver}", horizontal=True)
        if m["task"] == "bands":
            b = m["bands"]
            c = st.columns(3)
            b["radius"] = _num(c[0], "Radius bzw. halbe Kante (a)", f"td_br_{ver}", b["radius"], 0.05, 0.02, 0.49, "Einheiten der Gitterkonstante a; die Zelle ist [−0,5, 0,5]³.")
            b["eps"] = _num(c[1], "ε des Objekts", f"td_be_{ver}", b["eps"], 0.5, 0.01, 400.0)
            b["eps_bg"] = _num(c[2], "ε des Hintergrunds", f"td_bb_{ver}", b["eps_bg"], 0.5, 0.01, 400.0)
        else:
            m["radius_nm"] = _num(st, "Radius bzw. halbe Kante (nm)", f"td_r_{ver}", m["radius_nm"], 5.0, 1.0, 1e6)
            st.subheader("Material des Objekts")
            lam0 = ft.lam_check(m)[0] if ft.lam_check(m) else 500.0
            ui.material_editor(m, ver, lam0, {m["object"]}, _rename, _bump, prefix="td_")
            names = list(m["materials"])
            m["object"] = st.selectbox("Objekt besteht aus", names, index=names.index(m["object"]) if m["object"] in names else 0, key=f"td_obj_{ver}",
                                       help="Umgebung ist Luft (ε = 1). Die Permittivität wird bei jeder Wellenlänge des Durchlaufs ausgewertet (Resonanzen: bei der Zielwellenlänge).")
    with right:
        for lvl, text in ft.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        st.markdown("**Aufbau.** " + ("Einfach kubisches Gitter, Gitterkonstante a = 1, Zelle [−½, ½]³, ein Objekt in der Mitte; Pfad Γ–X–M–R–Γ."
                                      if m["task"] == "bands" else
                                      "Das Objekt sitzt im Ursprung einer Box mit PML-Rand. Ebene Welle x-polarisiert längs +z. Das Netz sehen Sie nach „Netz prüfen“ im Reiter 2."))


# ===================================================================================================================== run
def _estimate_box(est, m, ctx_threads):
    if est is None:
        return
    if "error" in est:
        st.error(f"Netz oder Schätzung nicht möglich: {est['error']}")
        return
    avail = ft.available_gb()
    st.info(f"**{est['cells']} Zellen, {est['dofs']} Unbekannte, Speicher ≈ {est['human']}** · Zeit je Löserlauf grob {ft.time_hint(est['dofs']):.0f} s "
            f"(SparseLU, ein Thread) · {est['text']}")
    gb = est["bytes"] / ft.GB
    if gb > float(m["max_gb"]):
        st.error(f"Die Schätzung ({gb:.1f} GB) liegt über der eingestellten Grenze von {m['max_gb']:g} GB: der Worker würde abbrechen. Netz oder Ordnung verkleinern oder die Grenze anheben.")
    elif avail is not None and gb > 0.6 * avail:
        st.warning(f"Die Schätzung ({gb:.1f} GB) beansprucht mehr als 60 % des freien Hauptspeichers ({avail:.0f} GB); andere Prozesse könnten gestört werden.")


def _tab_run(ctx, work):
    m, ver = S.td_model, S.td_ver
    t = m["task"]
    if t == "scatter":
        st.subheader("Wellenlängen")
        s = m["sweep"]
        c = st.columns(3)
        s["start"] = _num(c[0], "von (nm)", f"td_s0_{ver}", s["start"], 10.0, 50.0, 20000.0)
        s["stop"] = _num(c[1], "bis (nm)", f"td_s1_{ver}", s["stop"], 10.0, 50.0, 20000.0)
        s["count"] = int(c[2].number_input("Zahl der Wellenlängen", 1, 100, int(s["count"]), key=f"td_sn_{ver}",
                                           help="Jede Wellenlänge ist eine eigene Faktorisierung (Minuten bei feinen Netzen)."))
    elif t == "resonance":
        st.subheader("Resonanzsuche")
        r = m["resonance"]
        c = st.columns(3)
        r["lam0_nm"] = _num(c[0], "Zielwellenlänge (nm)", f"td_rl_{ver}", r["lam0_nm"], 10.0, 50.0, 20000.0, "Das Material wird hier eingefroren; das PML ist für diese Wellenlänge ausgelegt.")
        r["num_modes"] = int(c[1].number_input("Zahl der Moden", 1, 30, int(r["num_modes"]), key=f"td_rn_{ver}"))
        r["krylov"] = int(c[2].number_input("Krylov-Dimension (0 = automatisch)", 0, 600, int(r["krylov"]), 10, key=f"td_rk_{ver}"))
    else:
        st.subheader("Pfad und Bänder")
        b = m["bands"]
        c = st.columns(4)
        b["n_seg"] = int(c[0].number_input("Punkte je Segment", 1, 20, int(b["n_seg"]), key=f"td_bs_{ver}", help="k-Punkte je Pfadsegment ohne den Startpunkt; je Punkt ein Eigenwertproblem."))
        b["bands"] = int(c[1].number_input("Zahl der Bänder", 1, 20, int(b["bands"]), key=f"td_bn_{ver}"))
        b["cells"] = int(c[2].number_input("Zellen je Kante (Würfel)", 2, 40, int(b["cells"]), 2, key=f"td_bc_{ver}", help="Nur für die Form Würfel."))
        b["krylov"] = int(c[3].number_input("Krylov-Dimension (0 = auto)", 0, 600, int(b["krylov"]), 10, key=f"td_bk_{ver}"))
    st.subheader("Netz und Ordnung")
    c = st.columns(5)
    if t == "bands":
        m["n"] = int(c[0].number_input("Zellen je Radius (n, Kugel)", 1, 12, int(m["n"]), key=f"td_n_{ver}"))
    else:
        m["n"] = int(c[0].number_input("Zellen je Radius (n)", 1, 12, int(m["n"]), key=f"td_n_{ver}", help="Bei der Kugel: Zellen auf dem Radius; beim Würfel: Zellen auf der halben Kante."))
        m["half_width"] = _num(c[1], "Innenraum (halbe Breite, in r)", f"td_hw_{ver}", m["half_width"], 0.5, 1.0, 20.0, "Die Box um das Objekt bis zum PML.")
        m["outer"] = _num(c[2], "Außenrand (in r)", f"td_ou_{ver}", m["outer"], 0.5, 1.5, 30.0, "Ende des PML; zwischen Innenraum und Außenrand liegt die Schicht.")
    m["order"] = int(c[3].number_input("Ordnung p", 1, 6, int(m["order"]), key=f"td_p_{ver}"))
    if t == "scatter":
        m["quarter"] = c[4].checkbox("Viertelmodell", value=bool(m["quarter"]), key=f"td_q_{ver}", help=ft.SYM[True] + ": nur für die x-polarisierte Welle längs z; Faktor 4 weniger Speicher.")
    c = st.columns(2)
    m["fields"] = c[0].checkbox("Feldschnitte speichern", value=bool(m["fields"]), key=f"td_f_{ver}", disabled=t == "bands", help="|E|² auf zwei Ebenen je Wellenlänge bzw. Mode.")
    m["max_gb"] = _num(c[1], "Speichergrenze (GB)", f"td_mg_{ver}", m["max_gb"], 2.0, 0.5, 1000.0, "Der Worker bricht ab, wenn die Schätzung der Bibliothek darüber liegt.")
    errs = [x for lvl, x in ft.validate(m) if lvl == "error"]
    for e in errs:
        st.error(e)
    folder = work / "estimate"
    c = st.columns([1, 2])
    if c[0].button("Netz prüfen und Speicher schätzen", key="td_est_btn", disabled=bool(errs)):
        with st.spinner("Netz wird erzeugt …"):
            S.td_est = ft.run_estimate(ctx["py"], ctx["repo"], folder, ft.job(m), int(ctx["threads"]))
            S.td_est_model = copy.deepcopy(m)
    est = S.get("td_est")
    _estimate_box(est, m, ctx["threads"])
    if est and "error" not in est and (folder / f"mesh_{t}.npz").is_file():
        with st.expander("Netz (Schnitt durch das Modell)"):
            axis = {"x": 0, "y": 1, "z": 2}[st.radio("Schnittebene senkrecht zu", ["x", "y", "z"], index=1, horizontal=True, key="td_mesh_axis")]
            npz = np.load(folder / f"mesh_{t}.npz")
            unit, lab = (1.0, "a") if t == "bands" else (1e-9, "nm")
            lo, hi = float(npz["vertices"][:, axis].min() / unit), float(npz["vertices"][:, axis].max() / unit)
            coord = st.slider("Lage der Ebene", lo, hi, min(max(0.0, lo), hi) if lo <= 0 <= hi else (lo + hi) / 2, (hi - lo) / 100 or 0.01, key=f"td_mesh_c_{axis}")
            ui.show_fig(ft.fig_mesh(npz, axis, coord * unit, unit, lab), "netz.png", "td_dl_mesh")
            st.caption("Gerade Tetraederkanten; bei der Kugel liegt der Rand im Löser exakt (gekrümmte Elemente). Orange: Zellen des Objekts.")
    st.markdown("---")
    stale = est is not None and S.get("td_est_model") is not None and ft.job(S.td_est_model) != ft.job(m)
    if stale:
        st.caption("Die Schätzung gehört zu einer früheren Einstellung: bitte neu schätzen.")
    if st.button("Rechnen", type="primary", key="td_run", disabled=bool(errs)):
        try:
            work.mkdir(parents=True, exist_ok=True)
            (work / "job.json").write_text(json.dumps(ft.job(m), ensure_ascii=False), encoding="utf-8")
            for f in work.glob("field_*.npz"):
                f.unlink()
            S.td_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script="fem_threed_worker.py")
            S.td_dir, S.td_cancel, S.td_run_model = str(work), False, copy.deepcopy(m)
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("td_msg"):
        (st.success if S.td_msg[0] == "ok" else st.error)(S.td_msg[1] + (" Die Ergebnisse stehen im Reiter 3." if S.td_msg[0] == "ok" else ""))
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(work, tail=40))


def _running():
    proc, folder = S.td_proc, Path(S.td_dir)
    log = fr.read_log(folder)
    st.subheader("Rechnung läuft …")
    if proc.poll() is None:
        frac, label = fr.progress_info(log)
        st.progress(frac, text=label)
        st.code("\n".join(log.splitlines()[-10:]) or "(noch keine Ausgabe)")
        st.caption("Ein einzelner Löserlauf ist nicht unterbrechbar; der Abbruch wirkt nach dem laufenden Schritt.")
        if S.get("td_cancel"):
            st.info("Abbruch angefordert: die Rechnung hält nach dem laufenden Schritt an.")
        elif st.button("Abbrechen", key="td_cancel_btn"):
            fr.request_cancel(folder)
            S.td_cancel = True
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        S.td_proc = None
        ok = proc.returncode == 0 and not flags["errors"]
        S.td_msg = ("ok", "Abgebrochen." if flags["cancelled"] else "Fertig.") if ok else ("error", f"Beendet mit Code {proc.returncode}: " + "; ".join(flags["errors"][:3]))
        st.rerun()


# ===================================================================================================================== results
def _tab_results(work):
    res_file = work / "results.json"
    if not res_file.is_file():
        st.info("Noch keine Ergebnisse: im Reiter 2 rechnen.")
        return
    res = json.loads(res_file.read_text(encoding="utf-8"))
    meta = res["meta"]
    model = S.get("td_run_model") or S.td_model
    st.caption(f"{meta.get('name', '')} · {meta.get('task')} · {meta.get('cells')} Zellen, p = {meta.get('order')}, {meta.get('dofs')} Unbekannte, Speicher ≈ {meta.get('human')} · "
               f"hpfem {meta.get('hpfem', '?')} · Start {meta.get('started', '')}")
    if res.get("cancelled"):
        st.warning("Die Rechnung wurde abgebrochen: nur ein Teil liegt vor.")
    task = meta.get("task")
    if task == "scatter":
        _res_scatter(res, work, model, meta)
    elif task == "resonance":
        _res_resonance(res, work, model, meta)
    else:
        _res_bands(res, model)


def _field_view(npz_path, quarter, model, key_prefix, title):
    npz = np.load(npz_path)
    c = st.columns(2)
    sel = c[0].radio("Ebene", ["xz", "xy"], format_func={"xz": "x–z (y = 0, Welle längs z)", "xy": "x–y (z = 0)"}.get, horizontal=True, key=f"{key_prefix}_pl")
    radius = float(model["radius_nm"]) * 1e-9
    ui.show_fig(ft.fig_field(npz, sel, quarter, 1e-9, "nm", radius, model["shape"], title), f"feld_{sel}.png", f"{key_prefix}_dl")


def _res_scatter(res, work, model, meta):
    pts = res["points"]
    for p in pts:
        if "error" in p:
            st.error(f"λ = {p['lam_nm']:g} nm: {p['error']}")
    ok = [p for p in pts if "q_sca" in p]
    if not ok:
        return
    sub = st.tabs(["Wirkungsquerschnitte", "Felder", "Tabelle"])
    with sub[0]:
        if len(ok) > 1:
            ui.show_fig(ft.fig_efficiency(ok), "effizienz.png", "td_dl_eff")
        df = ft.scatter_dataframe(ok)
        ui.table_show(df.round(5))
        if "mie_q_sca" in ok[0]:
            worst = max(abs(p["q_sca"] - p["mie_q_sca"]) / max(abs(p["mie_q_sca"]), 1e-30) for p in ok) * 100
            st.caption(f"Größte Abweichung Q_sca zur Mie-Reihe: {worst:.1f} %. Das ist das Maß für die Netzgüte (Q_abs bei verlustfreier Kugel: Fehler des Flussintegrals).")
        else:
            st.caption("Würfel: keine analytische Referenz. Konvergenz prüfen, indem n und p erhöht werden.")
    with sub[1]:
        withf = [p for p in ok if p.get("field_file")]
        if not withf:
            st.info("Keine Feldschnitte gespeichert.")
        else:
            i = st.selectbox("Wellenlänge", range(len(withf)), format_func=lambda i: f"{withf[i]['lam_nm']:g} nm", key="td_sc_lam")
            _field_view(work / withf[i]["field_file"], bool(meta.get("quarter")), model, "td_sc", f"|E|²/|E₀|² bei {withf[i]['lam_nm']:g} nm")
            st.caption("Gestrichelt: Umriss des Objekts. Im Viertelmodell wird das berechnete Viertel an den Symmetrieebenen gespiegelt; die Welle läuft in +z.")
    with sub[2]:
        df = ft.scatter_dataframe(ok)
        c = st.columns(2)
        c[0].download_button("Tabelle (CSV)", ui.csv_bytes(df), file_name="streuung_3d.csv", mime="text/csv", key="td_dl_csv")
        c[1].download_button("Modell (JSON)", json.dumps(model, indent=1, ensure_ascii=False).encode("utf-8"), file_name="modell_3d.json", mime="application/json", key="td_dl_model")


def _res_resonance(res, work, model, meta):
    modes = res["modes"]
    if not modes:
        st.info("Keine Moden gefunden.")
        return
    df = ft.modes_dataframe(modes)
    ui.table_show(df.round(6))
    st.caption(f"Rechenzeit {res.get('seconds', 0):.0f} s. Moden nach Abstand zur Zielwellenlänge geordnet; gleiche λ: entartete Moden. Keine analytische Referenz in der Oberfläche; "
               "ein kleines Residuum zeigt nur, dass der Eigenwertlöser konvergiert ist, nicht dass das Netz fein genug ist: zur Probe n oder p erhöhen.")
    st.download_button("Moden (CSV)", ui.csv_bytes(df), file_name="resonanzen_3d.csv", mime="text/csv", key="td_dl_modes")
    withf = [r for r in modes if r.get("field_file")]
    if withf:
        j = st.selectbox("Mode", range(len(withf)), format_func=lambda j: f"Nr. {withf[j]['index'] + 1}: λ = {withf[j]['lam_nm']:.2f} nm, Q = {withf[j]['Q']:.1f}", key="td_rs_mode")
        _field_view(work / withf[j]["field_file"], False, model, "td_rs", f"|E|² (Skala beliebig), λ = {withf[j]['lam_nm']:.2f} nm")


def _res_bands(res, model):
    pts = [p for p in res["points"] if "f" in p]
    for p in res["points"]:
        if "error" in p:
            st.error(f"k-Punkt {p['index']}: {p['error']}")
    if not pts:
        return
    x, ticks = res["x"], res["ticks"]
    ref = st.checkbox("Leeres Gitter |k + G| als graue Referenz", value=True, key="td_b_ref")
    nb = max(len(p["f"]) for p in pts)
    empty = ft.empty_lattice_bands([p["k"] for p in pts], nb) if ref else None
    ui.show_fig(fb.fig_bands(pts, x, ticks, gaps=True, empty_lattice=empty), "bandstruktur_3d.png", "td_dl_bands")
    gaps = fb.gaps_dataframe(pts)
    if len(gaps):
        st.success("Lücken über den ganzen Pfad: " + "; ".join(f"Band {r['zwischen Band']}: f = {r['untere Kante f']:.4f} bis {r['obere Kante f']:.4f} (Δf/Mitte = {r['Δf/Mitte (%)']:.1f} %)"
                                                          for _, r in gaps.iterrows()))
        st.caption("Eine vollständige 3D-Lücke setzt voraus, dass die Extrema auf dem Pfad liegen; zusätzlich ist der Pfad Γ–X–M–R–Γ nur ein Ausschnitt der Brillouin-Zone.")
    else:
        st.info("Keine Lücke zwischen benachbarten Bändern über den ganzen Pfad.")
    df = ft.bands_dataframe(res["points"], x)
    ui.table_show(df.round(5))
    st.download_button("Bänder (CSV)", ui.csv_bytes(df), file_name="bandstruktur_3d.csv", mime="text/csv", key="td_dl_bcsv")
    st.caption("Nicht gegen Literatur validiert; grobe Netze verschieben die Bänder (Probe: leeres Gitter und höhere Ordnung). Das niedrigste Band beginnt bei f = 0.")
