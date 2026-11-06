"""Mode „Studie: Optimierung, Rekonstruktion, UQ“ of the FEM model builder (Streamlit). Called from fem_app.py: render() for the page.

A study is a job document of schema version 2 for the job runner of hpfem (python -m hpfem.run; tasks "optimize", "reconstruct", "uq",
hp-FEM M16, ADR-0012): it is edited here as JSON (from a template or a file), started in its own folder under <work>/study/<name>, and its
folder (or any other study folder) is shown: history, parameters, Pareto front, the result of the task (optimum, fit with uncertainties and
posterior, spread and Sobol' indices). Reading and figures in fem_study.py.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import streamlit as st

import fem_run as fr
import fem_study as fs
import fem_ui as ui

S = st.session_state


def init():
    if "st_job_text" not in S:
        S.st_job_text = json.dumps(fs.template("optimize"), indent=1, ensure_ascii=False)
    for key, value in (("st_job_ver", 0), ("st_proc", None), ("st_dir", None)):
        if key not in S:
            S[key] = value


def folder_name(job):
    name = str((job or {}).get("name") or "studie")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "studie"


def render(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    init()
    work = Path(ctx["work"]) / "study"
    st.title("FEM-Modellwerkstatt: Studien")
    st.caption("Optimierung, Rekonstruktion von Parametern aus Messwerten (mit Unsicherheiten) und Unsicherheits- und Sensitivitätsanalyse auf einer "
               "Gitterzelle, gerechnet vom Job-Runner der Bibliothek (python -m hpfem.run, Job-Schema 2). Jede Auswertung wird im Studienspeicher "
               "(*.study.jsonl) abgelegt: eine abgebrochene Studie setzt beim nächsten Start dort fort.")
    if S.st_proc is not None:
        running_view(ctx)
        return
    t1, t2, t3, t4, t5, t6 = st.tabs(["1 Studie", "2 Verlauf", "3 Pareto-Front", "4 Ergebnis", "5 Tabelle und Export", "6 Info"])
    with t1:
        tab_job(work, ctx)
    study = current_study(work)
    with t2:
        tab_history(study)
    with t3:
        tab_pareto(study)
    with t4:
        tab_result(study)
    with t5:
        tab_table(study)
    with t6:
        tab_info()


# ------------------------------------------------------------------------------------------------------------------ 1 job
def tab_job(work, ctx):
    left, right = st.columns([3, 2])
    with right:
        st.subheader("Vorlage oder Datei")
        task = st.selectbox("Vorlage", list(fs.TASKS), format_func=fs.TASK_LABELS.get, key="st_tpl_task",
                            help="Eine Lamellenzelle (Ordnung 2, zwei Konfigurationen) mit Breite und ε des Stegs als Parametern: rechnet in "
                                 "etwa einer Sekunde je Auswertung.")
        if st.button("Vorlage übernehmen", key="st_tpl_load"):
            S.st_job_text = json.dumps(fs.template(task), indent=1, ensure_ascii=False)
            S.st_job_ver += 1
            st.rerun()
        up = st.file_uploader("Job-Datei laden (JSON)", type=["json"], key="st_upload")
        if up is not None and st.button("Datei übernehmen", key="st_upload_load"):
            S.st_job_text = up.getvalue().decode("utf-8", errors="replace")
            S.st_job_ver += 1
            st.rerun()
        st.divider()
        st.subheader("Vorhandene Studie öffnen")
        folders = sorted([p for p in work.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True) if work.is_dir() else []
        if folders:
            pick = st.selectbox("Studien im Arbeitsordner", [str(p) for p in folders], format_func=lambda p: Path(p).name, key="st_pick")
            if st.button("Anzeigen", key="st_show"):
                S.st_dir = pick
                st.rerun()
        other = st.text_input("oder Ordner (mit results.json oder *.study.jsonl)", key="st_other")
        if other and st.button("Ordner anzeigen", key="st_show_other"):
            if Path(other).is_dir():
                S.st_dir = other
                st.rerun()
            else:
                st.error("Ordner nicht gefunden.")
        if S.st_dir:
            st.caption(f"Angezeigt: {S.st_dir}")
    with left:
        st.subheader("Job-Dokument")
        text = st.text_area("JSON (Schema 2)", value=S.st_job_text, height=520, key=f"st_text_{S.st_job_ver}",
                            help="Aufbau wie in der Dokumentation von hpfem.run: model, mesh, materials, stack, solver, parameters (Name, Grenzen, "
                                 "Form und Feld, Trapez oder Material), configurations (Wellenlänge, Winkel, Polarisation, Ordnungen) und der Block "
                                 "der Aufgabe. Werte in Job-Einheiten (Längen in unit, Winkel in Grad).")
        S.st_job_text = text
        try:
            job = json.loads(text)
            errs = fs.check_job(job)
        except ValueError as exc:
            job, errs = None, [f"Kein gültiges JSON: {exc}"]
        for e in errs:
            st.error(e)
        if job and not errs:
            names = [p["name"] for p in job["parameters"]]
            st.caption(f"{fs.TASK_LABELS[job['task']]} · Parameter: {', '.join(names)} · {len(job['configurations'])} Konfiguration(en)")
            if job["task"] == "reconstruct" and "posterior" in job.get("reconstruct", {}) and not ctx["FEATS"].get("opt_mcmc", True):
                st.warning("Der Posterior braucht emcee im Python der Rechnung (pip install emcee bzw. hpfem[opt-mcmc]).")
        st.download_button("Job-Datei speichern", text.encode("utf-8"), file_name=f"{folder_name(job)}.json", mime="application/json",
                           key="st_job_dl", disabled=job is None)
        folder = work / folder_name(job) if job else None
        resume = bool(folder and list(folder.glob("*.study.jsonl")))
        if folder:
            st.caption(f"Ordner der Studie: {folder}" + (" — dort liegt schon ein Studienspeicher: die Studie setzt fort (bereits gerechnete "
                                                         "Punkte kommen aus dem Speicher)." if resume else ""))
        lib_ok = not ctx["lib"].get("error") and ctx["FEATS"].get("study", False)
        if not lib_ok:
            st.warning("Das eingestellte Python hat kein hpfem mit Studien (hpfem.run, Schema 2; hp-FEM M16 S9). Anzeigen geht trotzdem.")
        c = st.columns(2)
        if c[0].button("Studie starten" if not resume else "Studie fortsetzen", type="primary", key="st_start", disabled=bool(errs) or not lib_ok):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "job.json").write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
            try:
                S.st_proc = fr.start_study(ctx["py"], ctx["repo"], folder, int(ctx["threads"]))
                S.st_dir, S.st_cancel = str(folder), False
                st.rerun()
            except Exception as exc:
                st.error(f"Start fehlgeschlagen: {exc}")
        if resume and c[1].button("Speicher löschen (neu beginnen)", key="st_reset",
                                  help="Löscht den Studienspeicher und die Ergebnisse dieses Ordners; die nächste Rechnung beginnt von vorn."):
            for f in list(folder.glob("*.study.jsonl")) + [folder / "results.json"]:
                if f.exists():
                    f.unlink()
            st.rerun()


def running_view(ctx):
    proc, folder = S.st_proc, Path(S.st_dir)
    study = fs.load_study(folder)
    st.subheader(f"Studie läuft: {folder.name}")
    if proc.poll() is None:
        frac, label = fs.progress(study)
        if frac is None:
            st.info(label)
        else:
            st.progress(frac, text=label)
        df = fs.history(study)
        inf = fs.info(study)
        if not df.empty and inf["observables"]:
            obs = inf["observables"][inf["objective"]] if inf["task"] == "optimize" else inf["observables"][0]
            ui.show_fig(fs.fig_history(df, obs, inf["maximize"]), "verlauf.png", "st_live_png")
        if S.get("st_cancel"):
            st.info("Abbruch angefordert: die Studie hält nach der laufenden Auswertung an.")
            if st.button("Sofort beenden", key="st_kill", help="Beendet den Prozess hart; die laufende Auswertung geht verloren, der Speicher bleibt."):
                proc.terminate()
                S.st_proc = None
                st.rerun()
        elif st.button("Abbrechen", key="st_cancel_btn", help="Hält nach der laufenden Auswertung an; der Studienspeicher bleibt und ein neuer Start setzt fort."):
            fr.request_cancel(folder)
            S.st_cancel = True
            st.rerun()
        with st.expander("Protokoll"):
            st.code("\n".join(study["log"].splitlines()[-30:]) or "(noch keine Ausgabe)")
        time.sleep(2.0)
        st.rerun()
    else:
        S.st_proc = None
        code = proc.returncode
        errs = [e.get("text") for e in study["events"] if e.get("event") == "error"]
        if code in (0, 2) and not errs:  # exit code 2 of the runner: cancelled
            S.st_msg = ("ok", "Studie abgebrochen; ein neuer Start setzt fort." if code == 2 else "Studie fertig.")
        else:
            tail = "\n".join(study["log"].splitlines()[-12:])
            S.st_msg = ("error", f"Die Studie endete mit Code {code}. " + (" | ".join(t for t in errs if t) or "") + (f"\n\n{tail}" if tail else ""))
        st.rerun()


def current_study(work):
    msg = S.pop("st_msg", None)
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])
    if not S.st_dir or not Path(S.st_dir).is_dir():
        return None
    return fs.load_study(S.st_dir)


def _no_study():
    st.info("Keine Studie geöffnet: im Reiter 1 starten oder einen Studienordner anzeigen.")


# ------------------------------------------------------------------------------------------------------------------ 2 history
def tab_history(study):
    if study is None:
        return _no_study()
    df, inf = fs.history(study), fs.info(study)
    st.caption(f"{fs.TASK_LABELS.get(inf['task'], inf['task'] or '?')} · {inf['name'] or ''} · {len(df)} Auswertungen" +
               (f", davon {int((df['Status'] != 'ok').sum())} nicht ok" if not df.empty and (df["Status"] != "ok").any() else ""))
    if df.empty:
        st.info("Noch keine Auswertungen.")
        return
    failed = df[df["Status"] != "ok"]
    if not failed.empty:
        st.warning(f"{len(failed)} Auswertung(en) fehlgeschlagen, z. B. Nr. {failed['Nr'].iloc[0]}: {failed['Fehler'].iloc[0] or failed['Status'].iloc[0]}. "
                   "„MeshQualityError“ heißt: das verformte Netz ist an diesen Werten ungültig; Grenzen enger setzen oder in „morph“ "
                   "\"remesh\": true (nur strukturierte Netze).")
    obs = inf["observables"]
    c = st.columns([3, 1])
    o = c[0].selectbox("Messgröße", obs, index=inf["objective"] if inf["objective"] < len(obs) else 0, key="st_hist_obs")
    mx = c[1].checkbox("Maximum suchen", value=inf["maximize"] if obs.index(o) == inf["objective"] else False, key=f"st_hist_max_{o}",
                       help="Die orange Treppe zeigt das bisher beste: Maximum oder Minimum.")
    opt = (study.get("results") or {}).get("optimize")
    optimum = opt["values"][obs.index(o)] if opt and opt.get("values") else None
    ui.show_fig(fs.fig_history(df, o, mx, optimum), "verlauf.png", "st_hist_png")
    st.markdown("**Parameter je Auswertung** (gepunktet: Grenzen, gestrichelt: Ergebnis)")
    best = opt["params"] if opt else ((study.get("results") or {}).get("reconstruct") or {}).get("params")
    ui.show_fig(fs.fig_parameters(df, inf["names"], inf["bounds"], best), "parameter.png", "st_par_png")
    if len(inf["names"]) >= 1:
        st.markdown("**Abgetastete Punkte**")
        cols = inf["names"] + obs
        c = st.columns(3)
        x = c[0].selectbox("x", cols, index=0, key="st_sc_x")
        y = c[1].selectbox("y", cols, index=1 if len(cols) > 1 else 0, key="st_sc_y")
        col = c[2].selectbox("Farbe", obs, index=obs.index(o), key="st_sc_c")
        ui.show_fig(fs.fig_scatter(df, x, y, col), "punkte.png", "st_sc_png")


# ------------------------------------------------------------------------------------------------------------------ 3 pareto
def tab_pareto(study):
    if study is None:
        return _no_study()
    df, inf = fs.history(study), fs.info(study)
    obs = inf["observables"]
    if df.empty or len(obs) < 2:
        st.info("Die Pareto-Front braucht Auswertungen mit mindestens zwei Messgrößen.")
        return
    st.caption("Nicht dominierte Auswertungen zweier Messgrößen: keine andere Auswertung ist in beiden mindestens so gut und in einer besser. "
               "Die Front zeigt den Kompromiss zwischen zwei Zielen aus allen bisherigen Auswertungen (die Optimierung selbst verfolgt ein Ziel).")
    c = st.columns(4)
    x = c[0].selectbox("Größe 1", obs, index=0, key="st_p_x")
    mx = c[1].radio("Größe 1", ["max", "min"], horizontal=True, key="st_p_mx", label_visibility="hidden") == "max"
    y = c[2].selectbox("Größe 2", obs, index=1, key="st_p_y")
    my = c[3].radio("Größe 2", ["max", "min"], horizontal=True, index=1, key="st_p_my", label_visibility="hidden") == "max"
    if x == y:
        st.warning("Zwei verschiedene Größen wählen.")
        return
    ui.show_fig(fs.fig_pareto(df, x, y, mx, my), "pareto.png", "st_p_png")
    front = fs.pareto_front(df[df["Status"] == "ok"], x, y, mx, my)
    ui.table_show(front[["Nr", *inf["names"], *obs]])
    st.download_button("Front als CSV", ui.csv_bytes(front), file_name="pareto.csv", mime="text/csv", key="st_p_csv")


# ------------------------------------------------------------------------------------------------------------------ 4 result
def tab_result(study):
    if study is None:
        return _no_study()
    res, inf = study.get("results"), fs.info(study)
    if not res:
        st.info("Noch kein Ergebnis (results.json entsteht am Ende der Studie; der Verlauf steht schon im Reiter 2).")
        return
    if res.get("cancelled"):
        st.warning("Die Studie wurde abgebrochen: das Ergebnis gilt für die bis dahin gerechneten Punkte.")
    task = res.get("task")
    if task == "optimize":
        o = res["optimize"]
        (st.success if o.get("success") else st.warning)(
            f"{o['method']}: {'Maximum' if inf['maximize'] else 'Minimum'} von „{inf['observables'][inf['objective']]}“ = {o['value']:.6g} "
            f"nach {o['evaluations']} Auswertungen ({o['new_evaluations']} neu, {o['cache_hits']} aus dem Speicher). {o.get('message', '')}")
        ui.table_show(fs.optimize_dataframe(study))
    elif task == "reconstruct":
        rec = res["reconstruct"]
        (st.success if rec.get("success") else st.warning)(
            f"Anpassung: χ²_red = {rec['chi2_red']:.3g}, {rec['iterations']} Iterationen, {rec['evaluations']} Auswertungen. {rec.get('message', '')}")
        for w in rec.get("warnings", []):
            st.warning(w)
        if rec.get("at_bounds"):
            st.warning("An den Grenzen: " + ", ".join(rec["at_bounds"]) + " (die Unsicherheit dieser Parameter ist nicht aussagekräftig).")
        st.markdown("**Parameter** (σ aus der Laplace-Näherung, d. h. der Fisher-Information der Jacobi-Matrix)")
        ui.table_show(fs.reconstruct_dataframe(study))
        st.markdown("**Randverteilungen**" + (" (blau Laplace, orange Posterior aus emcee mit 68 %- und 95 %-Bereich)" if rec.get("posterior") else ""))
        ui.show_fig(fs.fig_posterior(study), "posterior.png", "st_post_png")
        post = rec.get("posterior")
        if post:
            st.caption(f"emcee: {post['walkers']} Walker, {post['steps']} Schritte, Annahmerate {post['acceptance']:.2f}, "
                       f"Autokorrelationszeiten {', '.join(f'{t:.1f}' for t in post.get('autocorr', []))} Schritte.")
        c = st.columns([2, 3])
        with c[0]:
            st.markdown("**Korrelationen**")
            ui.show_fig(fs.fig_correlation(study), "korrelation.png", "st_corr_png")
        with c[1]:
            st.markdown("**Messwerte und Anpassung**")
            ui.table_show(fs.residual_dataframe(study))
    elif task == "uq":
        u = res["uq"]
        st.caption(("Linearisierung um den Referenzpunkt (eine Lösung mit Ableitungen)" if u["propagation"] == "linear" else
                    "Monte Carlo auf einem Gauß-Prozess-Ersatzmodell") + f" · unsichere Eingänge: {', '.join(u['inputs'])}")
        if u.get("clipped"):
            st.info(f"{u['clipped']} Stichproben lagen außerhalb der Parametergrenzen und wurden begrenzt.")
        ui.table_show(fs.uq_dataframe(study))
        ui.show_fig(fs.fig_uq(study), "streuung.png", "st_uq_png")
        if u.get("sobol"):
            st.markdown("**Sobol'-Indizes**: Anteil der Varianz einer Messgröße, der auf einen Eingang allein (S₁) bzw. mit allen Wechselwirkungen "
                        "(S_T) zurückgeht.")
            ui.show_fig(fs.fig_sobol(study), "sobol.png", "st_sobol_png")
            ui.table_show(fs.sobol_dataframe(study))
    else:
        st.info(f"Aufgabe {task!r}: keine Studienansicht.")


# ------------------------------------------------------------------------------------------------------------------ 5 table
def tab_table(study):
    if study is None:
        return _no_study()
    df = fs.history(study)
    ui.table_show(df)
    c = st.columns(3)
    c[0].download_button("Verlauf als CSV", ui.csv_bytes(df), file_name="verlauf.csv", mime="text/csv", key="st_csv")
    if study.get("results"):
        c[1].download_button("results.json", json.dumps(study["results"], indent=1).encode("utf-8"), file_name="results.json",
                             mime="application/json", key="st_res_dl")
    if study.get("store_path"):
        c[2].download_button("Studienspeicher (.study.jsonl)", Path(study["store_path"]).read_bytes(), file_name=Path(study["store_path"]).name,
                             key="st_store_dl")
    if study.get("log"):
        with st.expander("Protokoll"):
            st.code("\n".join(study["log"].splitlines()[-60:]))


def tab_info():
    st.markdown("""
**Studien** rechnen eine Gitterzelle (die Zelle des Job-Runners, nicht das Modell der anderen Modi) für viele Parameterwerte. Das Netz wird nicht
neu erzeugt, sondern verformt (`morph`), so dass die Ableitungen der Bibliothek zu den Werten passen.

- **Optimierung** (`optimize`): ein Ziel (`objective`, Index oder Name einer Messgröße), `maximize`, Verfahren `L-BFGS-B` (mit Ableitungen),
  `Nelder-Mead`, `differential-evolution` oder `bayesian` (Gauß-Prozess, `acquisition` `ei` oder `lcb`, `use_gradients`).
- **Rekonstruktion** (`reconstruct`): `measured` (Messwerte in der Reihenfolge der Messgrößen) mit `sigma`, oder `synthetic` (Daten aus dem
  Modell mit Rauschen, zum Prüfen); Levenberg-Marquardt mit Unsicherheiten aus der Laplace-Näherung, mit `posterior` zusätzlich Stichproben
  (emcee).
- **UQ** (`uq`): `inputs` mit `normal` [Mittel, σ] oder `uniform` [unten, oben], `propagation` `linear` oder `surrogate` (`points`,
  `samples`), `sobol` für die Sensitivitätsindizes.

Die **Messgrößen** sind die Effizienzen der Ordnungen jeder Konfiguration, z. B. „s 405nm 40deg: R0“. Die **Pareto-Front** wertet den Verlauf
aus: sie zeigt, welche der gerechneten Punkte zwei Ziele am besten ausgleichen.

Der Ordner einer Studie enthält `job.json`, `log.txt` (Ereignisse), `<name>.study.jsonl` (jede Auswertung, in SI) und am Ende `results.json`.
""")
