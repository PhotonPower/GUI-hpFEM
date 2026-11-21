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


BOTORCH_HINT = ("Mehrzieloptimierung (method „pareto“) und Mehrstufen-Optimierung (method „multi_fidelity“) brauchen BoTorch im Python der Rechnung: "
                "pip install \"hpfem[opt-bo]\" (installiert auch PyTorch).")


def optimize_needs_botorch(job):
    return fs.optimize_spec(job).get("method") in ("pareto", "multi_fidelity")


def ext_form(job, ctx):
    """Form „Nebenbedingungen und Mehrziel“: writes the extended keys of the block optimize into the job document (the JSON stays editable)."""
    import pandas as pd

    spec = fs.optimize_spec(job)
    labels = fs.observable_labels(job)
    names = [p["name"] for p in job["parameters"]]
    has_bo = bool(ctx["FEATS"].get("opt_bo", False))
    with st.expander("Nebenbedingungen und Mehrziel", expanded=bool(fr.study_needs_worker(job))):
        st.caption("Erweiterungen der Optimierung, die der Job-Runner der Bibliothek nicht kennt (die Studie läuft dann über den Studien-Worker dieser App): "
                   "Nebenbedingungen an Messgrößen und an die Parameter, Mehrzieloptimierung und Mehrstufen-Optimierung.")
        k = f"st_ext_{S.st_job_ver}"
        cur = spec.get("method", "bayesian")
        opts = list(fs.OPT_METHODS)
        method = st.selectbox("Verfahren", opts, index=opts.index(cur) if cur in opts else 0, key=f"{k}_m",
                              format_func=lambda m: {"pareto": "pareto (Mehrziel, BoTorch)", "multi_fidelity": "multi_fidelity (Mehrstufen, BoTorch)",
                                                     "bayesian": "bayesian (Gauß-Prozess)"}.get(m, m))
        if method in ("pareto", "multi_fidelity") and not has_bo:
            st.warning(BOTORCH_HINT + " Die Auswahl ist möglich, übernehmen und starten geht erst mit BoTorch.")
        c = st.columns(2)
        max_eval = c[0].number_input("Höchstzahl der Auswertungen", 2, 2000, int(spec.get("max_evaluations") or 30), key=f"{k}_n")
        st.markdown("**Nebenbedingungen an Messgrößen** (nur mit bayesian): Messgröße muss zwischen den Grenzen liegen; leer lassen = unbegrenzt.")
        rows = [{"Messgröße": labels[fs._observable_index(c_.get("observable", 0), labels)] if fs._observable_index(c_.get("observable", 0), labels) is not None
                 else None, "von": c_.get("lower"), "bis": c_.get("upper")} for c_ in spec.get("constraints") or []]
        oc = st.data_editor(pd.DataFrame(rows, columns=["Messgröße", "von", "bis"]).astype({"von": float, "bis": float}), num_rows="dynamic", key=f"{k}_oc",
                            hide_index=True, column_config={"Messgröße": st.column_config.SelectboxColumn(options=labels, required=True)})
        st.markdown("**Nebenbedingungen an Parameter** (lineare Kombination in Job-Einheiten): z. B. `width: 1, eps: 20` zwischen von und bis. "
                    f"Parameter: {', '.join(names)}.")
        rows = [{"Koeffizienten": ", ".join(f"{n}: {v:g}" for n, v in (c_.get("coefficients") or {}).items()), "von": c_.get("lower"), "bis": c_.get("upper")}
                for c_ in spec.get("linear_constraints") or []]
        lc = st.data_editor(pd.DataFrame(rows, columns=["Koeffizienten", "von", "bis"]).astype({"von": float, "bis": float}), num_rows="dynamic", key=f"{k}_lc",
                            hide_index=True)
        objectives, maximize, ref, fids = (), (), "", ""
        if method == "pareto":
            cur_obj = [o for o in spec.get("objectives", []) if o in labels] or labels[:2]
            objectives = st.multiselect("Zielgrößen (mindestens zwei)", labels, default=cur_obj, key=f"{k}_obj")
            mx = spec.get("maximize", False)
            mxl = list(mx) if isinstance(mx, list) else [bool(mx)] * len(objectives)
            maximize = [st.checkbox(f"{o}: maximieren", value=bool(mxl[i]) if i < len(mxl) else False, key=f"{k}_mx{i}") for i, o in enumerate(objectives)]
            ref = st.text_input("Referenzpunkt (ein Wert je Zielgröße, optional)", value=", ".join(f"{v:g}" for v in spec.get("ref_point") or []), key=f"{k}_ref",
                                help="Bezugspunkt der Hypervolumen-Verbesserung in Einheiten der Zielgrößen; leer: schlechtester beobachteter Wert minus 10 % der Spanne.")
        elif method == "multi_fidelity":
            fids = st.text_input("Ordnungen der Stufen (billigste zuerst)", value=", ".join(str(f.get("order")) for f in spec.get("fidelities", [])) or "2, 3",
                                 key=f"{k}_fid", help="Polynomordnungen p, z. B. 2, 3, 4: die letzte ist die Zielstufe.")
        elif method != "bayesian" and len(oc):
            st.info("Nebenbedingungen an Messgrößen gehen nur mit dem Verfahren bayesian.")
        blocked = method in ("pareto", "multi_fidelity") and not has_bo
        if st.button("In das Job-Dokument übernehmen", key=f"{k}_apply", disabled=blocked):
            try:
                new = fs.apply_extended(
                    job, method=method, max_evaluations=max_eval,
                    outcome=[(r["Messgröße"], r["von"], r["bis"]) for _, r in oc.iterrows() if r["Messgröße"]],
                    linear=[(r["Koeffizienten"], r["von"], r["bis"]) for _, r in lc.iterrows() if str(r["Koeffizienten"] or "").strip()],
                    objectives=objectives, maximize=maximize, ref_point=ref, fidelities=fids)
            except ValueError as exc:
                st.error(str(exc))
            else:
                S.st_job_text = json.dumps(new, indent=1, ensure_ascii=False)
                S.st_job_ver += 1
                st.rerun()


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
        task = st.selectbox("Vorlage", [*fs.TASKS, "optimize_constrained"], format_func={**fs.TASK_LABELS, "optimize_constrained": "Optimierung mit Nebenbedingungen"}.get,
                            key="st_tpl_task",
                            help="Eine Lamellenzelle (Ordnung 2, drei Messgrößen) mit Breite und ε des Stegs als Parametern: rechnet in "
                                 "etwa zwei Sekunden je Auswertung. „Mit Nebenbedingungen“ begrenzt eine Messgröße und die Breite.")
        if st.button("Vorlage übernehmen", key="st_tpl_load"):
            S.st_job_text = json.dumps(fs.template_constrained() if task == "optimize_constrained" else fs.template(task), indent=1, ensure_ascii=False)
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
        if job and not errs and job.get("task") == "optimize":
            ext_form(job, ctx)
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
        needs_ext = bool(job) and fr.study_needs_worker(job)
        if needs_ext and lib_ok and not ctx["FEATS"].get("study_ext", False):
            st.error("Die erweiterten Optionen (Nebenbedingungen, Pareto, Mehrstufen) brauchen hpfem 0.5.x mit hpfem.run._optimize im eingestellten Python.")
            lib_ok = False
        if needs_ext and optimize_needs_botorch(job) and not ctx["FEATS"].get("opt_bo", False):
            st.error(BOTORCH_HINT)
            lib_ok = False
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
    multi = inf.get("method") == "pareto" and inf.get("objectives")
    st.caption("Nicht dominierte Auswertungen zweier Messgrößen: keine andere Auswertung ist in beiden mindestens so gut und in einer besser. " +
               ("Diese Studie ist eine Mehrzieloptimierung: Vorgabe sind ihre Zielgrößen; der Referenzpunkt ist als Stern eingezeichnet. " if multi else
                "Die Front zeigt den Kompromiss zwischen zwei Zielen aus allen bisherigen Auswertungen (die Optimierung selbst verfolgt ein Ziel). ") +
               ("Auswertungen, die eine Nebenbedingung verletzen, sind rote Kreuze und zählen nicht zur Front." if "Zulässig" in df else ""))
    d0 = inf["objectives"][0] if multi else 0
    d1 = inf["objectives"][1] if multi else 1
    m0 = inf["maximize_list"][0] if multi else False
    m1 = inf["maximize_list"][1] if multi else True
    c = st.columns(4)
    x = c[0].selectbox("Größe 1", obs, index=d0, key="st_p_x")
    mx = c[1].radio("Größe 1", ["max", "min"], horizontal=True, index=0 if m0 else 1, key="st_p_mx", label_visibility="hidden") == "max"
    y = c[2].selectbox("Größe 2", obs, index=d1, key="st_p_y")
    my = c[3].radio("Größe 2", ["max", "min"], horizontal=True, index=0 if m1 else 1, key="st_p_my", label_visibility="hidden") == "max"
    if x == y:
        st.warning("Zwei verschiedene Größen wählen.")
        return
    ref = (study.get("results") or {}).get("optimize", {}).get("ref_point") if multi else None
    ref = (ref[0], ref[1]) if ref and len(ref) >= 2 and (x, y) == (obs[d0], obs[d1]) else None
    ui.show_fig(fs.fig_pareto(df, x, y, mx, my, ref), "pareto.png", "st_p_png")
    ok_df = df[(df["Status"] == "ok") & (df["Zulässig"] if "Zulässig" in df else True)]
    front = fs.pareto_front(ok_df, x, y, mx, my)
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
        if o.get("method") == "pareto":
            fr_df = fs.front_dataframe(study)
            st.markdown(f"**Pareto-Front der Mehrzieloptimierung** ({len(fr_df)} Punkte; Ziele: {', '.join(o.get('objectives', []))})")
            ui.table_show(fr_df)
            st.download_button("Front als CSV", ui.csv_bytes(fr_df), file_name="front.csv", mime="text/csv", key="st_front_csv")
            st.caption("Die Tabelle „Optimum“ oben zeigt den ersten Punkt der Front (kleinster Wert der ersten Zielgröße); die Front ist die Lösung.")
        if o.get("method") == "multi_fidelity":
            st.caption("Stufen: " + ", ".join(f"p = {f.get('order')}" for f in o.get("fidelities", [])) + ". Das Ergebnis ist die beste Auswertung auf der letzten Stufe.")
        cl = fs.constraint_list(study.get("job"))
        if cl:
            hist = fs.history(study)
            bad = fs.constraint_violations(study.get("job"), hist, inf["observables"])
            n_bad = {n: int(m.sum()) for n, m in bad.items()}
            st.markdown("**Nebenbedingungen**: " + "; ".join(f"{n} ({'Messgröße' if k == 'outcome' else 'Parameter'}): {n_bad.get(n, 0)} Auswertungen verletzen sie" for n, k, _ in cl))
            if o.get("feasible") is False:
                st.warning("Das Ergebnis erfüllt die Nebenbedingungen an Messgrößen nicht: es wurde kein zulässiger Punkt gefunden (Grenzen lockern oder mehr Auswertungen).")
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
  **Erweitert** (Formular „Nebenbedingungen und Mehrziel“, Studien-Worker der App statt `hpfem.run`): `constraints` (Messgröße zwischen `lower` und
  `upper`, nur `bayesian`), `linear_constraints` (`coefficients` auf die Parameter in Job-Einheiten, `lower`, `upper`; alle Verfahren), Verfahren
  `pareto` (`objectives`, `maximize`, `ref_point`; BoTorch, `pip install "hpfem[opt-bo]"`) und `multi_fidelity` (`fidelities`, z. B.
  `[{"order": 2}, {"order": 3}]`; BoTorch).
- **Rekonstruktion** (`reconstruct`): `measured` (Messwerte in der Reihenfolge der Messgrößen) mit `sigma`, oder `synthetic` (Daten aus dem
  Modell mit Rauschen, zum Prüfen); Levenberg-Marquardt mit Unsicherheiten aus der Laplace-Näherung, mit `posterior` zusätzlich Stichproben
  (emcee).
- **UQ** (`uq`): `inputs` mit `normal` [Mittel, σ] oder `uniform` [unten, oben], `propagation` `linear` oder `surrogate` (`points`,
  `samples`), `sobol` für die Sensitivitätsindizes.

Die **Messgrößen** sind die Effizienzen der Ordnungen jeder Konfiguration, z. B. „s 405nm 40deg: R0“. Die **Pareto-Front** wertet den Verlauf
aus: sie zeigt, welche der gerechneten Punkte zwei Ziele am besten ausgleichen.

Der Ordner einer Studie enthält `job.json`, `log.txt` (Ereignisse), `<name>.study.jsonl` (jede Auswertung, in SI) und am Ende `results.json`.
""")
