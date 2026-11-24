"""Cross grating (doubly periodic) in the FEM model builder (Streamlit), hp-FEM M21. Called from fem_app.py: ``render_xgrating(ctx)``.

Model, layers with holes and pillars, incidence, wavelength sweep, computation with ``hpfem.crossed`` (worker: fem_xgrating_worker.py), table of the
diffraction orders (R, T per order (m, n), energy balance), convergence in p, s / p symmetry check, comparison with FourierGrating's
``efficiencies.csv`` and import of FourierGrating models. Models, tables and figures: fem_xgrating.py.
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
import fem_ui as ui
import fem_xgrating as fx

S = st.session_state

INFO = """
**Modell.** Ein Gitter, das in x und y periodisch ist (Perioden p_x, p_y), mit der Stapelnormalen z. Von oben nach unten: Einfallsmedium, Schichten,
Substrat. Jede Schicht hat ein Hintergrundmaterial und Formen in der xy-Ebene (Kreis, Ellipse, Rechteck); spätere Formen überdecken frühere.
Ein **Loch** und eine **Säule** füllen ihr Gebiet mit dem gewählten Material; die Art legt nur das vorgeschlagene Material fest (Loch: Material
des Einfallsmediums, Säule: eigenes Material). Die Einheitszelle ist [0, p_x] × [0, p_y]; Formen, die über den Rand reichen, setzen sich im
Nachbarn periodisch fort.

**Einfall.** Wellenlänge im Vakuum, Polarwinkel θ gegen die Normale z, Azimut φ der Einfallsebene gegen x, Polarisation s (E senkrecht zur
Einfallsebene; bei θ = 0, φ = 0 längs −y) oder p (E in der Einfallsebene; bei θ = 0, φ = 0 längs +x). Zeitfaktor exp(−iωt).

**Rechnung.** `hpfem.crossed.solve`: Gmsh-Netz der Einheitszelle (gekrümmte Tetraeder, identische Netze auf gegenüberliegenden Seitenflächen), geschichteter
Hintergrund, PML oben und unten, Bloch-Phasen in x und y; s und p aus einer Faktorisierung. Einfallsmedium und Substrat müssen verlustfreie Dielektrika sein. Die
Kosten sind die Faktorisierung: das Lochgitter des Beispiels bei p = 3 hat rund 40 000 Unbekannte und braucht etwa eine Minute je Wellenlänge. Für eine
**Konvergenztabelle** bleibt „Elemente je Wellenlänge“ fest (nicht 0) und p wächst.

**Probe.** Energiebilanz ΣR + ΣT + A − 1 (verlustfrei ≈ 0). Symmetrie: bei vierzähliger Symmetrie des Lochs (Kreis, Quadrat in der Mitte der Zelle, gleiche Perioden)
sind die Ergebnisse für s und p bei senkrechtem Einfall gleich (Reiter „Ergebnisse“, Anzeige „s“ und „p“). Schräger Einfall (θ, φ ≠ 0) setzt die Bloch-Phasen in
beiden Richtungen.

**Ordnungen.** Vor der Rechnung zeigt Reiter 2, welche Beugungsordnungen (m, n) bei der gewählten Wellenlänge im Einfallsmedium (Reflexion) und im Substrat
(Transmission) ausbreitungsfähig sind: k_t = k_t,ein + 2π (m / p_x, n / p_y), |k_t| < k₀ n. Nach der Rechnung stehen R und T je Ordnung im Reiter 3.

**Vergleich mit FourierGrating.** Die `efficiencies.csv` von FourierGrating (Spalten `lambda_um, phi_deg, psi_deg, delta_deg, side, n, m, efficiency`) lässt sich im
Reiter 3 laden; die Ordnungen werden umbenannt (FourierGrating (n, m) längs (x, z) ist hier (m, n) = (n, −m)), δ = 0° gilt als s, δ = 90° als p. Der Import einer
FourierGrating-Konfiguration (Reiter 1) läuft über `hpfem.fouriergrating` im Python mit hpfem (Rahmen, Schichtreihenfolge, ψ = −φ, p = −p): nur Lochgitter und Säulen
aus Kreis, Ellipse, Polygon und Rechteck, konstante Materialien, δ = 0° oder 90°.

**Austausch.** Das Modell lässt sich als JSON speichern und laden. Fremde Materialangaben ohne „type“ (z. B. {"n": 1.5}) werden beim Laden
umgesetzt.
"""


def _init():
    if "xg_model" in S:
        return
    S.xg_model = copy.deepcopy(fx.default_model())
    S.xg_ver, S.xg_proc, S.xg_dir, S.xg_msg = 0, None, None, None


def _bump():
    S.xg_ver += 1


def _num(col, label, key, value, step=10.0, lo=-1e6, hi=1e6, help=None):
    return float(col.number_input(label, lo, hi, float(value), step, key=key, format="%.6g", help=help))


def render_xgrating(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    st.title("FEM-Modellwerkstatt: Kreuzgitter (doppelt periodisch)")
    st.caption("Lochgitter und Säulengitter, in x und y periodisch, mit Schichten, schrägem Einfall und Tabelle der Beugungsordnungen (hpfem.crossed, hp-FEM M21).")
    if S.xg_proc is not None:
        return _running()
    work = Path(ctx["work"]) / "xgrating"
    t1, t2, t3, t4 = st.tabs(["1 Modell", "2 Rechnung", "3 Ergebnisse", "4 Info"])
    with t1:
        _tab_model(ctx, work)
    with t2:
        _tab_run(ctx, work)
    with t3:
        _tab_results(work)
    with t4:
        st.markdown(INFO)


def _import_fg(ctx, work, up):
    """FourierGrating file -> model (``fem_xgrating_worker.py --import`` in the Python with hpfem)."""
    work.mkdir(parents=True, exist_ok=True)
    src, dst = work / "fg_import.json", work / "fg_model.json"
    src.write_bytes(up.getvalue())
    if dst.exists():
        dst.unlink()
    r = subprocess.run([str(ctx["py"]), str(fr.HERE / "fem_xgrating_worker.py"), "--import", str(src), str(dst)], capture_output=True, text=True,
                       cwd=str(work), env=fr.worker_env(ctx["repo"], 1), timeout=120)
    if r.returncode != 0 or not dst.is_file():
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("ERROR")] or (r.stdout + r.stderr).strip().splitlines()[-3:]
        raise ValueError("; ".join(ln.removeprefix("ERROR ") for ln in lines) or "unbekannter Fehler")
    d = json.loads(dst.read_text(encoding="utf-8"))
    return fx.load_model(json.dumps(d["model"])), d.get("notes", [])


def _tab_model(ctx, work):
    m, ver = S.xg_model, S.xg_ver
    left, right = st.columns([6, 5])
    with left:
        presets = fx.presets()
        c = st.columns([3, 1])
        pick = c[0].selectbox("Vorlage", ["– auswählen –"] + list(presets), key="xg_preset")
        if c[1].button("Laden", key="xg_preset_load", disabled=pick.startswith("–")):
            S.xg_model = copy.deepcopy(presets[pick])
            _bump()
            st.rerun()
        c = st.columns([3, 2])
        up = c[0].file_uploader("Modell laden (JSON)", type=["json"], key="xg_upload")
        if up is not None and S.get("xg_loaded") != (up.name, up.size):
            try:
                S.xg_model = fx.load_model(up.getvalue().decode("utf-8"))
                S.xg_loaded = (up.name, up.size)
                _bump()
                st.rerun()
            except Exception as exc:
                st.error(f"Datei nicht lesbar: {exc}")
        c[1].download_button("Modell speichern (JSON)", json.dumps(m, indent=1, ensure_ascii=False).encode("utf-8"), file_name="kreuzgitter.json",
                             mime="application/json", key="xg_save")
        upf = st.file_uploader("FourierGrating-Konfiguration importieren (JSON, 2D)", type=["json"], key="xg_fg_upload",
                               help="Über hpfem.fouriergrating im Python mit hpfem: Rahmen (FourierGrating: y ist die Normale), Schichtreihenfolge, ψ = −φ, p = −p. "
                                    "Unterstützt: Lochgitter und Säulen aus Kreis, Ellipse, Polygon, Rechteck, konstante Materialien, δ = 0° oder 90°.")
        if upf is not None and S.get("xg_fg_loaded") != (upf.name, upf.size):
            S.xg_fg_loaded = (upf.name, upf.size)
            try:
                S.xg_model, notes = _import_fg(ctx, work, upf)
                S.xg_imp_msg = "FourierGrating-Modell importiert." + "".join(f" {n}." for n in notes)
                _bump()
                st.rerun()
            except Exception as exc:
                st.error(f"Import nicht möglich: {exc}")
        if S.get("xg_imp_msg"):
            st.success(S.xg_imp_msg)
        m["name"] = st.text_input("Name", value=m["name"], key=f"xg_name_{ver}")
        c = st.columns(2)
        m["period_x_nm"] = _num(c[0], "Periode p_x (nm)", f"xg_px_{ver}", m["period_x_nm"], 10.0, 1.0)
        m["period_y_nm"] = _num(c[1], "Periode p_y (nm)", f"xg_py_{ver}", m["period_y_nm"], 10.0, 1.0)
        inc, sw = m["incidence"], m["sweep"]
        st.subheader("Materialien")
        lam0 = fx.sweep_list(sw, inc)[0]
        ui.material_editor(m, ver, lam0, fx.used_materials(m), fx.rename_material, _bump, prefix="xg_")
        names = list(m["materials"])
        st.subheader("Stapel (von oben nach unten)")
        m["cover"] = st.selectbox("Einfallsmedium (Superstrat)", names, index=names.index(m["cover"]) if m["cover"] in names else 0, key=f"xg_cov_{ver}")
        for i, la in enumerate(list(m["layers"])):
            with st.container(border=True):
                _layer_editor(m, i, la, names, ver)
        if st.button("Schicht hinzufügen", key=f"xg_addlayer_{ver}"):
            m["layers"].append({"thickness_nm": 100.0, "material": names[-1], "shapes": []})
            _bump()
            st.rerun()
        m["substrate"] = st.selectbox("Substrat", names, index=names.index(m["substrate"]) if m["substrate"] in names else 0, key=f"xg_sub_{ver}")
        st.subheader("Einfall")
        c = st.columns(4)
        inc["theta"] = _num(c[0], "Polarwinkel θ (°)", f"xg_th_{ver}", inc["theta"], 1.0, 0.0, 89.9, "Gegen die Stapelnormale z; 0° ist senkrechter Einfall.")
        inc["phi"] = _num(c[1], "Azimut φ (°)", f"xg_ph_{ver}", inc["phi"], 1.0, -360.0, 360.0, "Richtung der Einfallsebene gegen die x-Achse; bei θ = 0 ohne Wirkung auf die Ordnungen.")
        inc["pol"] = c[2].radio("Polarisation", list(fx.POLS), index=list(fx.POLS).index(inc["pol"]) if inc["pol"] in fx.POLS else 0, horizontal=True,
                                format_func=lambda k: k, key=f"xg_pol_{ver}", help="; ".join(fx.POLS.values()))
        sw["mode"] = c[3].radio("Wellenlänge", ["single", "wavelength"], index=["single", "wavelength"].index(sw["mode"]) if sw["mode"] in ("single", "wavelength") else 0,
                                format_func={"single": "ein Wert", "wavelength": "Durchlauf"}.get, key=f"xg_swm_{ver}")
        c = st.columns(3)
        if sw["mode"] == "single":
            inc["wavelength_nm"] = _num(c[0], "λ (nm)", f"xg_lam_{ver}", inc["wavelength_nm"], 10.0, 1.0, 100000.0)
        else:
            sw["start"] = _num(c[0], "λ von (nm)", f"xg_swa_{ver}", sw["start"], 10.0, 1.0, 100000.0)
            sw["stop"] = _num(c[1], "λ bis (nm)", f"xg_swb_{ver}", sw["stop"], 10.0, 1.0, 100000.0)
            sw["n"] = int(c[2].number_input("Punkte", 2, 400, int(sw["n"]), key=f"xg_swn_{ver}"))
    with right:
        for lvl, text in fx.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        try:
            ui.show_fig(fx.fig_stack(m), "kreuzgitter_stapel.png", f"xg_stack_dl_{ver}")
            if m["layers"]:
                c = st.columns(2)
                k = int(c[0].selectbox("Schicht in der Aufsicht", range(len(m["layers"])), format_func=lambda i: f"Schicht {i + 1}", key=f"xg_viewlayer_{ver}"))
                nc = int(c[1].radio("Zellen", [1, 2, 3], horizontal=True, key=f"xg_viewn_{ver}", help="Zahl der gezeichneten Perioden je Richtung."))
                ui.show_fig(fx.fig_cell(m, min(k, len(m["layers"]) - 1), nc), "kreuzgitter_zelle.png", f"xg_cell_dl_{ver}")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")


def _layer_editor(m, i, la, names, ver):
    k = f"xg_l{ver}_{i}"
    head = st.columns([3, 3, 1, 1, 1])
    head[0].markdown(f"**Schicht {i + 1}**")
    la["material"] = head[1].selectbox("Hintergrund", names, index=names.index(la["material"]) if la["material"] in names else 0, key=f"{k}_m",
                                       label_visibility="collapsed", help="Hintergrundmaterial der Schicht")
    if head[2].button("↑", key=f"{k}_up", disabled=i == 0, help="Schicht nach oben"):
        m["layers"][i - 1], m["layers"][i] = m["layers"][i], m["layers"][i - 1]
        _bump()
        st.rerun()
    if head[3].button("↓", key=f"{k}_dn", disabled=i == len(m["layers"]) - 1, help="Schicht nach unten"):
        m["layers"][i + 1], m["layers"][i] = m["layers"][i], m["layers"][i + 1]
        _bump()
        st.rerun()
    if head[4].button("✕", key=f"{k}_del", help="Schicht entfernen"):
        m["layers"].pop(i)
        _bump()
        st.rerun()
    la["thickness_nm"] = _num(st, "Dicke (nm)", f"{k}_d", la["thickness_nm"], 10.0, 0.0)
    for j, s in enumerate(list(la["shapes"])):
        _shape_editor(m, la, j, s, names, f"{k}_s{j}")
    c = st.columns([2, 2, 1])
    t = c[0].selectbox("Neue Form", list(fx.SHAPES), format_func=fx.SHAPES.get, key=f"{k}_nt", label_visibility="collapsed")
    kind = c[1].selectbox("Art", list(fx.KINDS), format_func=fx.KINDS.get, key=f"{k}_nk", label_visibility="collapsed")
    if c[2].button("＋ Form", key=f"{k}_add"):
        mat = m["cover"] if kind == "hole" else next((n for n in names if n != la["material"]), names[0])
        la["shapes"].append(fx.new_shape(t, mat, kind, (m["period_x_nm"], m["period_y_nm"])))
        _bump()
        st.rerun()


def _shape_editor(m, la, j, s, names, k):
    head = st.columns([2, 2, 3, 1])
    s["type"] = head[0].selectbox("Form", list(fx.SHAPES), index=list(fx.SHAPES).index(s["type"]), format_func=fx.SHAPES.get, key=f"{k}_t", label_visibility="collapsed")
    for key_, default in fx.DEFAULT_SHAPE[s["type"]].items():
        s.setdefault(key_, default)
    s["kind"] = head[1].selectbox("Art", list(fx.KINDS), index=list(fx.KINDS).index(s.get("kind", "hole")), format_func=fx.KINDS.get, key=f"{k}_k",
                                  label_visibility="collapsed")
    s["material"] = head[2].selectbox("Material", names, index=names.index(s["material"]) if s["material"] in names else 0, key=f"{k}_mat",
                                      label_visibility="collapsed")
    if head[3].button("✕", key=f"{k}_del", help="Form entfernen"):
        la["shapes"].pop(j)
        _bump()
        st.rerun()
    if s["type"] == "polygon":
        text = "\n".join(f"{p[0]:g}, {p[1]:g}" for p in s["points"])
        new = st.text_area("Eckpunkte (x, y in nm, eine Zeile je Punkt)", value=text, key=f"{k}_pts", height=100)
        try:
            pts = [[float(v) for v in ln.replace(";", ",").split(",")[:2]] for ln in new.splitlines() if ln.strip()]
            if all(len(p) == 2 for p in pts):
                s["points"] = pts
        except ValueError:
            st.error("Die Punkte als „x, y“ eingeben.")
        return
    c = st.columns(5)
    s["cx"], s["cy"] = _num(c[0], "Mitte x (nm)", f"{k}_cx", s["cx"]), _num(c[1], "Mitte y (nm)", f"{k}_cy", s["cy"])
    if s["type"] == "circle":
        s["radius"] = _num(c[2], "Radius (nm)", f"{k}_r", s["radius"], 5.0, 0.0)
    elif s["type"] == "ellipse":
        s["rx"], s["ry"] = _num(c[2], "Halbachse x (nm)", f"{k}_rx", s["rx"], 5.0, 0.0), _num(c[3], "Halbachse y (nm)", f"{k}_ry", s["ry"], 5.0, 0.0)
    else:
        s["wx"], s["wy"] = _num(c[2], "Breite x (nm)", f"{k}_wx", s["wx"], 5.0, 0.0), _num(c[3], "Breite y (nm)", f"{k}_wy", s["wy"], 5.0, 0.0)
    if s["type"] != "circle":
        s["angle"] = _num(c[4], "Drehung (°)", f"{k}_ang", s.get("angle", 0.0), 5.0, -360.0, 360.0, "Gegen den Uhrzeigersinn um die Mitte.")


def _tab_run(ctx, work):
    m, ver = S.xg_model, S.xg_ver
    errs = [t for lvl, t in fx.validate(m) if lvl == "error"]
    for e in errs:
        st.error(e)
    me, cv = m["mesh"], m["convergence"]
    st.subheader("Netz und Ordnung")
    c = st.columns(3)
    me["order"] = int(c[0].number_input("Ordnung p", 1, 6, int(me["order"]), key=f"xg_p_{ver}"))
    me["elements_per_wavelength"] = _num(c[1], "Elemente je Wellenlänge (0 = automatisch)", f"xg_epw_{ver}", me["elements_per_wavelength"], 0.5, 0.0, 40.0,
                                         "0: die Bibliothek wählt 8 / p je Wellenlänge im Material. Für Konvergenztabellen fest wählen (z. B. 2,5) und p erhöhen.")
    cv["enabled"] = c[2].checkbox("Konvergenzstudie in p", value=bool(cv["enabled"]), key=f"xg_cv_{ver}",
                                  help="Zusätzlich je eine Rechnung bei den genannten Ordnungen mit der ersten Wellenlänge (Tabelle und Bild im Reiter 3).")
    if cv["enabled"]:
        txt = st.text_input("Ordnungen der Studie (durch Komma getrennt)", value=", ".join(str(o) for o in cv["orders"]), key=f"xg_cvo_{ver}")
        try:
            cv["orders"] = sorted({int(v) for v in txt.replace(";", ",").split(",") if v.strip()})
            if not cv["orders"] or min(cv["orders"]) < 1 or max(cv["orders"]) > 6:
                st.error("Ordnungen zwischen 1 und 6 angeben.")
                errs.append("Ordnungen")
        except ValueError:
            st.error("Ganze Zahlen angeben, z. B. 2, 3, 4.")
            errs.append("Ordnungen")
        if not me["elements_per_wavelength"]:
            st.warning("Für eine Konvergenztabelle „Elemente je Wellenlänge“ fest wählen (nicht 0), sonst ändert sich mit p auch das Netz.")
    fl = m["field"]
    fl["enabled"] = st.checkbox("Nahfeld |E|² auf Schnittebenen speichern", value=bool(fl["enabled"]), key=f"xg_fe_{ver}",
                                help="Nach jeder Rechnung wird |E|²/|E₀|² (Gesamtfeld, Einfall mit Betrag 1) auf einer xy-, xz- und yz-Ebene ausgewertet (Reiter 3, „Nahfeld“).")
    if fl["enabled"]:
        pl = fx.field_planes(m)
        c = st.columns(4)
        fl["resolution"] = int(c[0].number_input("Punkte je Ebene (längs x)", 20, 300, int(fl["resolution"]), 10, key=f"xg_fr_{ver}"))
        fl["x_nm"] = _num(c[1], "Schnitt yz bei x (nm)", f"xg_fx_{ver}", pl["x_nm"], 10.0, 0.0, m["period_x_nm"])
        fl["y_nm"] = _num(c[2], "Schnitt xz bei y (nm)", f"xg_fy_{ver}", pl["y_nm"], 10.0, 0.0, m["period_y_nm"])
        fl["z_nm"] = _num(c[3], "Schnitt xy bei z (nm)", f"xg_fz_{ver}", pl["z_nm"], 10.0, -1e5, 1e5,
                          "z = 0 ist die Oberkante der Schichten, die Schichten liegen bei negativem z. Standard: Mitte der ersten Schicht.")
        st.caption("Die Auswertung der Punkte kostet je Wellenlänge zusätzlich Zeit (drei Ebenen); bei vielen Wellenlängen die Punktzahl klein halten.")
    lams = fx.sweep_list(m["sweep"], m["incidence"])
    n_runs = len(lams) + (len([o for o in cv["orders"] if o != me["order"]]) if cv["enabled"] else 0)
    st.markdown(f"**{n_runs} Rechnung(en).** Jede ist eine volle Lösung mit Netz; das Lochgitter des Beispiels hat bei p = 3 etwa 40 000 Unbekannte und braucht rund eine Minute.")
    if ctx["lib"].get("error"):
        st.warning("Das eingestellte Python hat kein hpfem: Rechnen geht nicht, die Vorschau der Ordnungen unten schon.")
    elif not ctx["FEATS"].get("crossed"):
        st.warning("Das eingestellte hpfem kann keine Kreuzgitter rechnen: `hpfem.crossed` samt `OrderPlane` und `diffraction_orders` fehlt (hp-FEM M21, ab Stand "
                   "von PhotonPower/hp-FEM#240; bei einem Quell-Build die Erweiterung neu bauen).")
    if st.button("Rechnen", type="primary", key="xg_run", disabled=bool(errs) or bool(ctx["lib"].get("error")) or not ctx["FEATS"].get("crossed")):
        try:
            work.mkdir(parents=True, exist_ok=True)
            (work / "job.json").write_text(json.dumps(fx.job(m), ensure_ascii=False), encoding="utf-8")
            S.xg_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script="fem_xgrating_worker.py")
            S.xg_dir, S.xg_cancel, S.xg_msg = str(work), False, None
            S.xg_run_model = copy.deepcopy(m)
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("xg_msg"):
        (st.success if S.xg_msg[0] == "ok" else st.error)(S.xg_msg[1] + " Die Ergebnisse stehen im Reiter 3.")
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(work, tail=40))
    if not errs or all(e == "Ordnungen" for e in errs):
        st.subheader("Ausbreitungsfähige Ordnungen vor der Rechnung")
        lam = lams[0] if len(lams) == 1 else float(st.select_slider("Wellenlänge (nm)", options=lams, value=lams[0], key="xg_order_lam"))
        po = fx.propagating_orders(m, lam)
        st.markdown(f"**λ = {lam:g} nm:** {len(po['R'])} Ordnung(en) in Reflexion, {len(po['T'])} in Transmission ausbreitungsfähig "
                    "(Gittergleichung, ohne Rechnung).")
        rows = [{"m": r["m"], "n": r["n"], "Reflexion": "ja" if r["reflektiert"] else "–", "Transmission": "ja" if r["transmittiert"] else "–"}
                for r in fx.orders_table(m, lam)]
        ui.table_show(pd.DataFrame(rows))


def _running():
    proc, folder = S.xg_proc, Path(S.xg_dir)
    log = fr.read_log(folder)
    st.subheader("Rechnung läuft …")
    if proc.poll() is None:
        frac, label = fr.progress_info(log)
        st.progress(frac, text=label)
        st.code("\n".join(log.splitlines()[-10:]) or "(noch keine Ausgabe)")
        if S.get("xg_cancel"):
            st.info("Abbruch angefordert: die Rechnung hält nach der laufenden Lösung an.")
        elif st.button("Abbrechen", key="xg_cancel_btn"):
            fr.request_cancel(folder)
            S.xg_cancel = True
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        S.xg_proc = None
        ok = proc.returncode == 0 and not flags["errors"]
        S.xg_msg = ("ok", "Abgebrochen." if flags["cancelled"] else "Fertig.") if ok else ("error", f"Beendet mit Code {proc.returncode}: " + "; ".join(flags["errors"][:3]))
        st.rerun()


def _field_tab(work, pts, labels, meta):
    withf = [i for i, p in enumerate(pts) if p.get("field_file") and (work / p["field_file"]).is_file()]
    for p in pts:
        if p.get("field_error"):
            st.error(f"Nahfeld bei {p['lam_nm']:g} nm: {p['field_error']}")
    if not withf:
        st.info("Kein Nahfeld in dieser Rechnung: im Reiter 2 „Nahfeld |E|² auf Schnittebenen speichern“ ankreuzen und neu rechnen.")
        return
    i = withf[int(st.selectbox("Wellenlänge", range(len(withf)), format_func=lambda j: labels[withf[j]], key="xg_fld_lam"))] if len(withf) > 1 else withf[0]
    model = S.get("xg_run_model") or S.xg_model
    npz = np.load(work / pts[i]["field_file"])
    title = f"|E|²/|E₀|², λ = {pts[i]['lam_nm']:g} nm, Polarisation {meta['pol']}, θ = {meta['theta']:g}°"
    names = {"xy": f"xy bei z = {float(npz['z_nm']):g} nm", "xz": f"xz bei y = {float(npz['y_nm']):g} nm", "yz": f"yz bei x = {float(npz['x_nm']):g} nm"}
    c = st.columns(3)
    for col, key in zip(c, ("xy", "xz", "yz")):
        with col:
            st.markdown(f"**{names[key]}**")
            ui.show_fig(fx.fig_field(npz, key, model), f"nahfeld_{key}.png", f"xg_dl_fld_{key}")
    st.caption(title + ". Gesamtfeld (einfallendes plus gestreutes) für eine einfallende Welle mit |E₀| = 1; weiße gestrichelte Linien: Schichtgrenzen bzw. Formen. "
               "z = 0 ist die Oberkante der Schichten. Nur bei der Hauptrechnung (nicht in der Konvergenzstudie) gespeichert. Außerhalb des Netzes (weit über/unter dem Gitter) bleibt die Ebene leer.")


def _tab_results(work):
    res_file = work / "results.json"
    if not res_file.is_file():
        st.info("Noch keine Ergebnisse: im Reiter 2 rechnen.")
        return
    res = json.loads(res_file.read_text(encoding="utf-8"))
    pts = [p for p in res["points"] if "error" not in p]
    for p in res["points"] + res.get("convergence", []):
        if "error" in p:
            st.error(f"λ = {p['lam_nm']:g} nm, p = {p.get('order', '?')}: {p['error']}")
    meta = res["meta"]
    st.caption(f"{meta.get('name', '')} · {len(pts)} Wellenlänge(n), p = {meta.get('order')}, θ = {meta['theta']:g}°, φ = {meta['phi']:g}°, "
               f"Polarisation {meta['pol']} · hpfem {meta.get('hpfem', '?')} · Start {meta.get('started', '')}")
    if not pts:
        return
    sub = st.tabs(["Ordnungen", "Spektrum", "Konvergenz", "s und p", "Vergleich mit FourierGrating", "Nahfeld"])
    labels = [f"{p['lam_nm']:g} nm" for p in pts]
    ref_up = None
    with sub[4]:
        ref_up = st.file_uploader("efficiencies.csv von FourierGrating", type=["csv"], key="xg_ref_upload",
                                  help="Spalten lambda_um, phi_deg, psi_deg, delta_deg, side, n, m, efficiency. δ = 0° gilt als s, δ = 90° als p.")
    with sub[0]:
        k = int(st.selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: labels[i], key="xg_res_lam")) if len(pts) > 1 else 0
        df = fx.orders_dataframe(pts[k])
        ui.table_show(df)
        st.markdown(fx.balance_text(pts[k]))
        st.caption(f"{pts[k]['dofs']} Unbekannte, {pts[k]['cells']} Zellen, {pts[k]['time_s']:.1f} s. R: im Einfallsmedium reflektiert, T: im Substrat transmittiert; "
                   "nur ausbreitungsfähige Ordnungen. Ordnung (m, n): m längs x, n längs y.")
        if pts[k].get("balance") is not None and abs(pts[k]["balance"]) > 0.01:
            st.warning(f"Die Energiebilanz weicht um {pts[k]['balance']:+.3f} ab: p oder die Netzdichte erhöhen (Reiter 2).")
        st.download_button("Tabelle (CSV)", ui.csv_bytes(df), file_name=f"ordnungen_{pts[k]['lam_nm']:g}nm.csv", mime="text/csv", key="xg_dl_orders")
    with sub[1]:
        sp = fx.spectrum_dataframe(pts)
        if len(pts) > 1:
            ui.show_fig(fx.fig_spectrum(pts), "kreuzgitter_spektrum.png", "xg_dl_spec")
        ui.table_show(sp)
        st.download_button("Spektrum (CSV)", ui.csv_bytes(sp), file_name="kreuzgitter_spektrum.csv", mime="text/csv", key="xg_dl_spec_csv")
    ref = None
    if ref_up is not None:
        try:
            ref = fx.read_fg_csv(ref_up.getvalue().decode("utf-8"), pts[0]["lam_nm"], meta["pol"], meta["theta"], meta["phi"])
        except ValueError as exc:
            with sub[4]:
                st.error(str(exc))
    with sub[2]:
        conv = [c for c in res.get("convergence", []) if "error" not in c]
        if not conv:
            st.info("Keine Konvergenzstudie in dieser Rechnung (Reiter 2: „Konvergenzstudie in p“).")
        else:
            cdf = fx.convergence_dataframe(conv, ref)
            if len(conv) > 1:
                ui.show_fig(fx.fig_convergence(conv, ref), "kreuzgitter_konvergenz.png", "xg_dl_conv")
            ui.table_show(cdf)
            st.caption(f"Erste Wellenlänge ({pts[0]['lam_nm']:g} nm), Elemente je Wellenlänge "
                       f"{'automatisch (8 / p)' if not meta.get('elements_per_wavelength') else meta['elements_per_wavelength']}. "
                       "Mit der Datei von FourierGrating (Reiter „Vergleich“) kommt die Abweichung gegen die RCWA dazu.")
    with sub[3]:
        k = int(st.selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: labels[i], key="xg_pol_lam")) if len(pts) > 1 else 0
        dfs, dfp = fx.orders_dataframe(pts[k], "s"), fx.orders_dataframe(pts[k], "p")
        c = st.columns(2)
        with c[0]:
            st.markdown("**s**")
            ui.table_show(dfs)
            st.markdown(fx.balance_text(pts[k], "s"))
        with c[1]:
            st.markdown("**p**")
            ui.table_show(dfp)
            st.markdown(fx.balance_text(pts[k], "p"))
        d = fx.pol_difference(pts[k])
        st.markdown(f"Abweichung der Summen: |ΔΣR| = **{d['sum_R']:.2e}**, |ΔΣT| = **{d['sum_T']:.2e}**. Größte Abweichung je Ordnung bei vertauschtem (m, n) ↔ (n, m): "
                    f"**{d['swapped']:.2e}**; ohne Vertauschen: {d['plain']:.2e}.")
        st.caption("Beide Polarisationen stammen aus derselben Lösung. Bei senkrechtem Einfall und vierzähliger Symmetrie der Zelle (Kreis in der Mitte, p_x = p_y) "
                   "sind die Summen gleich, und s bei (m, n) gleicht p bei (n, m) (E längs −y bzw. +x): die Ordnungen (1, 0) und (0, 1) tauschen die Rollen. "
                   "Die Abweichung ist der Diskretisierungsfehler (sinkt mit p).")
    with sub[5]:
        _field_tab(work, pts, labels, meta)
    with sub[4]:
        if ref is None:
            st.info("Datei laden, um die Ordnungen mit der RCWA zu vergleichen." if ref_up is None else "")
        else:
            cdf = fx.comparison_dataframe(pts[0], ref)
            ui.table_show(cdf)
            d = cdf["Δ = FEM − RCWA"].abs()
            d_t0 = d[(cdf["Seite"] == "T") & (cdf["m"] == 0) & (cdf["n"] == 0)].max()
            st.markdown(f"Größte Abweichung: **{d.max():.2e}**, T₀: {d_t0:.2e}. "
                        f"(Erste Wellenlänge, {pts[0]['lam_nm']:g} nm; Ordnungen von FourierGrating (n, m) → (m, n) = (n, −m).)")
            st.download_button("Vergleich (CSV)", ui.csv_bytes(cdf), file_name="vergleich_fourier_grating.csv", mime="text/csv", key="xg_dl_cmp")
    st.download_button("Modell der Rechnung (JSON)", json.dumps(S.get("xg_run_model", S.xg_model), indent=1, ensure_ascii=False).encode("utf-8"),
                       file_name="kreuzgitter_modell.json", mime="application/json", key="xg_dl_model")
