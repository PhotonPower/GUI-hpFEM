"""Waveguides and ports in the FEM model builder (Streamlit), hp-FEM 0.5. Called from fem_app.py: ``render_waveguide(ctx)``.

Task "ports": a device in the chip plane between modal ports (hpfem.WaveguidePort, hpfem.s_parameters): S-matrix, transmission, reflection,
loss, port modes, field of one excitation. Task "modes": effective indices and fields of the guided modes of a cross-section
(hpfem.PropagatingMode). Models, meshing and figures: fem_wg.py; worker: fem_wg_worker.py.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

import fem_materials as fm
import fem_run as fr
import fem_ui as ui
import fem_wg as fw

S = st.session_state

INFO = """
**Aufgaben.**
* **Ports** rechnet ein Bauteil in der Chipebene (Effektivindex-Modell, Polarisation H_z, Feld in der Ebene) zwischen zwei modalen Ports
  (links bei x = 0, rechts bei x = L). Die Portmoden sind die geführten Moden des Querschnitts der Portlinie (1D-p-FEM der Bibliothek). Die
  S-Matrix ist leistungsnormiert: |S_ij|² ist der Leistungsanteil von Kanal j nach Kanal i, die Summe jeder Spalte ist 1 − Verlust.
* **Moden** rechnet die geführten Moden eines geraden Wellenleiters aus seinem Querschnitt (vektorielles Eigenproblem, keine
  Geistermoden): effektiver Index n_eff und die Felder E_t und E_z.

**Konventionen.** Zeitfaktor exp(−iωt). Die Wände oben und unten (y = ±B/2) sind ideale Leiter (PEC); sie müssen mehrere Abklinglängen
von den Wellenleitern entfernt sein. Die Portmoden sind für verlustfreie Schnitte gedacht. Ein Mode, der am Port nicht ausbreitungsfähig
ist, taucht in der S-Matrix nicht auf. Die Wellenlänge ist die im Vakuum; die Indizes sind die des Effektivindex-Modells (z. B. 2,4 für
eine 220 nm dicke Silizium-Schicht in Quarz bei 1550 nm).

**Genauigkeit.** Die Ränder der Formen sind gerade Elemente. Merkmale unter etwa einem Drittel der Zellgröße werden nicht aufgelöst
(Zähne von Gittern, kleine Spalte): „Zellen je Wellenlänge“ und den Faktor an Grenzflächen verkleinern. Als Probe: ein verlustfreies
Bauteil muss Σ|S|² = 1 liefern (Verlust ≈ 0).
"""


def _init():
    if "wg_model" in S:
        return
    S.wg_model = copy.deepcopy(fw.default_model())
    S.wg_ver, S.wg_proc, S.wg_dir, S.wg_msg = 0, None, None, None


def _bump():
    S.wg_ver += 1


def _rename(model, old, new):
    model["materials"] = {(new if k == old else k): v for k, v in model["materials"].items()}
    if model["background"] == old:
        model["background"] = new
    for s in model["shapes"]:
        if s["material"] == old:
            s["material"] = new


def render_waveguide(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    work = Path(ctx["work"]) / "waveguide"
    st.title("FEM-Modellwerkstatt: Wellenleiter und Ports")
    st.caption("Bauteile in der Chipebene zwischen modalen Ports (S-Matrix, Transmission, Reflexion, Felder) und Moden des Querschnitts "
               "(effektive Indizes, Modenfelder). hp-FEM: WaveguidePort, s_parameters, PropagatingMode.")
    if S.wg_proc is not None:
        return _running()
    if not (ctx["FEATS"].get("ports") or ctx["lib"].get("error")):
        st.warning("Das eingestellte Python hat kein hpfem mit Ports und PropagatingMode. Anzeigen geht trotzdem.")
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
def _num(col, label, key, value, step=10.0, lo=-1e6, hi=1e6, help=None):
    return float(col.number_input(label, lo, hi, float(value), step, key=key, format="%.6g", help=help))


def _shape_editor(m, i, s, names, ver, bump=None, prefix="wg_s"):
    """Editor of shape i; ``bump`` (default: this module's) is called before every st.rerun, ``prefix`` keeps the widget keys apart (fem_time_app)."""
    k = f"{prefix}_{ver}_{i}"
    bump = bump or _bump
    head = st.columns([2, 3, 1, 1, 1])
    s["type"] = head[0].selectbox("Form", list(fw.SHAPES), index=list(fw.SHAPES).index(s["type"]), format_func=fw.SHAPES.get, key=f"{k}_t",
                                  label_visibility="collapsed")
    s["material"] = head[1].selectbox("Material", names, index=names.index(s["material"]) if s["material"] in names else 0, key=f"{k}_m",
                                      label_visibility="collapsed")
    if head[2].button("↑", key=f"{k}_up", disabled=i == 0, help="Priorität erhöhen (spätere Formen überdecken frühere)"):
        m["shapes"][i - 1], m["shapes"][i] = m["shapes"][i], m["shapes"][i - 1]
        bump()
        st.rerun()
    if head[3].button("↓", key=f"{k}_dn", disabled=i == len(m["shapes"]) - 1, help="Priorität verringern"):
        m["shapes"][i + 1], m["shapes"][i] = m["shapes"][i], m["shapes"][i + 1]
        bump()
        st.rerun()
    if head[4].button("✕", key=f"{k}_del", help="Form entfernen"):
        m["shapes"].pop(i)
        bump()
        st.rerun()
    c = st.columns(4)
    t = s["type"]
    if t == "strip":
        s.setdefault("x0", -100.0), s.setdefault("x1", m["length_nm"] + 100.0), s.setdefault("yc", 0.0), s.setdefault("w", 450.0)
        s["x0"], s["x1"] = _num(c[0], "x von (nm)", f"{k}_x0", s["x0"]), _num(c[1], "x bis (nm)", f"{k}_x1", s["x1"])
        s["yc"], s["w"] = _num(c[2], "Mitte y (nm)", f"{k}_yc", s["yc"]), _num(c[3], "Breite (nm)", f"{k}_w", s["w"], 10.0, 1.0)
    elif t == "rect":
        for key_, default in (("x0", 0.0), ("x1", 500.0), ("y0", -100.0), ("y1", 100.0)):
            s.setdefault(key_, default)
        s["x0"], s["x1"] = _num(c[0], "x von (nm)", f"{k}_x0", s["x0"]), _num(c[1], "x bis (nm)", f"{k}_x1", s["x1"])
        s["y0"], s["y1"] = _num(c[2], "y von (nm)", f"{k}_y0", s["y0"]), _num(c[3], "y bis (nm)", f"{k}_y1", s["y1"])
    elif t == "disc":
        s.setdefault("cx", m["length_nm"] / 2), s.setdefault("cy", 0.0), s.setdefault("r", 300.0)
        s["cx"], s["cy"] = _num(c[0], "Mitte x (nm)", f"{k}_cx", s["cx"]), _num(c[1], "Mitte y (nm)", f"{k}_cy", s["cy"])
        s["r"] = _num(c[2], "Radius (nm)", f"{k}_r", s["r"], 10.0, 1.0)
    elif t == "ring":
        s.setdefault("cx", m["length_nm"] / 2), s.setdefault("cy", 0.0), s.setdefault("r", 2500.0), s.setdefault("w", 450.0)
        s["cx"], s["cy"] = _num(c[0], "Mitte x (nm)", f"{k}_cx", s["cx"]), _num(c[1], "Mitte y (nm)", f"{k}_cy", s["cy"])
        s["r"] = _num(c[2], "Radius der Mittellinie (nm)", f"{k}_r", s["r"], 10.0, 1.0)
        s["w"] = _num(c[3], "Breite (nm)", f"{k}_w", s["w"], 10.0, 1.0)
    else:
        s.setdefault("points", [[0.0, -100.0], [500.0, -100.0], [250.0, 100.0]])
        text = "\n".join(f"{p[0]:g}, {p[1]:g}" for p in s["points"])
        new = st.text_area("Eckpunkte (x, y in nm, eine Zeile je Punkt)", value=text, key=f"{k}_pts", height=100)
        try:
            pts = [[float(v) for v in ln.replace(";", ",").split(",")[:2]] for ln in new.splitlines() if ln.strip()]
            if all(len(p) == 2 for p in pts):
                s["points"] = pts
        except ValueError:
            st.error("Die Punkte als „x, y“ eingeben.")


def _tab_model(ctx):
    m, ver = S.wg_model, S.wg_ver
    left, right = st.columns([6, 5])
    with left:
        presets = fw.presets()
        c = st.columns([3, 1])
        pick = c[0].selectbox("Vorlage", ["– auswählen –"] + list(presets), key="wg_preset")
        if c[1].button("Laden", key="wg_preset_load", disabled=pick.startswith("–")):
            S.wg_model = copy.deepcopy(presets[pick])
            _bump()
            st.rerun()
        m["name"] = st.text_input("Name", value=m["name"], key=f"wg_name_{ver}")
        m["task"] = st.radio("Aufgabe", list(fw.TASKS), index=list(fw.TASKS).index(m["task"]), format_func=fw.TASKS.get, key=f"wg_task_{ver}",
                             horizontal=True)
        st.subheader("Materialien")
        lam0 = fw.sweep_list(m["sweep"])[0]
        ui.material_editor(m, ver, lam0, fw.used_materials(m), _rename, _bump, prefix="wg_",
                           fit=None if ctx["lib"].get("error") else (lambda sp, lo, hi, n: fr.fit_material(ctx["py"], ctx["repo"], sp, lo, hi, n)))
        names = list(m["materials"])
        st.subheader("Gebiet")
        c = st.columns(3)
        m["background"] = c[0].selectbox("Hintergrund (Mantel)", names, index=names.index(m["background"]) if m["background"] in names else 0,
                                         key=f"wg_bg_{ver}")
        if m["task"] == "ports":
            m["length_nm"] = _num(c[1], "Länge L (nm), Port bei x = 0 und x = L", f"wg_L_{ver}", m["length_nm"], 100.0, 100.0)
            m["width_nm"] = _num(c[2], "Breite B (nm), PEC bei y = ±B/2", f"wg_W_{ver}", m["width_nm"], 100.0, 100.0)
        else:
            m["length_nm"] = _num(c[1], "Breite des Querschnitts (nm, x)", f"wg_L_{ver}", m["length_nm"], 100.0, 100.0)
            m["width_nm"] = _num(c[2], "Höhe des Querschnitts (nm, y)", f"wg_W_{ver}", m["width_nm"], 100.0, 100.0)
        st.subheader("Formen (spätere überdecken frühere)")
        for i, s in enumerate(list(m["shapes"])):
            with st.container(border=True):
                _shape_editor(m, i, s, names, ver)
        c = st.columns([2, 1])
        new_type = c[0].selectbox("Neue Form", list(fw.SHAPES), format_func=fw.SHAPES.get, key=f"wg_newtype_{ver}", label_visibility="collapsed")
        if c[1].button("Form hinzufügen", key=f"wg_add_{ver}"):
            m["shapes"].append({"type": new_type, "material": names[-1] if names else ""})
            _bump()
            st.rerun()
        if m["task"] == "ports":
            st.subheader("Ports")
            c = st.columns(4)
            for col, side, label in ((c[0], "left", "links"), (c[2], "right", "rechts")):
                p = m["ports"][side]
                p["enabled"] = col.checkbox(f"Port {label} (x = {'0' if side == 'left' else 'L'})", value=bool(p["enabled"]), key=f"wg_pe_{side}_{ver}")
                nxt = c[1] if side == "left" else c[3]
                p["modes"] = int(nxt.number_input(f"Moden {label}", 1, 12, int(p["modes"]), key=f"wg_pm_{side}_{ver}",
                                                  help="Zahl der Moden der Entwicklung am Port: zuerst die geführten, dann die am stärksten abklingenden. "
                                                       "Mehr Moden als geführte nehmen abklingende Anteile nahe am Port mit."))
        else:
            m["modes"]["num"] = int(st.number_input("Zahl der gesuchten Moden", 1, 20, int(m["modes"]["num"]), key=f"wg_nm_{ver}"))
    with right:
        for lvl, text in fw.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        try:
            ui.show_fig(fw.fig_preview(m), "wellenleiter.png", f"wg_prev_dl_{ver}")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")
        rows = []
        for n_ in sorted(fw.used_materials(m)):
            try:
                nn, kk = fm.nk(m["materials"][n_], fw.sweep_list(m["sweep"])[0])
                rows.append({"Material": n_, "n": round(float(nn), 4), "k": round(float(kk), 4)})
            except Exception as exc:
                rows.append({"Material": n_, "n": None, "k": str(exc)[:50]})
        ui.table_show(rows)


# ===================================================================================================================== run tab
def _tab_run(ctx, work):
    m, ver = S.wg_model, S.wg_ver
    st.subheader("Wellenlänge")
    sw = m["sweep"]
    sw["mode"] = st.radio("Wellenlänge", ["single", "wavelength"], index=["single", "wavelength"].index(sw["mode"]), horizontal=True, key=f"wg_swm_{ver}",
                          format_func={"single": "ein Wert", "wavelength": "Durchlauf"}.get)
    c = st.columns(3)
    sw["start"] = _num(c[0], "λ (nm)" if sw["mode"] == "single" else "λ von (nm)", f"wg_swa_{ver}", sw["start"], 10.0, 100.0, 20000.0)
    if sw["mode"] != "single":
        sw["stop"] = _num(c[1], "λ bis (nm)", f"wg_swb_{ver}", sw["stop"], 10.0, 100.0, 20000.0)
        sw["n"] = int(c[2].number_input("Punkte", 2, 400, int(sw["n"]), key=f"wg_swn_{ver}"))
    st.subheader("Netz und Ordnung")
    c = st.columns(3)
    me = m["mesh"]
    me["cells_per_wavelength"] = _num(c[0], "Zellen je Wellenlänge im Material", f"wg_cpw_{ver}", me["cells_per_wavelength"], 0.5, 1.0, 40.0,
                                      "Zellgröße = λ / (Zellen · n). Bei p = 3 genügen 4 bis 6 für glatte Strukturen.")
    me["order"] = int(c[1].number_input("Ordnung p", 1, 6, int(me["order"]), key=f"wg_p_{ver}"))
    me["interface_factor"] = _num(c[2], "Faktor an Grenzflächen", f"wg_if_{ver}", me.get("interface_factor", 0.8), 0.1, 0.2, 2.0,
                                  "Kleiner als 1 verfeinert die Zellen entlang der Materialgrenzen (Zähne, schmale Spalte).")
    if m["task"] == "ports":
        st.subheader("Anregung und Felder")
        sides = [s for s in ("left", "right") if m["ports"][s]["enabled"]] or ["left"]
        c = st.columns(3)
        ex = m["excite"]
        ex["port"] = int(c[0].selectbox("Anregung am Port", range(len(sides)), index=min(int(ex["port"]), len(sides) - 1), key=f"wg_exp_{ver}",
                                        format_func=lambda i: {"left": "links", "right": "rechts"}[sides[i]],
                                        help="Die S-Matrix enthält alle Kanäle. Transmission, Reflexion und das Feld gelten für diesen Kanal."))
        ex["mode"] = int(c[1].number_input("Mode", 0, 11, int(ex["mode"]), key=f"wg_exm_{ver}"))
        m["field"] = c[2].selectbox("Feldkarten", list(fw.FIELDS), index=list(fw.FIELDS).index(m["field"]), format_func=fw.FIELDS.get, key=f"wg_fld_{ver}",
                                    help="Jede Feldkarte kostet eine weitere Lösung je Wellenlänge.")
    errs = [t for lvl, t in fw.validate(m) if lvl == "error"]
    for e in errs:
        st.error(e)
    n_l = len(fw.sweep_list(sw))
    st.markdown(f"**{n_l} Wellenlänge(n).** Etwa 3 bis 10 s je Wellenlänge für das Streifenbeispiel, bei feinen Netzen (Ring, Querschnitt) deutlich länger.")
    if st.button("Netz erzeugen und rechnen", type="primary", key="wg_run", disabled=bool(errs)):
        try:
            work.mkdir(parents=True, exist_ok=True)
            with st.spinner("Netz wird erzeugt …"):
                stats = fw.build_mesh(m, min(fw.sweep_list(sw)), work)
            (work / "job.json").write_text(json.dumps(fw.job(m), ensure_ascii=False), encoding="utf-8")
            for f in list(work.glob("field_*.npz")) + list(work.glob("mode_*.npz")):
                f.unlink()
            S.wg_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script="fem_wg_worker.py")
            S.wg_dir, S.wg_cancel = str(work), False
            S.wg_stats = stats
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("wg_msg"):
        (st.success if S.wg_msg[0] == "ok" else st.error)(S.wg_msg[1] + " Die Ergebnisse stehen im Reiter 3.")
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(work, tail=40))


def _running():
    proc, folder = S.wg_proc, Path(S.wg_dir)
    log = fr.read_log(folder)
    st.subheader("Rechnung läuft …")
    if proc.poll() is None:
        frac, label = fr.progress_info(log)
        st.progress(frac, text=label)
        st.code("\n".join(log.splitlines()[-10:]) or "(noch keine Ausgabe)")
        if S.get("wg_cancel"):
            st.info("Abbruch angefordert: die Rechnung hält nach der laufenden Wellenlänge an.")
        elif st.button("Abbrechen", key="wg_cancel_btn"):
            fr.request_cancel(folder)
            S.wg_cancel = True
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        S.wg_proc = None
        ok = proc.returncode == 0 and not flags["errors"]
        S.wg_msg = ("ok", "Abgebrochen." if flags["cancelled"] else "Fertig.") if ok else ("error", f"Beendet mit Code {proc.returncode}: " + "; ".join(flags["errors"][:3]))
        st.rerun()


# ===================================================================================================================== results
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
    meta = res["meta"]
    st.caption(f"{meta.get('name', '')} · {meta.get('task')} · {len(pts)} Wellenlänge(n) · {meta.get('cells')} Zellen, p = {meta.get('order')} · "
               f"hpfem {meta.get('hpfem', '?')} · Start {meta.get('started', '')}")
    if meta.get("task") == "ports":
        _results_ports(work, pts)
    else:
        _results_modes(work, pts)
    st.download_button("Modell (JSON)", json.dumps(S.wg_model, indent=1, ensure_ascii=False).encode("utf-8"), file_name="wellenleiter_modell.json",
                       mime="application/json", key="wg_dl_model")


def _results_ports(work, pts):
    sub = st.tabs(["Spektrum", "S-Matrix", "Portmoden", "Feld"])
    labels = [f"{p['lam_nm']:g} nm" for p in pts]
    with sub[0]:
        ex = pts[0]["excitation"]
        st.caption(f"Anregung: Port {fw.channel_label(ex)}. Reflexion und Transmission summieren über alle Moden des jeweiligen Ports; "
                   "Verlust = 1 − R − T (Abstrahlung in die PML-freien Wände ist ausgeschlossen, das Gebiet ist PEC-berandet, Verlust kommt nur "
                   "vom Material).")
        ui.show_fig(fw.fig_spectrum(pts), "spektrum.png", "wg_dl_spec")
        df = fw.ports_dataframe(pts)
        ui.table_show(df)
        st.download_button("Tabelle (CSV)", ui.csv_bytes(df), file_name="wellenleiter_spektrum.csv", mime="text/csv", key="wg_dl_csv")
        worst = max(abs(p["loss"]) for p in pts)
        if worst > 0.02 and not any(fm.nk(S.wg_model["materials"][n], pts[0]["lam_nm"])[1] > 0 for n in fw.used_materials(S.wg_model)
                                    if n in S.wg_model["materials"]):
            st.warning(f"Das Bauteil ist verlustfrei, die Bilanz weicht aber bis {worst:.3f} von 1 ab: Netz verfeinern (Zellen je Wellenlänge, Faktor "
                       "an Grenzflächen) oder die Ordnung erhöhen.")
    with sub[1]:
        k = st.selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: labels[i], key="wg_s_lam")
        kind = st.radio("Anzeige", ["abs2", "phase"], horizontal=True, key="wg_s_kind", format_func={"abs2": "|S|² (Leistung)", "phase": "Phase (Grad)"}.get)
        df = fw.s_matrix_frame(pts[k], kind)
        ui.table_show(df.round(5))
        st.download_button("S-Matrix (CSV)", ui.csv_bytes(df.reset_index()), file_name=f"s_matrix_{pts[k]['lam_nm']:g}nm.csv", mime="text/csv", key="wg_dl_s")
        st.caption("Spalte = Anregung, Zeile = Antwort; die Kanäle sind die ausbreitungsfähigen Moden der Ports. Die Matrix ist reziprok (S = Sᵀ).")
    with sub[2]:
        ui.show_fig(fw.fig_neff(pts), "neff.png", "wg_dl_neff")
        k = st.selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: labels[i], key="wg_pm_lam")
        rows = []
        for side, name in (("left", "links"), ("right", "rechts")):
            for j, mo in enumerate(pts[k]["port_modes"].get(side, [])):
                rows.append({"Port": name, "Mode": j, "n_eff": round(mo["n_eff"], 5), "ausbreitungsfähig": "ja" if mo["propagating"] else "nein"})
        ui.table_show(rows)
    with sub[3]:
        withf = [(i, p) for i, p in enumerate(pts) if p.get("field")]
        if not withf:
            st.info("Keine Feldkarte: im Reiter 2 „Feldkarten“ einschalten.")
        else:
            c = st.columns(2)
            k = c[0].selectbox("Wellenlänge", range(len(withf)), format_func=lambda i: labels[withf[i][0]], key="wg_f_lam")
            q = c[1].selectbox("Größe", ["abs", "re_x", "re_y"], key="wg_f_q", format_func={"abs": "|E|", "re_x": "Re E_x", "re_y": "Re E_y"}.get)
            npz = np.load(work / withf[k][1]["field"])
            ex = withf[k][1]["excitation"]
            ui.show_fig(fw.fig_field(npz, q, f"λ = {withf[k][1]['lam_nm']:g} nm, Anregung {fw.channel_label(ex)}", S.wg_model), "feld.png", "wg_dl_field")
            st.caption("Auf das Maximum normiert; die weißen Linien sind die Formen des aktuellen Modells (bei geänderter Geometrie neu rechnen).")


def _results_modes(work, pts):
    sub = st.tabs(["Effektive Indizes", "Modenfelder"])
    labels = [f"{p['lam_nm']:g} nm" for p in pts]
    with sub[0]:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.6))
        for k in range(max(len(p["n_eff"]) for p in pts)):
            xs = [p["lam_nm"] for p in pts if len(p["n_eff"]) > k]
            ax.plot(xs, [p["n_eff"][k] for p in pts if len(p["n_eff"]) > k], "o-", ms=4, label=f"Mode {k}")
        ax.set_xlabel("Wellenlänge (nm)")
        ax.set_ylabel("n_eff")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        ui.show_fig(fig, "neff_moden.png", "wg_dl_nm")
        k = st.selectbox("Wellenlänge", range(len(pts)), format_func=lambda i: labels[i], key="wg_nm_lam")
        rows = [{"Mode": j, "n_eff": round(v, 6), "β (1/µm)": round(pts[k]["beta"][j] * 1e-6, 5)} for j, v in enumerate(pts[k]["n_eff"])]
        ui.table_show(rows)
        st.download_button("Tabelle (CSV)", ui.csv_bytes(__import__("pandas").DataFrame(rows)), file_name=f"moden_{pts[k]['lam_nm']:g}nm.csv", mime="text/csv",
                           key="wg_dl_mcsv")
        st.caption("Moden nach fallendem n_eff sortiert. n_eff liegt zwischen dem Index des Mantels und dem größten Index des Querschnitts.")
    with sub[1]:
        withf = [p for p in pts if p.get("modes_file")]
        if not withf:
            st.info("Keine Modenfelder vorhanden.")
            return
        c = st.columns(3)
        i = c[0].selectbox("Wellenlänge", range(len(withf)), format_func=lambda i: f"{withf[i]['lam_nm']:g} nm", key="wg_mf_lam")
        j = c[1].selectbox("Mode", range(len(withf[i]["n_eff"])), format_func=lambda j: f"Mode {j} (n_eff = {withf[i]['n_eff'][j]:.4f})", key="wg_mf_k")
        q = c[2].selectbox("Größe", ["abs_t", "abs_z", "ex", "ey"], key="wg_mf_q",
                           format_func={"abs_t": "|E_t| (quer)", "abs_z": "|E_z| (längs)", "ex": "Re E_x", "ey": "Re E_y"}.get)
        npz = np.load(work / withf[i]["modes_file"])
        ui.show_fig(fw.fig_mode(npz, j, q, f"Mode {j}, n_eff = {withf[i]['n_eff'][j]:.4f}, λ = {withf[i]['lam_nm']:g} nm", S.wg_model), "mode.png", "wg_dl_mode")
