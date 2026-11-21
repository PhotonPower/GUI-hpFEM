"""Band structures of photonic crystals in the FEM model builder (Streamlit), hp-FEM 0.5. Called from fem_app.py: ``render_bands(ctx)``.

hpfem.BandStructure2D (Floquet-Bloch, in-plane E field, lossless): one unit cell (square or hexagonal lattice, scale free in units of the lattice
constant), the path through the Brillouin zone, the bands f = omega a / (2 pi c), gaps and band fields. Models, meshing and figures: fem_bands.py;
worker: fem_bands_worker.py.
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
import fem_ui as ui

S = st.session_state

INFO = """
**Was gerechnet wird.** Die Bänder f = ω a / (2π c) = a / λ eines zweidimensionalen photonischen Kristalls entlang des Pfads durch die irreduzible
Brillouin-Zone (Floquet-Bloch-Eigenproblem der Einheitszelle, `hpfem.BandStructure2D`, ohne Geistermoden). Die Struktur ist in z unendlich ausgedehnt,
der Wellenvektor liegt in der Ebene. Es gibt deshalb **keinen Lichtkegel** (kein Substrat, keine Abstrahlung): alle Moden sind gebunden.

**Polarisation.** Der Löser rechnet das elektrische Feld in der Ebene, E = (E_x, E_y), mit H_z aus der Ebene (in Joannopoulos et al. „TE“). Die
Polarisation mit E_z („TM“) deckt dieser Löser nicht ab. Das bekannte TM-Gap quadratischer Stabgitter erscheint hier also nicht; für die
Polarisation mit dem E-Feld in der Ebene öffnen Lochgitter (Luftlöcher in einem Dielektrikum) Lücken.

**Maßstabsfrei.** Alle Längen sind in Einheiten der Gitterkonstante a, die Frequenz ist f = a/λ, der Wellenvektor in Einheiten 2π/a. Formen werden
relativ zur Zellmitte angegeben und im ganzen Gitter wiederholt; eine Form, die über den Zellrand hinausreicht (großer Radius), setzt sich am
gegenüberliegenden Rand fort. Spätere Formen überdecken frühere. Es gibt nur konstante, reelle Permittivitäten (verlustfrei): ein
Bandstruktur-Problem hat keine Wellenlänge, bei der man eine Dispersion auswerten könnte.

**Pfade.** Quadratisch: Γ–X–M–Γ. Hexagonal (Dreiecksgitter, a₁ = (1, 0), a₂ = (½, √3/2)): Γ–M–K–Γ. Lücken werden zwischen benachbarten Bändern über
den **ganzen Pfad** bestimmt (größtes Minimum des oberen und kleinstes Maximum des unteren Bandes); bei den üblichen Gittern liegen die Extrema auf
dem Pfad, eine Lücke außerhalb der Hochsymmetriepunkte ist damit aber nicht ausgeschlossen. „Δf/Mitte“ ist die Lückenbreite relativ zur Mittenfrequenz.

**Das Band bei f = 0.** Das niedrigste Band ist das lineare („akustische“) Band; bei Γ liefert der Löser eine zweite Null (konstantes Feld, kein
Gradient einer periodischen Funktion), sie wird entfernt, damit der Bandindex längs des Pfads gleich bleibt.

**Genauigkeit.** Höhere Bänder brauchen ein feineres Netz und höhere Ordnung. „Größte Frequenz“ legt die Zellgröße fest (Zellgröße ≈ a / (f_max · Zellen ·
n)); Bänder oberhalb von f_max sind schlecht aufgelöst. Probe: das leere Gitter muss die Bänder |k + G| treffen (graue Linien im Diagramm). Die Ableitungen
der Bänder (Gruppengeschwindigkeit, Empfindlichkeit gegen ε und Form: `hpfem.group_velocity`, `band_permittivity_derivative`, `band_shape_derivative`) hat
die Bibliothek; die Oberfläche zeigt sie noch nicht.
"""


def _init():
    if "bd_model" in S:
        return
    S.bd_model = copy.deepcopy(fb.default_model())
    S.bd_ver, S.bd_proc, S.bd_dir, S.bd_msg = 0, None, None, None


def _bump():
    S.bd_ver += 1


def _unique(m, base):
    n, i = base, 2
    while n in m["materials"]:
        n, i = f"{base} {i}", i + 1
    return n


def _rename(m, old, new):
    m["materials"] = {(new if k == old else k): v for k, v in m["materials"].items()}
    if m["background"] == old:
        m["background"] = new
    for s in m["shapes"]:
        if s["material"] == old:
            s["material"] = new


def render_bands(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    work = Path(ctx["work"]) / "bands"
    st.title("FEM-Modellwerkstatt: Bandstruktur")
    st.caption("Bänder photonischer Kristalle (2D, Floquet-Bloch, Polarisation mit E in der Ebene) entlang des Pfads durch die Brillouin-Zone, "
               "Bandlücken und Modenfelder. hp-FEM: BandStructure2D.")
    if S.bd_proc is not None:
        return _running()
    if not (ctx["FEATS"].get("bands") or ctx["lib"].get("error")):
        st.warning("Das eingestellte Python hat kein hpfem mit BandStructure2D. Anzeigen geht trotzdem.")
    t1, t2, t3, t4 = st.tabs(["1 Modell", "2 Rechnung", "3 Ergebnisse", "4 Info"])
    with t1:
        _tab_model()
    with t2:
        _tab_run(ctx, work)
    with t3:
        _tab_results(work)
    with t4:
        st.markdown(INFO)


# ===================================================================================================================== model tab
def _num(col, label, key, value, step=0.05, lo=-100.0, hi=100.0, help=None):
    return float(col.number_input(label, lo, hi, float(value), step, key=key, format="%.5g", help=help))


def _material_editor(m, ver):
    for name in list(m["materials"]):
        spec = m["materials"][name]
        c = st.columns([3, 2, 1])
        new = c[0].text_input("Name", value=name, key=f"bd_mn_{ver}_{name}", label_visibility="collapsed")
        eps = fb.eps_of(spec).real
        m["materials"][name] = {"type": "eps", "re": float(c[1].number_input("ε", 0.01, 400.0, float(eps), 0.5, key=f"bd_me_{ver}_{name}", format="%.5g",
                                                                              label_visibility="collapsed", help="Relative Permittivität ε (n = √ε)")),
                                "im": 0.0}
        if new != name and new.strip():
            if new in m["materials"]:
                c[0].error("Name schon vergeben.")
            else:
                _rename(m, name, new.strip())
                _bump()
                st.rerun()
        if c[2].button("✕", key=f"bd_md_{ver}_{name}", disabled=name in fb.used_materials(m) or len(m["materials"]) <= 1,
                       help="Nur möglich, wenn das Material nirgends verwendet wird."):
            del m["materials"][name]
            _bump()
            st.rerun()
    if st.button("Material hinzufügen", key=f"bd_madd_{ver}"):
        m["materials"][_unique(m, "Material")] = {"type": "eps", "re": 4.0, "im": 0.0}
        _bump()
        st.rerun()


def _shape_editor(m, i, s, names, ver):
    k = f"bd_s_{ver}_{i}"
    head = st.columns([2, 3, 1, 1, 1])
    s["type"] = head[0].selectbox("Form", list(fb.SHAPES), index=list(fb.SHAPES).index(s["type"]) if s["type"] in fb.SHAPES else 0, format_func=fb.SHAPES.get,
                                  key=f"{k}_t", label_visibility="collapsed")
    s["material"] = head[1].selectbox("Material", names, index=names.index(s["material"]) if s["material"] in names else 0, key=f"{k}_m",
                                      label_visibility="collapsed")
    if head[2].button("↑", key=f"{k}_up", disabled=i == 0, help="Priorität erhöhen (spätere Formen überdecken frühere)"):
        m["shapes"][i - 1], m["shapes"][i] = m["shapes"][i], m["shapes"][i - 1]
        _bump()
        st.rerun()
    if head[3].button("↓", key=f"{k}_dn", disabled=i == len(m["shapes"]) - 1, help="Priorität verringern"):
        m["shapes"][i + 1], m["shapes"][i] = m["shapes"][i], m["shapes"][i + 1]
        _bump()
        st.rerun()
    if head[4].button("✕", key=f"bd_{k}_del", help="Form entfernen"):
        m["shapes"].pop(i)
        _bump()
        st.rerun()
    c = st.columns(4)
    t = s["type"]
    if t == "rect":
        for key_, default in (("x0", -0.25), ("x1", 0.25), ("y0", -0.25), ("y1", 0.25)):
            s.setdefault(key_, default)
        s["x0"], s["x1"] = _num(c[0], "x von (a)", f"{k}_x0", s["x0"]), _num(c[1], "x bis (a)", f"{k}_x1", s["x1"])
        s["y0"], s["y1"] = _num(c[2], "y von (a)", f"{k}_y0", s["y0"]), _num(c[3], "y bis (a)", f"{k}_y1", s["y1"])
    elif t == "disc":
        s.setdefault("cx", 0.0), s.setdefault("cy", 0.0), s.setdefault("r", 0.3)
        s["cx"], s["cy"] = _num(c[0], "Mitte x (a)", f"{k}_cx", s["cx"]), _num(c[1], "Mitte y (a)", f"{k}_cy", s["cy"])
        s["r"] = _num(c[2], "Radius (a)", f"{k}_r", s["r"], 0.01, 0.001, 5.0)
    elif t == "ring":
        s.setdefault("cx", 0.0), s.setdefault("cy", 0.0), s.setdefault("r", 0.3), s.setdefault("w", 0.1)
        s["cx"], s["cy"] = _num(c[0], "Mitte x (a)", f"{k}_cx", s["cx"]), _num(c[1], "Mitte y (a)", f"{k}_cy", s["cy"])
        s["r"] = _num(c[2], "Radius der Mittellinie (a)", f"{k}_r", s["r"], 0.01, 0.001, 5.0)
        s["w"] = _num(c[3], "Breite (a)", f"{k}_w", s["w"], 0.01, 0.001, 5.0)
    else:
        s.setdefault("points", [[-0.3, -0.3], [0.3, -0.3], [0.0, 0.3]])
        text = "\n".join(f"{p[0]:g}, {p[1]:g}" for p in s["points"])
        new = st.text_area("Eckpunkte (x, y in a, eine Zeile je Punkt)", value=text, key=f"{k}_pts", height=100)
        try:
            pts = [[float(v) for v in ln.replace(";", ",").split(",")[:2]] for ln in new.splitlines() if ln.strip()]
            if all(len(p) == 2 for p in pts):
                s["points"] = pts
        except ValueError:
            st.error("Die Punkte als „x, y“ eingeben.")


def _tab_model():
    m, ver = S.bd_model, S.bd_ver
    left, right = st.columns([6, 5])
    with left:
        presets = fb.presets()
        c = st.columns([3, 1])
        pick = c[0].selectbox("Vorlage", ["– auswählen –"] + list(presets), key="bd_preset")
        if c[1].button("Laden", key="bd_preset_load", disabled=pick.startswith("–")):
            S.bd_model = copy.deepcopy(presets[pick])
            _bump()
            st.rerun()
        m["name"] = st.text_input("Name", value=m["name"], key=f"bd_name_{ver}")
        m["lattice"] = st.radio("Gitter", list(fb.LATTICES), index=list(fb.LATTICES).index(m["lattice"]) if m["lattice"] in fb.LATTICES else 0,
                                format_func=fb.LATTICES.get, key=f"bd_lat_{ver}", horizontal=True,
                                help="Quadratisch: a₁ = (1, 0), a₂ = (0, 1), Pfad Γ–X–M–Γ. Hexagonal: a₁ = (1, 0), a₂ = (½, √3/2), Pfad Γ–M–K–Γ. Die Zelle ist um den Ursprung zentriert.")
        st.subheader("Materialien (reelles ε)")
        _material_editor(m, ver)
        names = list(m["materials"])
        m["background"] = st.selectbox("Hintergrund", names, index=names.index(m["background"]) if m["background"] in names else 0, key=f"bd_bg_{ver}")
        st.subheader("Formen in der Zelle (relativ zur Zellmitte, in Einheiten a)")
        for i, s in enumerate(list(m["shapes"])):
            with st.container(border=True):
                _shape_editor(m, i, s, names, ver)
        c = st.columns([2, 1])
        new_type = c[0].selectbox("Neue Form", list(fb.SHAPES), format_func=fb.SHAPES.get, key=f"bd_newtype_{ver}", label_visibility="collapsed")
        if c[1].button("Form hinzufügen", key=f"bd_add_{ver}"):
            m["shapes"].append({"type": new_type, "material": names[-1] if names else ""})
            _bump()
            st.rerun()
    with right:
        for lvl, text in fb.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        try:
            ui.show_fig(fb.fig_preview(m), "zelle.png", f"bd_prev_dl_{ver}")
            st.caption("Dick umrandet: die Einheitszelle; blass: ihre Nachbarn. Pfad: " + "–".join(lab for lab, _ in fb.path_corners(m)) + ".")
        except Exception as exc:
            st.error(f"Vorschau nicht möglich: {exc}")


# ===================================================================================================================== run tab
def _tab_run(ctx, work):
    m, ver = S.bd_model, S.bd_ver
    st.subheader("Pfad und Bänder")
    c = st.columns(3)
    m["n_seg"] = int(c[0].number_input("Punkte je Pfadsegment", 1, 40, int(m["n_seg"]), key=f"bd_nseg_{ver}",
                                       help="Zahl der k-Punkte je Segment ohne den Startpunkt; jeder Punkt ist eine eigene Eigenwertrechnung."))
    m["bands"] = int(c[1].number_input("Zahl der Bänder", 1, 20, int(m["bands"]), key=f"bd_nb_{ver}"))
    m["field"] = c[2].selectbox("Modenfelder", list(fb.FIELDS), index=list(fb.FIELDS).index(m["field"]) if m["field"] in fb.FIELDS else 0,
                                format_func=fb.FIELDS.get, key=f"bd_fld_{ver}",
                                help="Die Felder aller Bänder an den Eckpunkten des Pfads (Γ, X, M bzw. Γ, M, K) werden gespeichert.")
    st.subheader("Netz und Ordnung")
    c = st.columns(4)
    me = m["mesh"]
    me["cells_per_wavelength"] = _num(c[0], "Zellen je Wellenlänge im Material", f"bd_cpw_{ver}", me["cells_per_wavelength"], 0.5, 1.0, 40.0,
                                      "Zellgröße = 1 / (f_max · Zellen · n) in Einheiten a.")
    me["order"] = int(c[1].number_input("Ordnung p", 1, 8, int(me["order"]), key=f"bd_p_{ver}"))
    me["f_max"] = _num(c[2], "Größte Frequenz f_max (a/λ)", f"bd_fmax_{ver}", me["f_max"], 0.1, 0.1, 5.0,
                       "Bestimmt die Zellgröße; Bänder oberhalb von f_max sind schlechter aufgelöst.")
    me["interface_factor"] = _num(c[3], "Faktor an Grenzflächen", f"bd_if_{ver}", me.get("interface_factor", 0.8), 0.1, 0.2, 2.0,
                                  "Kleiner als 1 verfeinert die Zellen entlang der Materialgrenzen.")
    m["krylov"] = int(st.number_input("Krylov-Dimension (0 = automatisch: 40 + 4·(Bänder + 1))", 0, 400, int(m.get("krylov", 0)), 10, key=f"bd_kry_{ver}",
                                      help="Die Automatik der Bibliothek kommt bei Γ nicht zur Ruhe; die App setzt deshalb eine feste Dimension."))
    errs = [t for lvl, t in fb.validate(m) if lvl == "error"]
    for e in errs:
        st.error(e)
    n_k = len(fb.k_path(m)[0])
    st.markdown(f"**{n_k} k-Punkte.** Je Punkt eine Faktorisierung und ein Eigenwertproblem: bei etwa 400 Zellen und p = 3 rund 5 s je Punkt, bei feinen Netzen "
                "und hoher Ordnung deutlich mehr.")
    if st.button("Netz erzeugen und rechnen", type="primary", key="bd_run", disabled=bool(errs)):
        try:
            work.mkdir(parents=True, exist_ok=True)
            with st.spinner("Netz wird erzeugt …"):
                stats = fb.build_mesh(m, work)
            (work / "job.json").write_text(json.dumps(fb.job(m), ensure_ascii=False), encoding="utf-8")
            for f in work.glob("mode_*.npz"):
                f.unlink()
            S.bd_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script="fem_bands_worker.py")
            S.bd_dir, S.bd_cancel, S.bd_stats = str(work), False, stats
            S.bd_run_model = copy.deepcopy(m)
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("bd_msg"):
        (st.success if S.bd_msg[0] == "ok" else st.error)(S.bd_msg[1] + " Die Ergebnisse stehen im Reiter 3.")
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            st.code(fr.read_log(work, tail=40))


def _running():
    proc, folder = S.bd_proc, Path(S.bd_dir)
    log = fr.read_log(folder)
    st.subheader("Rechnung läuft …")
    if proc.poll() is None:
        frac, label = fr.progress_info(log)
        st.progress(frac, text=label)
        st.code("\n".join(log.splitlines()[-10:]) or "(noch keine Ausgabe)")
        if S.get("bd_cancel"):
            st.info("Abbruch angefordert: die Rechnung hält nach dem laufenden k-Punkt an.")
        elif st.button("Abbrechen", key="bd_cancel_btn"):
            fr.request_cancel(folder)
            S.bd_cancel = True
            st.rerun()
        time.sleep(1.5)
        st.rerun()
    else:
        flags = fr.log_flags(log)
        S.bd_proc = None
        ok = proc.returncode == 0 and not flags["errors"]
        S.bd_msg = ("ok", "Abgebrochen." if flags["cancelled"] else "Fertig.") if ok else ("error", f"Beendet mit Code {proc.returncode}: " + "; ".join(flags["errors"][:3]))
        st.rerun()


# ===================================================================================================================== results
def _tab_results(work):
    res_file = work / "results.json"
    if not res_file.is_file():
        st.info("Noch keine Ergebnisse: im Reiter 2 rechnen.")
        return
    res = json.loads(res_file.read_text(encoding="utf-8"))
    pts = res["points"]
    for p in pts:
        if "error" in p:
            st.error(f"k-Punkt {p['index']}: {p['error']}")
    if not any("f" in p for p in pts):
        return
    meta = res["meta"]
    model = S.get("bd_run_model") or S.bd_model
    st.caption(f"{meta.get('name', '')} · {sum('f' in p for p in pts)} von {len(res['x'])} k-Punkten · {meta.get('cells')} Zellen, p = {meta.get('order')}, "
               f"{meta.get('dofs')} Freiheitsgrade · hpfem {meta.get('hpfem', '?')} · Start {meta.get('started', '')}")
    if res.get("cancelled"):
        st.warning("Die Rechnung wurde abgebrochen: nur ein Teil des Pfads liegt vor.")
    sub = st.tabs(["Bandstruktur", "Lücken und Tabelle", "Modenfelder"])
    x, ticks = res["x"], res["ticks"]
    with sub[0]:
        ref = st.checkbox("Leeres Gitter |k + G| als graue Referenz", value=False, key="bd_ref",
                          help="Die Bänder ohne Kristall: zeigt, wo der Kristall die Bänder verschiebt und Lücken öffnet.")
        empty = fb.empty_lattice_bands(model, int(meta["bands"]))[:len(pts)] if ref and model["lattice"] == meta.get("lattice") else None
        ui.show_fig(fb.fig_bands(pts, x, ticks, gaps=True, empty_lattice=empty), "bandstruktur.png", "bd_dl_bands")
        gaps = fb.gaps_dataframe(pts)
        if len(gaps):
            st.success("Lücken über den ganzen Pfad: " + "; ".join(f"Band {r['zwischen Band']}: f = {r['untere Kante f']:.4f} bis {r['obere Kante f']:.4f} "
                                                              f"(Δf/Mitte = {r['Δf/Mitte (%)']:.1f} %)" for _, r in gaps.iterrows()))
        else:
            st.info("Keine Lücke zwischen benachbarten Bändern über den ganzen Pfad.")
        st.caption("f = ω a / (2π c) = a / λ. Orange: Frequenzbereiche ohne Band über den ganzen Pfad, daneben Δf/Mitte. Das niedrigste Band beginnt bei f = 0.")
    with sub[1]:
        gaps = fb.gaps_dataframe(pts)
        if len(gaps):
            ui.table_show(gaps.round(5))
        df = fb.bands_dataframe(pts, x)
        ui.table_show(df.round(5))
        c = st.columns(2)
        c[0].download_button("Bänder (CSV)", ui.csv_bytes(df), file_name="bandstruktur.csv", mime="text/csv", key="bd_dl_csv")
        c[1].download_button("Lücken (CSV)", ui.csv_bytes(gaps), file_name="bandluecken.csv", mime="text/csv", key="bd_dl_gaps")
        worst = max((p["residual"] for p in pts if "residual" in p), default=0.0)
        st.caption(f"Größtes Eigenwert-Residuum: {worst:.2e}. Feldkarten und Bänder ändern sich mit dem Netz: zur Probe die Ordnung erhöhen.")
        st.download_button("Modell (JSON)", json.dumps(model, indent=1, ensure_ascii=False).encode("utf-8"), file_name="bandstruktur_modell.json",
                           mime="application/json", key="bd_dl_model")
    with sub[2]:
        withf = [p for p in pts if p.get("modes_file")]
        if not withf:
            st.info("Keine Modenfelder: im Reiter 2 „Modenfelder“ einschalten.")
            return
        labels = {p["index"]: next((lab for xt, lab in ticks if abs(xt - x[p["index"]]) < 1e-9), f"k {p['index']}") for p in withf}
        c = st.columns(3)
        i = c[0].selectbox("k-Punkt", range(len(withf)), format_func=lambda i: f"{labels[withf[i]['index']]} (k = {withf[i]['k'][0]:.3f}, {withf[i]['k'][1]:.3f})",
                           key="bd_mf_k")
        b = c[1].selectbox("Band", range(len(withf[i]["f"])), format_func=lambda b: f"Band {b + 1} (f = {withf[i]['f'][b]:.4f})", key="bd_mf_b")
        q = c[2].selectbox("Größe", ["abs2", "ex", "ey"], key="bd_mf_q", format_func={"abs2": "|E|²", "ex": "Re E_x", "ey": "Re E_y"}.get)
        npz = np.load(work / withf[i]["modes_file"])
        ui.show_fig(fb.fig_mode(npz, b, q, f"{labels[withf[i]['index']]}, Band {b + 1}, f = {withf[i]['f'][b]:.4f}", model if model["lattice"] == meta.get("lattice") else None),
                    "bandfeld.png", "bd_dl_mode")
        st.caption("Bloch-Feld der Zelle (Betrag periodisch, Re E_x/E_y tragen die Bloch-Phase). Bei entarteten Bändern ist die Wahl der Basis willkürlich.")
