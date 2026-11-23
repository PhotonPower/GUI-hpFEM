"""Cross grating (doubly periodic) in the FEM model builder (Streamlit), input mask of hp-FEM M21. Called from fem_app.py: ``render_xgrating(ctx)``.

Model, layers with holes and pillars, incidence, wavelength sweep and the table of diffraction orders. There is no computation yet: the solver
interface follows with hp-FEM M21 S2 (PhotonPower/hp-FEM#237). Models and figures: fem_xgrating.py.
"""
from __future__ import annotations

import copy
import json

import pandas as pd
import streamlit as st

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
Einfallsebene) oder p. Zeitfaktor exp(−iωt). Die Reiter „Rechnung“ und die Spalten R und T der Ordnungstabelle sind noch leer: die Rechnung
folgt mit der Schnittstelle der Bibliothek (hp-FEM M21).

**Ordnungen.** Die Tabelle zeigt, welche Beugungsordnungen (m, n) bei der gewählten Wellenlänge im Einfallsmedium (Reflexion) und im Substrat
(Transmission) ausbreitungsfähig sind: k_t = k_t,ein + 2π (m / p_x, n / p_y), |k_t| < k₀ n. Die Energiebilanz ΣR + ΣT + A − 1 steht
nach der Rechnung unter der Tabelle.

**Austausch.** Das Modell lässt sich als JSON speichern und laden. Fremde Materialangaben ohne „type“ (z. B. {"n": 1.5}) werden beim Laden
umgesetzt. FourierGrating-Dateien setzt der Import in hpfem um (y als Normale → z), nicht diese Seite.
"""


def _init():
    if "xg_model" in S:
        return
    S.xg_model = copy.deepcopy(fx.default_model())
    S.xg_ver = 0


def _bump():
    S.xg_ver += 1


def _num(col, label, key, value, step=10.0, lo=-1e6, hi=1e6, help=None):
    return float(col.number_input(label, lo, hi, float(value), step, key=key, format="%.6g", help=help))


def render_xgrating(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    st.title("FEM-Modellwerkstatt: Kreuzgitter (doppelt periodisch)")
    st.caption("Lochgitter und Säulengitter, in x und y periodisch, mit Schichten, schrägem Einfall und Tabelle der Beugungsordnungen. "
               "Eingabemaske ohne Rechnung (hp-FEM M21).")
    t1, t2, t3 = st.tabs(["1 Modell", "2 Ordnungen und Ergebnisse", "3 Info"])
    with t1:
        _tab_model()
    with t2:
        _tab_orders()
    with t3:
        st.markdown(INFO)


def _tab_model():
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
    c = st.columns(4)
    s["cx"], s["cy"] = _num(c[0], "Mitte x (nm)", f"{k}_cx", s["cx"]), _num(c[1], "Mitte y (nm)", f"{k}_cy", s["cy"])
    if s["type"] == "circle":
        s["radius"] = _num(c[2], "Radius (nm)", f"{k}_r", s["radius"], 5.0, 0.0)
    elif s["type"] == "ellipse":
        s["rx"], s["ry"] = _num(c[2], "Halbachse x (nm)", f"{k}_rx", s["rx"], 5.0, 0.0), _num(c[3], "Halbachse y (nm)", f"{k}_ry", s["ry"], 5.0, 0.0)
    else:
        s["wx"], s["wy"] = _num(c[2], "Breite x (nm)", f"{k}_wx", s["wx"], 5.0, 0.0), _num(c[3], "Breite y (nm)", f"{k}_wy", s["wy"], 5.0, 0.0)


def _tab_orders():
    m = S.xg_model
    errs = [t for lvl, t in fx.validate(m) if lvl == "error"]
    for e in errs:
        st.error(e)
    if errs:
        return
    lams = fx.sweep_list(m["sweep"], m["incidence"])
    lam = lams[0] if len(lams) == 1 else float(st.select_slider("Wellenlänge für die Ordnungstabelle (nm)", options=lams, value=lams[0], key="xg_order_lam"))
    po = fx.propagating_orders(m, lam)
    st.markdown(f"**λ = {lam:g} nm:** {len(po['R'])} Ordnung(en) in Reflexion, {len(po['T'])} in Transmission ausbreitungsfähig.")
    rows = [{"m": r["m"], "n": r["n"], "Reflexion": "ja" if r["reflektiert"] else "–", "Transmission": "ja" if r["transmittiert"] else "–",
             "R": None, "T": None} for r in fx.orders_table(m, lam)]
    ui.table_show(pd.DataFrame(rows))
    st.info("R und T je Ordnung und die Energiebilanz ΣR + ΣT + A − 1 erscheinen, sobald die Rechnung angebunden ist (hp-FEM M21, "
            "Schnittstelle in PhotonPower/hp-FEM#237). Berechnen lässt sich hier noch nichts.")
    st.download_button("Ordnungstabelle (CSV)", ui.csv_bytes(pd.DataFrame(rows)), file_name="ordnungen.csv", mime="text/csv", key="xg_csv")
