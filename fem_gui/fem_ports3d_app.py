"""User interface of the task "Wellenleiter und Ports (3D)" of the mode "3D-Strukturen" (hp-FEM 0.5): model, parameters, results. Called from fem_threed_app.py
(``tab_model``, ``tab_params``, ``results``). Model, job and figures: fem_ports3d.py; worker: fem_ports3d_worker.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

import fem_ports3d as fp
import fem_run as fr
import fem_ui as ui

S = st.session_state


def _num(col, label, key, value, step=10.0, lo=0.0, hi=1e7, help=None, fmt="%.5g"):
    return float(col.number_input(label, lo, hi, float(value), step, key=key, format=fmt, help=help))


def _rename(m, old, new):
    m["materials"] = {(new if k == old else k): v for k, v in m["materials"].items()}
    for key in ("core", "clad"):
        if m[key] == old:
            m[key] = new
    f = m["file"]
    f["tag_material"] = {t: (new if nm == old else nm) for t, nm in f["tag_material"].items()}


def tab_model(m, ver, bump, ctx, work):
    """Model editor of the ports model m (S.td_model["p3"])."""
    left, right = st.columns([6, 5])
    with left:
        m["device"] = st.radio("Bauteil", list(fp.DEVICES), index=list(fp.DEVICES).index(m["device"]), format_func=fp.DEVICES.get, key=f"p3_dev_{ver}", horizontal=True)
        if m["device"] != "file":
            c = st.columns(3)
            m["width_nm"] = _num(c[0], "Kernbreite (nm)", f"p3_w_{ver}", m["width_nm"], 10.0, 20.0, 5000.0)
            m["height_nm"] = _num(c[1], "Kernhöhe (nm)", f"p3_h_{ver}", m["height_nm"], 10.0, 20.0, 5000.0)
            if m["device"] == "ring":
                r = m["ring"]
                r["radius_nm"] = _num(c[2], "Ringradius, Mittellinie (nm)", f"p3_rr_{ver}", r["radius_nm"], 50.0, 300.0, 50000.0, "Radius der Mittellinie des Rings; die Zuleitungen liegen bei y = ±(R + Breite + Spalt).")
                m["gap_nm"] = _num(st, "Spalt Ring–Zuleitung (nm)", f"p3_g_{ver}", m["gap_nm"], 10.0, 10.0, 5000.0, "Mantel zwischen Ring und jeder der beiden Zuleitungen.")
                c = st.columns(3)
                r["margin_x_nm"] = _num(c[0], "Rand längs der Zuleitungen (nm)", f"p3_rmx_{ver}", r["margin_x_nm"], 50.0, 100.0, 20000.0, "Mantel hinter dem Ring bis zu den Port-Flächen (x).")
                r["margin_y_nm"] = _num(c[1], "Rand neben den Zuleitungen (nm)", f"p3_rmy_{ver}", r["margin_y_nm"], 50.0, 100.0, 20000.0, "Mantel außerhalb der äußeren Zuleitungskanten bis zur PEC-Wand (y).")
                r["margin_z_nm"] = _num(c[2], "Rand oben und unten (nm)", f"p3_rmz_{ver}", r["margin_z_nm"], 50.0, 100.0, 20000.0, "Mantel unter und über den Kernen bis zur PEC-Wand (z).")
            else:
                m["length_nm"] = _num(c[2], "Länge (nm)", f"p3_l_{ver}", m["length_nm"], 100.0, 100.0, 200000.0, "Länge des Bauteils zwischen den Ports (z).")
            if m["device"] == "coupler":
                c = st.columns(2)
                m["gap_nm"] = _num(c[0], "Spalt zwischen den Kernen (nm)", f"p3_g_{ver}", m["gap_nm"], 10.0, 10.0, 5000.0)
                m["ports"] = c[1].radio("Ports", list(fp.PORT_STYLES), index=list(fp.PORT_STYLES).index(m["ports"]), format_func=lambda k: fp.PORT_STYLES[k].split(" (")[0],
                                        key=f"p3_pst_{ver}", help="Halbflächen: " + fp.PORT_STYLES["half"] + ". Supermoden: " + fp.PORT_STYLES["full"] + ".")
            if m["device"] != "ring":
                c = st.columns(2)
                m["margin_x_nm"] = _num(c[0], "Rand seitlich (nm)", f"p3_mx_{ver}", m["margin_x_nm"], 50.0, 50.0, 20000.0, "Mantel neben den Kernen bis zur PEC-Wand (x).")
                m["margin_y_nm"] = _num(c[1], "Rand oben und unten (nm)", f"p3_my_{ver}", m["margin_y_nm"], 50.0, 50.0, 20000.0, "Mantel unter und über den Kernen bis zur PEC-Wand (y).")
            st.subheader("Materialien")
            lam0 = fp.lam_ref(m)
            ui.material_editor(m, ver, lam0, {m["core"], m["clad"]}, _rename, bump, prefix="p3_")
            names = list(m["materials"])
            c = st.columns(2)
            m["core"] = c[0].selectbox("Kern", names, index=names.index(m["core"]) if m["core"] in names else 0, key=f"p3_core_{ver}")
            m["clad"] = c[1].selectbox("Mantel", names, index=names.index(m["clad"]) if m["clad"] in names else 0, key=f"p3_clad_{ver}")
        else:
            _file_editor(m, ver, bump, ctx, work)
    with right:
        for lvl, text in fp.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        if m["device"] == "ring":
            st.markdown("**Aufbau.** Add-Drop-Filter der integrierten Photonik: ein Ring zwischen zwei geraden Zuleitungen gleichen Querschnitts (x längs der Zuleitungen, y quer, z nach oben), "
                        "Mantel zwischen idealen Leiterwänden (PEC), Gmsh-Netz mit gekrümmten Tetraedern, vier modale Ports (1 Eingang, 2 Through, 3 Drop, 4 Add). Angeregt wird Port 1. "
                        "Das halbe Modell (z ≥ h/2, magnetische Spiegelebene) spart etwa ein Drittel der Unbekannten; die S-Parameter sind die des ganzen Bauteils.")
            st.caption("Referenz: der isolierte Ring als Rotationskörper (`AxisymmetricCavity`) liefert λ der Grundmode je Azimutordnung m (Bibliothek: für R = 2 µm m = 19 bei 1557,39 nm); aus Through und Drop wird "
                       "der gemeinsame komplexe Pol (Resonanz, Halbwertsbreite, Güte) angepasst. FSR-Schätzung: λ²/(n_g·2πR) aus dem Gruppenindex der geraden Zuleitung.")
        elif m["device"] == "file":
            st.markdown("**Aufbau.** Ein Gmsh-Netz (Version 4.1, ASCII) mit physikalischen Volumina (Materialien) und Flächen (PEC-Wände, Ports), z. B. der Ringresonator der Bibliothek "
                        "(`examples/ring_resonator_3d/ring_mesh.py`). Die Ports sind Flächenmarken; jeder Port löst seine Moden mit dem 2D-Modenlöser.")
        else:
            st.markdown("**Aufbau.** z längs des Wellenleiters, x quer, y nach oben. Seitenflächen: ideale Leiter (PEC), die Moden fallen innerhalb der Ränder ab. Stirnflächen z = 0 und z = L: "
                        "modale Ports. Strukturiertes Tetraedernetz, Zellkanten treffen die Kernkanten.")
            st.caption("Referenz: Streifenwellenleiter: β der Grundmode des Querschnitts (2D-Modenlöser) gegen die Phase der Transmission; Koppler: Theorie gekoppelter Moden "
                       "P_cross = sin²(κL) mit κ = (β_gerade − β_ungerade)/2 aus den Supermoden.")


def _file_editor(m, ver, bump, ctx, work):
    f = m["file"]
    up = st.file_uploader("Gmsh-Datei (.msh, Version 4.1 ASCII)", type=["msh"], key=f"p3_up_{ver}")
    if up is not None and up.name != f.get("name"):
        folder = Path(work) / "ports_files"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "mesh.msh"
        path.write_bytes(up.getvalue())
        f.update(name=up.name, path=str(path), tags={}, tag_material={}, pec_tags=[], ports=[])
        bump()
        st.rerun()
    if not f.get("name"):
        return
    st.caption(f"Datei: {f['name']}")
    c = st.columns([1, 1, 1])
    f["scale"] = c[0].selectbox("Einheit der Koordinaten", ["nm", "µm", "mm", "m"], index=["nm", "µm", "mm", "m"].index(f["scale"]), key=f"p3_unit_{ver}")
    if c[1].button("Datei lesen", key=f"p3_read_{ver}"):
        import fem_threed as ft

        info = ft.run_inspect(ctx["py"], ctx["repo"], Path(work) / "ports_files" / "inspect", f["path"], f["scale"], "ports", int(ctx["threads"]))
        if "error" in info:
            st.error(info["error"])
        else:
            f["tags"] = info
            names = list(m["materials"])
            for tag in info["cells"]:
                f["tag_material"].setdefault(tag, names[0])
            bump()
            st.rerun()
    info = f.get("tags")
    if not info:
        return
    st.caption(f"{info['n_cells']} Zellen, Geometrieordnung {info.get('geometry_order', '?')}; Ausdehnung "
               + " × ".join(f"{(b - a) / 1e-9:.0f}" for a, b in zip(info['bbox'][0], info['bbox'][1])) + " nm")
    st.markdown("**Zellmarken → Material**")
    ui.material_editor(m, ver, fp.wavelengths(m)[0], set(f["tag_material"].values()), _rename, bump, prefix="p3f_")
    names = list(m["materials"])
    for tag, count in info["cells"].items():
        cur = f["tag_material"].get(tag, names[0])
        f["tag_material"][tag] = st.selectbox(f"Zellmarke {tag} ({count} Zellen)", names, index=names.index(cur) if cur in names else 0, key=f"p3_tm_{ver}_{tag}")
    st.markdown("**Flächenmarken → Rolle**")
    roles = {"none": "ungenutzt", "pec": "PEC-Wand", "port": "Port"}
    cur_ports = {str(p["tag"]): p["modes"] for p in f["ports"]}
    pec, ports = [], []
    for tag, count in info["facets"].items():
        role = "port" if tag in cur_ports else ("pec" if int(tag) in f["pec_tags"] else "none")
        c = st.columns([2, 1])
        role = c[0].selectbox(f"Flächenmarke {tag} ({count} Flächen)", list(roles), index=list(roles).index(role), format_func=roles.get, key=f"p3_fr_{ver}_{tag}")
        if role == "pec":
            pec.append(int(tag))
        elif role == "port":
            n = int(c[1].number_input("Moden", 1, 8, int(cur_ports.get(tag, 1)), key=f"p3_fm_{ver}_{tag}"))
            ports.append(dict(tag=int(tag), modes=n))
    f["pec_tags"], f["ports"] = pec, ports
    m["excite_port"] = int(st.number_input("Anregung: Port (Reihenfolge der Ports oben, ab 1)", 1, max(len(ports), 1), int(m["excite_port"]) + 1, key=f"p3_xp_{ver}")) - 1


def tab_params(m, ver):
    """Wavelengths, mesh and order of the ports model."""
    if m["device"] == "ring":
        return _ring_params(m, ver)
    st.subheader("Wellenlängen")
    s = m["sweep"]
    c = st.columns(3)
    s["start"] = _num(c[0], "von (nm)", f"p3_s0_{ver}", s["start"], 10.0, 200.0, 20000.0)
    s["stop"] = _num(c[1], "bis (nm)", f"p3_s1_{ver}", s["stop"], 10.0, 200.0, 20000.0)
    s["count"] = int(c[2].number_input("Zahl der Wellenlängen", 1, 100, int(s["count"]), key=f"p3_sn_{ver}", help="Jede Wellenlänge ist eine eigene Faktorisierung (mit allen Anregungskanälen)."))
    st.subheader("Netz und Ordnung")
    c = st.columns(4)
    if m["device"] != "file":
        m["cell_nm"] = _num(c[0], "Zelle (nm)", f"p3_cell_{ver}", m["cell_nm"], 10.0, 20.0, 2000.0, "Zellkante des Tetraedernetzes (Würfelzellen, je sechs Tetraeder); die Gitterlinien treffen die Kernkanten.")
        m["cell_z_nm"] = _num(c[1], "Zelle längs z (nm, 0 = wie Zelle)", f"p3_cellz_{ver}", m["cell_z_nm"], 10.0, 0.0, 5000.0, "Gröber längs des Wellenleiters spart viele Zellen (die Felder variieren dort langsam).")
    m["order"] = int(c[2].number_input("Ordnung p", 1, 4, int(m["order"]), key=f"p3_p_{ver}"))
    if m["device"] == "straight":
        m["num_modes"] = int(c[3].number_input("Moden je Port", 1, 6, int(m["num_modes"]), key=f"p3_nm_{ver}"))
    c = st.columns(3)
    m["excite_port"] = int(c[0].number_input("Anregung: Port", 1, 4 if m["device"] != "file" else 8, int(m["excite_port"]) + 1, key=f"p3_ep_{ver}", disabled=m["device"] == "file")) - 1 if m["device"] != "file" else m["excite_port"]
    m["excite_mode"] = int(c[1].number_input("Anregung: Mode", 1, 6, int(m["excite_mode"]) + 1, key=f"p3_em_{ver}")) - 1
    m["field"] = c[2].checkbox("Feld der Anregung speichern", value=bool(m["field"]), key=f"p3_fld_{ver}", help="Eine zusätzliche Lösung bei der ersten Wellenlänge: |E|² im Längsschnitt und im Querschnitt.")
    n = fp.cell_count(m)
    if n:
        st.caption(f"Etwa {n} Würfelzellen, das sind rund {6 * n} Tetraeder; die genaue Zahl und der Speicher stehen nach „Netz prüfen“.")


def _ring_params(m, ver):
    r = m["ring"]
    st.subheader("Wellenlängen")
    r["scan"] = st.radio("Durchlauf", ["manual", "auto"], index=["manual", "auto"].index(r["scan"]), horizontal=True, key=f"p3_rsc_{ver}",
                         format_func={"manual": "feste Wellenlängen (von, bis, Zahl)", "auto": "automatisch: Suche um die Resonanz des isolierten Rings, dann feiner um den angepassten Pol"}.get)
    if r["scan"] == "auto":
        c = st.columns(2)
        r["target_nm"] = _num(c[0], "Zielwellenlänge (nm)", f"p3_rtg_{ver}", r["target_nm"], 10.0, 400.0, 20000.0, "Daraus werden die Azimutordnung und die nächste Resonanz des isolierten Rings bestimmt; das Material ist hier eingefroren.")
        r["search_nm"] = _num(c[1], "Suchfenster ± (nm)", f"p3_rse_{ver}", r["search_nm"], 0.5, 0.1, 100.0, "Neun Wellenlängen im Fenster um die Referenzresonanz, danach neun innerhalb ±2 Halbwertsbreiten des angepassten Pols (18 Läufe).")
    else:
        s = m["sweep"]
        c = st.columns(3)
        s["start"] = _num(c[0], "von (nm)", f"p3_s0_{ver}", s["start"], 10.0, 200.0, 20000.0)
        s["stop"] = _num(c[1], "bis (nm)", f"p3_s1_{ver}", s["stop"], 10.0, 200.0, 20000.0)
        s["count"] = int(c[2].number_input("Zahl der Wellenlängen", 1, 200, int(s["count"]), key=f"p3_sn_{ver}", help="Jede Wellenlänge ist eine eigene Faktorisierung (die Matrix hängt von ω ab, kein Wiederverwenden möglich)."))
    st.subheader("Netz und Ordnung")
    c = st.columns(4)
    r["cell_core_nm"] = _num(c[0], "Zelle am Kern (nm)", f"p3_rcc_{ver}", r["cell_core_nm"], 10.0, 30.0, 1000.0, "Zielgröße der Tetraeder in und an den Kernen (Bibliothek: 100 nm, Test: 150 nm).")
    r["cell_far_nm"] = _num(c[1], "Zelle fern (nm)", f"p3_rcf_{ver}", r["cell_far_nm"], 10.0, 50.0, 2000.0, "Zielgröße im Mantel im Staffelabstand vom Kern.")
    r["grade_nm"] = _num(c[2], "Staffelweg (nm)", f"p3_rgr_{ver}", r["grade_nm"], 50.0, 100.0, 5000.0, "Abstand vom Kern, in dem die Zellgröße von „am Kern“ auf „fern“ wächst.")
    r["half"] = c[3].checkbox("Halbes Modell", value=bool(r["half"]), key=f"p3_rhf_{ver}", help="Nur z ≥ h/2 vernetzen, die Mittelebene der Kerne ist eine magnetische Wand (TE-ähnliche Moden: E tangential gerade). "
                                                                                           "Spart etwa ein Drittel der Unbekannten und mehr Speicher; die S-Parameter sind die des ganzen Bauteils.")
    c = st.columns(4)
    r["order_core"] = int(c[0].number_input("Ordnung p am Kern", 1, 5, int(r["order_core"]), key=f"p3_roc_{ver}", help="Zellen innerhalb der Reichweite vom Kern bekommen diese Ordnung."))
    r["order_far"] = int(c[1].number_input("Ordnung p fern", 1, 5, int(r["order_far"]), key=f"p3_rof_{ver}"))
    r["reach_nm"] = _num(c[2], "Reichweite Kernordnung (nm)", f"p3_rre_{ver}", r["reach_nm"], 25.0, 0.0, 2000.0, "Abstand vom Kern (Zellschwerpunkte), bis zu dem die höhere Ordnung gilt (Bibliothek: 250 nm).")
    backs = ["auto", "sparselu", "mumps", "cudss"]
    r["backend"] = c[3].selectbox("Löser", backs, index=backs.index(r["backend"]), key=f"p3_rbk_{ver}",
                                  help="Direkter Löser. Das Wheel enthält nur SparseLU; MUMPS und cuDSS nur, wenn die Bibliothek damit gebaut ist (sonst bricht der Worker mit einer klaren Meldung ab).")
    st.subheader("Referenz und Felder")
    c = st.columns(4)
    r["reference"] = c[0].checkbox("Referenz und FSR", value=bool(r["reference"]), key=f"p3_rrf_{ver}", help="Isolierter Ring als Rotationskörper (λ der Grundmode für die Azimutordnungen m0−1, m0, m0+1) "
                                                                                                         "und Gruppenindex der Zuleitung aus dem 2D-Modenlöser (drei Wellenlängen).")
    r["ref_p"] = int(c[1].number_input("Ordnung der Referenz", 1, 6, int(r["ref_p"]), key=f"p3_rrp_{ver}", help="Bibliothek: p = 4 auf 50-nm-Zellen (0,01 nm); p = 2 auf 100-nm-Zellen weicht etwa 0,7 nm ab."))
    r["ref_cell_nm"] = _num(c[2], "Zelle der Referenz (nm)", f"p3_rrc_{ver}", r["ref_cell_nm"], 10.0, 20.0, 400.0)
    r["field_res"] = c[3].checkbox("Feld beim stärksten Drop", value=bool(r["field_res"]), key=f"p3_rfl_{ver}", help="Eine zusätzliche Lösung bei der Wellenlänge mit dem größten Drop: |E|² in der x–y-Ebene durch die Kernmitte.")
    st.caption("Maße der Bibliothek: Produktionsnetz 100 nm, p = 3/2: 0,36 bis 1,7 Millionen Unbekannte (MUMPS, Server); das Testnetz (R = 1,5 µm, 150 nm, p = 1) hat rund 17 000 Unbekannte im halben Modell "
               "und braucht je Wellenlänge einige Sekunden. Die genaue Zahl und der Speicher stehen nach „Netz prüfen“.")


# ===================================================================================================================== results
def _results_ring(res, work, model, ok):
    meta, ring, m = res["meta"], res.get("ring") or {}, model["p3"]
    sub = st.tabs(["Spektrum und Resonanz", "S-Matrix", "Feld", "Tabelle"])
    with sub[0]:
        ui.show_fig(fp.fig_ring(ok, ring), "ringspektrum.png", "p3_dl_ring")
        c = st.columns([3, 2])
        with c[1]:
            tab = fp.ring_summary_dataframe(ring)
            if len(tab):
                ui.table_show(tab)
        with c[0]:
            worst = max(abs(p["loss"]) for p in ok)
            notes = [f"Leistungsbilanz: |Σ Port-Leistungen − 1| ≤ {worst:.1e} (verlustfreie Materialien: der Fehler der Portnormierung, nicht des Netzes)."]
            f = ring.get("fit")
            if f is None:
                notes.append("Keine Resonanz angepasst: dafür braucht es mindestens vier Wellenlängen, und der gemeinsame Pol von Through und Drop muss im Fenster liegen.")
            if f and ring.get("lam_ref_nm"):
                notes.append(f"Die 3D-Resonanz liegt {f['wavelength'] - ring['lam_ref_nm']:+.2f} nm neben der des isolierten Rings (Kopplung an die Zuleitungen und Diskretisierung).")
            if ring.get("fsr_ref_nm") and ring.get("fsr_ng_nm"):
                ratio = ring["fsr_ng_nm"] / ring["fsr_ref_nm"]
                notes.append(f"FSR: isolierter Ring {ring['fsr_ref_nm']:.1f} nm, λ²/(n_g·L) der geraden Zuleitung {ring['fsr_ng_nm']:.1f} nm (Verhältnis {ratio:.2f}). Für R = 2 µm nennt die Bibliothek 45 bis 48 nm "
                             "für den isolierten Ring; bei kleinem Radius kann die Schätzung stark abweichen (R = 1,5 µm: 62 gegen 85 nm; vermutlich, weil die Mode im Bogen nach außen rückt, nicht geprüft).")
            if f and f["residual"] > 0.05:
                notes.append(f"Der Rest der Anpassung ist groß ({f['residual']:.2f}): das Spektrum im Fenster wird nicht von einem einzelnen Pol beschrieben (Scheinresonanzen bei zu grobem Netz, mehrere Resonanzen im Fenster); "
                             "Lage, Breite und Güte sind dann nicht belastbar.")
            if int(m["ring"]["order_core"]) == 1:
                notes.append("p = 1: das Netz löst den Resonator nicht auf; Spitzen im Spektrum können Scheinresonanzen sein (Bibliothek: alle paar Nanometer, Rückreflexion bis 0,47).")
            st.caption(" ".join(notes))
    with sub[1]:
        i = st.selectbox("Wellenlänge", range(len(ok)), format_func=lambda i: f"{ok[i]['lam_nm']:.4f} nm", key="p3_ring_lam") if len(ok) > 1 else 0
        ui.table_show(fp.s_matrix_dataframe(ok[i]).round(6))
        st.caption("Anregung: Port 1 (Grundmode). |S|² sind Leistungsanteile; die Phase ist die der leistungsnormierten Amplitude gegen die einlaufende Welle.")
    with sub[2]:
        withf = [p for p in ok if p.get("field_file")]
        if not withf:
            st.info("Kein Feld gespeichert (Reiter 2: „Feld beim stärksten Drop“).")
        else:
            npz = np.load(Path(work) / withf[0]["field_file"])
            ui.show_fig(fp.fig_ring_field(npz, m, f"|E|² in der Kernmitte, λ = {withf[0]['lam_nm']:.4f} nm, Anregung an Port 1"), "ringfeld.png", "p3_dl_ringfld")
            st.caption("Beim halben Modell: 10 nm über der Mittelebene. Weiß gestrichelt: Kanten von Ring und Zuleitungen. Skala beliebig.")
    with sub[3]:
        df = fp.results_dataframe(ok, meta.get("device"), "")
        c = st.columns(3)
        c[0].download_button("Tabelle (CSV)", ui.csv_bytes(df), file_name="ringresonator_3d.csv", mime="text/csv", key="p3_dl_rcsv")
        c[1].download_button("Resonanz (CSV)", ui.csv_bytes(fp.ring_summary_dataframe(ring)), file_name="ringresonanz.csv", mime="text/csv", key="p3_dl_rres")
        c[2].download_button("Modell (JSON)", json.dumps(m, indent=1, ensure_ascii=False).encode("utf-8"), file_name="ringresonator_3d.json", mime="application/json", key="p3_dl_rmodel")
        ui.table_show(df.round(6))


def results(res, work, model):
    pts = res["points"]
    meta = res["meta"]
    for p in pts:
        if "error" in p:
            st.error(f"λ = {p['lam_nm']:g} nm: {p['error']}")
    ok = [p for p in pts if "R" in p]
    if not ok:
        return
    m = model["p3"]
    if meta.get("device") == "ring":
        return _results_ring(res, work, model, ok)
    sub = st.tabs(["Transmission", "S-Matrix", "Felder", "Tabelle"])
    with sub[0]:
        if len(ok) > 1:
            ui.show_fig(fp.fig_sparams(ok), "sparameter.png", "p3_dl_sp")
        df = fp.results_dataframe(ok, meta.get("device"), m["ports"])
        ui.table_show(df.round(5))
        p = ok[0]
        notes = []
        if "phase_error" in p:
            ref = f", feines Referenznetz (p + 2, halbe Zelle): {p['n_eff_ref']:.4f}" if "n_eff_ref" in p else ""
            notes.append(f"n_eff der Portmode {p['n_eff0']:.4f} (2D-Modenlöser auf demselben Netz{ref}): der Unterschied ist der Diskretisierungsfehler des Querschnitts. Phasenfehler "
                         f"|arg(t · e^(iβL))| = {p['phase_error']:.3f} rad: numerische Dispersion längs z, wächst mit der Länge, fällt mit kleinerer Zelle und höherem p (bei p = 1 groß).")
        if "cross_cmt" in p:
            ref = f"; mit dem feinen Referenznetz {p['coupling_length_ref_nm'] / 1000:.1f} µm und cross = {p['cross_cmt_ref']:.4f}" if "cross_cmt_ref" in p else ""
            notes.append(f"Gekoppelte Moden auf dem Netz der 3D-Rechnung: Kopplungslänge {p['coupling_length_nm'] / 1000:.1f} µm, cross = {p['cross_cmt']:.4f}{ref}. Ein großer Abstand zwischen beiden "
                         "heißt: das Netz ist für die Supermoden nicht aufgelöst (p = 1 mit 150-nm-Zellen koppelt viel zu stark).")
            if "phase_error_even" in p:
                notes.append(f"Supermoden-Ports: Phasenfehler gerade/ungerade {p['phase_error_even']:.3f} / {p['phase_error_odd']:.3f} rad. cross wird aus der Phasendifferenz der beiden Supermoden "
                             "gewonnen und ist bei kurzen Kopplern (cross klein) sehr empfindlich auf ihren Fehler.")
            else:
                notes.append("Halbflächen-Ports: die Anregung ist die Mode des halben Querschnitts (PEC an der Port-Ebene im Spalt); das verfälscht cross bei kurzen Kopplern um einige Prozent "
                             "(Bibliothek: etwa 1 % absolut), Supermoden-Ports sind genauer.")
        notes.append("Leistungsbilanz R + T + Verlust = 1: ein Verlust ungleich 0 bei verlustfreien Materialien ist der Fehler der Portnormierung.")
        st.caption(" ".join(notes))
    with sub[1]:
        i = st.selectbox("Wellenlänge", range(len(ok)), format_func=lambda i: f"{ok[i]['lam_nm']:g} nm", key="p3_s_lam") if len(ok) > 1 else 0
        df = fp.s_matrix_dataframe(ok[i])
        ui.table_show(df.round(5))
        st.caption("|S|² zwischen den Kanälen (Port · Mode): Leistungsanteile; Zeile = Anregung, Spalte = Ziel. Port-Moden: "
                   + "; ".join(f"Port {int(k) + 1}: " + ", ".join(f"n_eff = {mm['n_eff']:.4f}{'' if mm['propagating'] else ' (abklingend)'}" for mm in v) for k, v in ok[i]["port_modes"].items()))
    with sub[2]:
        withf = [p for p in ok if p.get("field_file")]
        if not withf:
            st.info("Kein Feld gespeichert.")
        else:
            npz = np.load(Path(work) / withf[0]["field_file"])
            ol = fp.outline(m)
            c = st.columns(2)
            with c[0]:
                ui.show_fig(fp.fig_field(npz, "xz", ol, f"Längsschnitt durch die Kerne, λ = {withf[0]['lam_nm']:g} nm"), "ports_xz.png", "p3_dl_xz")
            with c[1]:
                ui.show_fig(fp.fig_field(npz, "xy", ol, "Querschnitt bei halber Länge"), "ports_xy.png", "p3_dl_xy")
            st.caption("|E|² der Lösung mit Anregung des gewählten Kanals (Port und Mode in Reiter 2), Skala beliebig. Weiß gestrichelt: Kernkanten.")
    with sub[3]:
        df = fp.results_dataframe(ok, meta.get("device"), m["ports"])
        c = st.columns(2)
        c[0].download_button("Tabelle (CSV)", ui.csv_bytes(df), file_name="ports_3d.csv", mime="text/csv", key="p3_dl_csv")
        c[1].download_button("Modell (JSON)", json.dumps(m, indent=1, ensure_ascii=False).encode("utf-8"), file_name="wellenleiter_3d.json", mime="application/json", key="p3_dl_model")
