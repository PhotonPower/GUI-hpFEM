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
            m["length_nm"] = _num(c[2], "Länge (nm)", f"p3_l_{ver}", m["length_nm"], 100.0, 100.0, 200000.0, "Länge des Bauteils zwischen den Ports (z).")
            if m["device"] == "coupler":
                c = st.columns(2)
                m["gap_nm"] = _num(c[0], "Spalt zwischen den Kernen (nm)", f"p3_g_{ver}", m["gap_nm"], 10.0, 10.0, 5000.0)
                m["ports"] = c[1].radio("Ports", list(fp.PORT_STYLES), index=list(fp.PORT_STYLES).index(m["ports"]), format_func=lambda k: fp.PORT_STYLES[k].split(" (")[0],
                                        key=f"p3_pst_{ver}", help="Halbflächen: " + fp.PORT_STYLES["half"] + ". Supermoden: " + fp.PORT_STYLES["full"] + ".")
            c = st.columns(2)
            m["margin_x_nm"] = _num(c[0], "Rand seitlich (nm)", f"p3_mx_{ver}", m["margin_x_nm"], 50.0, 50.0, 20000.0, "Mantel neben den Kernen bis zur PEC-Wand (x).")
            m["margin_y_nm"] = _num(c[1], "Rand oben und unten (nm)", f"p3_my_{ver}", m["margin_y_nm"], 50.0, 50.0, 20000.0, "Mantel unter und über den Kernen bis zur PEC-Wand (y).")
            st.subheader("Materialien")
            lam0 = fp.wavelengths(m)[0]
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
        if m["device"] == "file":
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


# ===================================================================================================================== results
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
