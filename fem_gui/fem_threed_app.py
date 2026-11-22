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
import fem_ports3d_app as pa3
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
- Nicht in diesem Modus: 3D-Wärme und -Mechanik (nur 2D, Modus „Wärme und Mechanik“).

**Substrat und Schichtstapel (Streuung, Gmsh-Geometrie).** Die Hintergrundlösung ist ein analytischer Schichtstapel (`LayerStack3D`: Einfallsmedium oben, endliche Schichten, Substrat unten;
Fresnel/Airy) mit ebener Welle unter beliebigem Einfallswinkel (s oder p, Azimut); das Objekt streut nur die Abweichung vom Stapel (Streufeld-Formulierung). Die Grenzflächen sind Ebenen im
Gmsh-Netz (jede Schicht ein eigenes Volumen), die Oberkante des Stapels liegt bei z = 0. Die Tabelle zeigt R, T, A des Stapels ohne Objekt. Der Querschnitt ist auf die **Intensität der einfallenden
Welle** bezogen, Q auf die Projektion des Objekts auf die x-y-Ebene. Grenzen: das PML ist für den Index 1 ausgelegt (Substrat und Schichten höher: nur Richtwerte, die Bibliothek meldet
ein unteraufgelöstes PML im Protokoll); der Querschnitt aus dem Fluss um das Objekt ist an lossy Schichten, die das Objekt berühren, nicht eindeutig; ein Objekt in einer Schicht (das eine Grenzfläche schneidet) gibt es nicht.
Geprüft: ein „Stapel“ aus Luft liefert dieselbe Kugel wie ohne Stapel (Q_sca 0,811 gegen 0,795, grobes Netz; Mie 0,797); R des Stapels stimmt mit der Fresnel-Formel überein.

**Gmsh-Netze.** (1) *Gmsh-Geometrie*: Kugel, Zylinder (Achse z) oder Quader in einer Box mit PML-Rand, vernetzt von Gmsh (Python-Modul `gmsh` im hpfem-Python nötig), Elemente zweiter Ordnung (gekrümmt);
Maschenweite am Objekt und fern, Übergangslänge, Innenraum, PML-Dicke und Abstand nach oben sind einstellbar, „Maschenweiten vorschlagen“ setzt Richtwerte aus der Wellenlänge.
**Das PML braucht mindestens zwei bis drei Zellen Dicke**: an einer Kugel n = 2, ka = 1 liegt Q_sca bei PML = 2 Zellen, fern λ/4,2 und p = 2 (18 000 Unbekannte, 82 s) um 0,3 % neben der Mie-Reihe,
bei einer Zelle im PML um 25 %. (2) *Eigene Datei*: ein Gmsh-Netz (Version 4.1 ASCII) wird hochgeladen, „Datei lesen“ zeigt Zellmarken (physikalische Volumina), Flächenmarken und
Ausdehnung; Sie weisen jeder Zellmarke ein Material zu und wählen die Zellmarken des Streukörpers. Die äußeren Randflächen ohne Marke werden zu den Seiten der Box (PEC), das PML liegt in der
äußersten Schicht der Box (Dicke einstellbar). Q ist auf die Projektion des Objekts bezogen (aus der Oberfläche mit geraden Facetten berechnet, bei gekrümmten Körpern ungenau); σ ist davon unabhängig.

**hp-adaptive Verfeinerung (Streuung).** Je Schritt: lösen, residualer Fehlerschätzer (`Scattering3D.estimate`), Dörfler-Markierung (θ), Entscheidung h oder p nach der Vorhersage, Verfeinern
(`AdaptiveMesh3D`, `hp_refine`). Die Schleife endet bei der Schrittzahl, der Obergrenze für Unbekannte, der Speichergrenze oder dem relativen Schätzer η/η₀. Ergebnis: η, Q_sca, Q_abs und Zellzahl je Schritt,
Netz mit den Ordnungen. η fällt mit den Schritten, aber Q_sca konvergiert bei grobem PML nicht monoton; die Schleife ersetzt keine Konvergenzprüfung gegen eine Referenz.

**Wellenleiter und Ports.** Rechteckkern im Mantel (`Streifenwellenleiter`) oder zwei Kerne (`Richtkoppler`) zwischen modalen Ports (`s_parameters`): PEC-Wände an den vier Seiten, Ports auf den Stirnflächen
z = 0 und z = L; strukturiertes Tetraedernetz mit Gitterlinien auf den Kernkanten. Ergebnis: leistungsnormierte S-Matrix aller Kanäle, Reflexion, Transmission, Verlust, n_eff der Portmoden, |E|² im Längs- und
Querschnitt. Referenzen: Streifen: β der Grundmode des Querschnitts (2D-Modenlöser) und seine Phase gegen die der Transmission; Koppler: Theorie gekoppelter Moden P_cross = sin²(κL) mit κ aus den Supermoden
(gleiches Netz und ein feineres Referenznetz mit p + 2 und halber Zelle: der Unterschied zeigt den Diskretisierungsfehler). Eigene Gmsh-Datei: Flächenmarken werden als PEC-Wand oder Port (mit Modenzahl) zugewiesen,
z. B. der Ringresonator `examples/ring_resonator_3d/ring_mesh.py` der Bibliothek. **3D-Wellenleiter sind teuer**: n_eff des 450 × 220 nm Si/SiO₂-Streifens ist mit p = 1 und 150-nm-Zellen um
9 % zu klein (2,14 gegen etwa 2,36), mit p = 2 und 75-nm-Zellen konvergiert es, kostet aber Minuten und Gigabyte; die Phase der Transmission läuft mit der Länge davon.
"""


def _init():
    if "td_model" in S:
        ft.ensure(S.td_model)
        return
    S.td_model = copy.deepcopy(ft.default_model())
    S.td_ver, S.td_proc, S.td_dir, S.td_msg, S.td_est = 0, None, None, None, None


def _bump():
    S.td_ver += 1


def render_threed(ctx):
    """ctx: py, repo, work (Path), threads, lib, FEATS."""
    _init()
    work = Path(ctx["work"]) / "threed"
    st.title("FEM-Modellwerkstatt: 3D-Strukturen")
    st.caption("Streuung (Kugel, Würfel, Zylinder, Quader, auch über Schichtstapeln, hp-adaptiv), Resonanzen, Bandstruktur eines kubischen Kristalls, Wellenleiter und Ports. "
               "hp-FEM: Scattering3D, Resonance3D, BandStructure3D, s_parameters, AdaptiveMesh3D, LayerStack3D; Netze: Bibliothek, Gmsh, eigene Datei.")
    if S.td_proc is not None:
        return _running()
    if not (ctx["FEATS"].get("threed") or ctx["lib"].get("error")):
        st.warning("Das eingestellte Python hat kein hpfem mit den 3D-Lösern. Anzeigen geht trotzdem.")
    elif not (ctx["FEATS"].get("threed_ext") or ctx["lib"].get("error")):
        st.warning("Dem hpfem fehlen Schichtstapel, hp-Adaptivität oder Ports in 3D (ältere Version): diese Aufgaben schlagen im Worker fehl.")
    if ctx["FEATS"].get("gmsh") is False and not ctx["lib"].get("error"):
        st.warning("Im hpfem-Python fehlt das Modul `gmsh` (pip install gmsh): Gmsh-Geometrien gehen nicht; Bibliotheksnetze, hochgeladene .msh-Dateien und Wellenleiter schon.")
    t1, t2, t3, t4 = st.tabs(["1 Modell", "2 Rechnung", "3 Ergebnisse", "4 Info"])
    with t1:
        _tab_model(ctx, work)
    with t2:
        _tab_run(ctx, work)
    with t3:
        _tab_results(work)
    with t4:
        st.markdown(INFO)


def _num(col, label, key, value, step=0.1, lo=0.0, hi=1e6, help=None, fmt="%.5g"):
    return float(col.number_input(label, lo, hi, float(value), step, key=key, format=fmt, help=help))


# ===================================================================================================================== model
def _rename(m, old, new):
    m["materials"] = {(new if k == old else k): v for k, v in m["materials"].items()}
    if m["object"] == old:
        m["object"] = new
    st_ = m["stack"]
    for key in ("incidence", "substrate"):
        if st_[key] == old:
            st_[key] = new
    for layer in st_["layers"]:
        if layer["material"] == old:
            layer["material"] = new
    f = m["file"]
    f["tag_material"] = {t: (new if nm == old else nm) for t, nm in f["tag_material"].items()}


def _head(m, ver):
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


def _tab_model(ctx, work):
    m, ver = S.td_model, S.td_ver
    _head(m, ver)
    if m["task"] == "ports":
        return pa3.tab_model(m["p3"], ver, _bump, ctx, work)
    left, right = st.columns([6, 5])
    with left:
        if m["task"] == "bands":
            m["shape"] = st.radio("Form", list(ft.SHAPES), index=list(ft.SHAPES).index(m["shape"]), format_func=ft.SHAPES.get, key=f"td_shape_{ver}", horizontal=True)
            b = m["bands"]
            c = st.columns(3)
            b["radius"] = _num(c[0], "Radius bzw. halbe Kante (a)", f"td_br_{ver}", b["radius"], 0.05, 0.02, 0.49, "Einheiten der Gitterkonstante a; die Zelle ist [−0,5, 0,5]³.")
            b["eps"] = _num(c[1], "ε des Objekts", f"td_be_{ver}", b["eps"], 0.5, 0.01, 400.0)
            b["eps_bg"] = _num(c[2], "ε des Hintergrunds", f"td_bb_{ver}", b["eps_bg"], 0.5, 0.01, 400.0)
        else:
            m["mesh_kind"] = st.radio("Netz", list(ft.MESH_KINDS), index=list(ft.MESH_KINDS).index(m["mesh_kind"]), format_func=ft.MESH_KINDS.get, key=f"td_mk_{ver}",
                                      help="Bibliotheksnetz: Kugel (gekrümmt) oder Würfel in einer PML-Box, ohne Gmsh. Gmsh-Geometrie: Kugel, Zylinder oder Quader, wahlweise über einem "
                                           "Schichtstapel. Eigene Datei: ein Gmsh-Netz mit Tabelle Zellmarke → Material.")
            if m["mesh_kind"] == "builtin":
                m["shape"] = st.radio("Form", list(ft.SHAPES), index=list(ft.SHAPES).index(m["shape"]), format_func=ft.SHAPES.get, key=f"td_shape_{ver}", horizontal=True)
                m["radius_nm"] = _num(st, "Radius bzw. halbe Kante (nm)", f"td_r_{ver}", m["radius_nm"], 5.0, 1.0, 1e6)
            elif m["mesh_kind"] == "gmsh":
                _gmsh_geometry(m, ver)
            else:
                _file_import(m, ver, ctx, work)
            st.subheader("Materialien")
            lam0 = ft.lam_check(m)[0] if ft.lam_check(m) else 500.0
            used = {m["object"]} | set(ft.stack_regions(m) if m["stack"]["enabled"] else []) | set(m["file"]["tag_material"].values())
            ui.material_editor(m, ver, lam0, used, _rename, _bump, prefix="td_")
            names = list(m["materials"])
            if m["mesh_kind"] != "file":
                m["object"] = st.selectbox("Objekt besteht aus", names, index=names.index(m["object"]) if m["object"] in names else 0, key=f"td_obj_{ver}",
                                           help="Umgebung ist Luft (ε = 1), beim Schichtstapel das Einfallsmedium. Die Permittivität wird bei jeder Wellenlänge des Durchlaufs ausgewertet "
                                                "(Resonanzen: bei der Zielwellenlänge).")
            if m["task"] == "scatter" and m["mesh_kind"] == "gmsh":
                _stack_editor(m, ver)
    with right:
        for lvl, text in ft.validate(m):
            (st.error if lvl == "error" else st.warning)(text)
        st.markdown("**Aufbau.** " + _aufbau(m))


def _aufbau(m):
    if m["task"] == "bands":
        return "Einfach kubisches Gitter, Gitterkonstante a = 1, Zelle [−½, ½]³, ein Objekt in der Mitte; Pfad Γ–X–M–R–Γ."
    base = "Ebene Welle x-polarisiert längs +z. " if not m["stack"]["enabled"] else "Ebene Welle von oben auf den Schichtstapel (Einfallswinkel, s oder p wählbar). "
    if m["mesh_kind"] == "gmsh":
        g = m["gmsh"]
        what = {"sphere": "Kugel", "cylinder": "Zylinder (Achse z)", "block": "Quader"}[g["kind"]]
        pos = " über dem Stapel (Oberkante z = 0)" if m["stack"]["enabled"] and m["task"] == "scatter" else " im Ursprung"
        return f"{what}{pos}, Box mit PML-Rand, Netz von Gmsh (Elemente zweiter Ordnung, gekrümmt). " + base + "Das Netz sehen Sie nach „Netz prüfen“ im Reiter 2."
    if m["mesh_kind"] == "file":
        return "Ihr Gmsh-Netz; die äußeren Randflächen werden zu den Seiten der Box (PEC), die PML-Schicht liegt im äußersten Bereich der Box. " + base
    return "Das Objekt sitzt im Ursprung einer Box mit PML-Rand. " + base + "Das Netz sehen Sie nach „Netz prüfen“ im Reiter 2."


def _gmsh_geometry(m, ver):
    g = m["gmsh"]
    g["kind"] = st.radio("Körper", list(ft.gm.KINDS), index=list(ft.gm.KINDS).index(g["kind"]), format_func=ft.gm.KINDS.get, key=f"td_gk_{ver}", horizontal=True)
    c = st.columns(3)
    if g["kind"] in ("sphere", "cylinder"):
        m["radius_nm"] = _num(c[0], "Radius (nm)", f"td_r_{ver}", m["radius_nm"], 5.0, 1.0, 1e6)
    if g["kind"] == "cylinder":
        g["height_nm"] = _num(c[1], "Höhe (nm)", f"td_gh_{ver}", g["height_nm"], 5.0, 1.0, 1e6, "Achse längs z.")
    if g["kind"] == "block":
        g["lx_nm"] = _num(c[0], "Länge x (nm)", f"td_glx_{ver}", g["lx_nm"], 5.0, 1.0, 1e6)
        g["ly_nm"] = _num(c[1], "Länge y (nm)", f"td_gly_{ver}", g["ly_nm"], 5.0, 1.0, 1e6)
        g["lz_nm"] = _num(c[2], "Höhe z (nm)", f"td_glz_{ver}", g["lz_nm"], 5.0, 1.0, 1e6)


def _stack_editor(m, ver):
    s = m["stack"]
    with st.expander("Substrat und Schichtstapel (LayerStack3D)", expanded=bool(s["enabled"])):
        s["enabled"] = st.checkbox("Objekt über einem Schichtstapel", value=bool(s["enabled"]), key=f"td_st_on_{ver}",
                                   help="Die Hintergrundlösung ist der analytische Schichtstapel (Fresnel / Airy); das Objekt streut nur die Abweichung. Die Grenzflächen sind Ebenen im "
                                        "Gmsh-Netz. Die Oberkante des Stapels liegt bei z = 0, das Objekt darüber (Abstand), der Stapel nach unten: Schichten, dann das Substrat.")
        if not s["enabled"]:
            return
        names = list(m["materials"])

        def pick(label, key, cur):
            return st.selectbox(label, names, index=names.index(cur) if cur in names else 0, key=key)

        s["incidence"] = pick("Einfallsmedium oben (verlustfrei)", f"td_st_inc_{ver}", s["incidence"])
        for k, layer in enumerate(s["layers"]):
            c = st.columns([3, 2, 1])
            layer["material"] = pick(f"Schicht {k + 1}", f"td_st_lm_{ver}_{k}", layer["material"])
            layer["thickness_nm"] = _num(c[1], "Dicke (nm)", f"td_st_lt_{ver}_{k}", layer["thickness_nm"], 5.0, 0.1, 1e6)
            if c[2].button("Entfernen", key=f"td_st_ld_{ver}_{k}"):
                s["layers"].pop(k)
                _bump()
                st.rerun()
        if st.button("Schicht hinzufügen", key=f"td_st_add_{ver}"):
            s["layers"].append({"material": names[min(1, len(names) - 1)], "thickness_nm": 50.0})
            _bump()
            st.rerun()
        s["substrate"] = pick("Substrat unten", f"td_st_sub_{ver}", s["substrate"])
        c = st.columns(3)
        s["gap_nm"] = _num(c[0], "Abstand Objekt – Oberfläche (nm)", f"td_st_gap_{ver}", s["gap_nm"], 5.0, 0.0, 1e5, "0: das Objekt sitzt auf der Oberfläche (Zylinder, Quader). Kugeln brauchen Abstand.")
        s["depth_nm"] = _num(c[1], "Tiefe des Substrats bis zum PML (nm)", f"td_st_dep_{ver}", s["depth_nm"], 10.0, 1.0, 1e6)
        s["angle_deg"] = _num(c[2], "Einfallswinkel (°)", f"td_st_ang_{ver}", s["angle_deg"], 5.0, 0.0, 89.0)
        c = st.columns(2)
        s["pol"] = c[0].radio("Polarisation", list(ft.POLS), index=list(ft.POLS).index(s["pol"]), format_func=ft.POLS.get, key=f"td_st_pol_{ver}", horizontal=True,
                              help="s: E senkrecht zur Einfallsebene; p: E in der Einfallsebene. Bei senkrechtem Einfall gilt der Azimut (Richtung von E gegen x).")
        s["azimuth_deg"] = _num(c[1], "Azimut der Einfallsebene (°)", f"td_st_az_{ver}", s["azimuth_deg"], 15.0, -360.0, 360.0)
        regs = ft.stack_regions(m)
        st.caption("Aufbau von oben nach unten: " + " → ".join(f"{nm}" + (f" ({s['layers'][k - 1]['thickness_nm']:g} nm)" if 0 < k < len(regs) - 1 else "") for k, nm in enumerate(regs)))


def _file_import(m, ver, ctx, work):
    f = m["file"]
    up = st.file_uploader("Gmsh-Datei (.msh, Version 4.1 ASCII)", type=["msh"], key=f"td_fup_{ver}")
    if up is not None and up.name != f.get("name"):
        folder = Path(work) / "files"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "mesh.msh"
        path.write_bytes(up.getvalue())
        f.update(name=up.name, path=str(path), tags={}, tag_material={}, object_tags=[])
        _bump()
        st.rerun()
    if not f.get("name"):
        st.info("Eine .msh-Datei hochladen. Zellen tragen die Marke der physikalischen Gruppe, im Netz muss die Box mit Luft und PML-Bereich enthalten sein.")
        return
    c = st.columns([1, 1, 2])
    f["scale"] = c[0].selectbox("Einheit der Koordinaten", ["nm", "µm", "mm", "m"], index=["nm", "µm", "mm", "m"].index(f["scale"]), key=f"td_funit_{ver}")
    if c[1].button("Datei lesen", key=f"td_fread_{ver}"):
        info = ft.run_inspect(ctx["py"], ctx["repo"], Path(work) / "files" / "inspect", f["path"], f["scale"], "scatter", int(ctx["threads"]))
        if "error" in info:
            st.error(info["error"])
        else:
            f["tags"] = info
            names = list(m["materials"])
            for tag in info["cells"]:
                f["tag_material"].setdefault(tag, ft.AIR if ft.AIR in names else names[0])
            _bump()
            st.rerun()
    info = f.get("tags")
    if not info:
        return
    ext = [(b - a) / 1e-9 for a, b in zip(info["bbox"][0], info["bbox"][1])]
    st.caption(f"{info['n_cells']} Zellen, Geometrieordnung {info.get('geometry_order', '?')}, Ausdehnung {ext[0]:.0f} × {ext[1]:.0f} × {ext[2]:.0f} nm; Flächenmarken: "
               + (", ".join(f"{t} ({n})" for t, n in info["facets"].items()) or "keine"))
    names = list(m["materials"])
    obj = []
    for tag, count in info["cells"].items():
        c = st.columns([3, 2])
        cur = f["tag_material"].get(tag, names[0])
        f["tag_material"][tag] = c[0].selectbox(f"Zellmarke {tag} ({count} Zellen): Material", names, index=names.index(cur) if cur in names else 0, key=f"td_ftm_{ver}_{tag}")
        if c[1].checkbox("Streukörper", value=int(tag) in f["object_tags"], key=f"td_fto_{ver}_{tag}", help="Um die Zellen mit dieser Marke wird die Fläche für die Wirkungsquerschnitte gelegt."):
            obj.append(int(tag))
    f["object_tags"] = obj
    f["pml_nm"] = _num(st, "PML-Dicke am Rand der Box (nm)", f"td_fpml_{ver}", f["pml_nm"], 10.0, 1.0, 1e6, "Das PML liegt im äußersten Bereich des Netzes; es muss mit mindestens zwei bis drei Zellen vernetzt sein.")


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


def _adaptive_params(m, ver):
    a = m["adaptive"]
    a["enabled"] = st.checkbox("hp-adaptive Verfeinerung (AdaptiveMesh3D)", value=bool(a["enabled"]), key=f"td_ad_on_{ver}",
                               help="Je Schritt: lösen, residualer Fehlerschätzer, Dörfler-Markierung, Entscheidung h oder p nach der Vorhersage, Verfeinern. Das Netz wächst, bis die Schrittzahl, "
                                    "die Obergrenze für Unbekannte oder der Speicher erreicht ist. Querschnitte und Felder sind die des letzten Netzes.")
    if a["enabled"]:
        c = st.columns(4)
        a["max_steps"] = int(c[0].number_input("Schritte", 1, 12, int(a["max_steps"]), key=f"td_ad_n_{ver}"))
        a["theta"] = _num(c[1], "Dörfler-Anteil θ", f"td_ad_t_{ver}", a["theta"], 0.05, 0.05, 1.0, "Anteil des quadratischen Gesamtfehlers, den die markierten Zellen tragen.")
        a["max_dofs"] = int(c[2].number_input("Obergrenze Unbekannte", 100, 400000, int(a["max_dofs"]), 1000, key=f"td_ad_d_{ver}", help="Ein weiterer Schritt wird nicht mehr gerechnet, wenn das nächste Netz größer wäre."))
        a["tol"] = _num(c[3], "Toleranz η/η₀ (0 = aus)", f"td_ad_tol_{ver}", a["tol"], 0.05, 0.0, 1.0, "Abbruch, wenn der Schätzer unter diesen Bruchteil seines Anfangswertes fällt.")
        st.caption("Die Startordnung p ist die kleinste Ordnung; die Schritte erhöhen p dort, wo die Lösung glatt ist, und teilen Zellen dort, wo sie es nicht ist.")


def _gmsh_mesh_params(m, ver):
    g = m["gmsh"]
    c = st.columns(3)
    g["hw_nm"] = _num(c[0], "Innenraum, halbe Breite (nm)", f"td_ghw_{ver}", g["hw_nm"], 10.0, 10.0, 1e6, "Bis hierher reicht die Box um das Objekt (x, y); danach beginnt das PML.")
    g["pml_nm"] = _num(c[1], "PML-Dicke (nm)", f"td_gpml_{ver}", g["pml_nm"], 10.0, 10.0, 1e6, "Mindestens zwei bis drei ferne Zellen, sonst meldet die Bibliothek ein unteraufgelöstes PML und das Ergebnis ist falsch.")
    g["clear_nm"] = _num(c[2], "Abstand über dem Objekt (nm)", f"td_gcl_{ver}", g["clear_nm"], 10.0, 10.0, 1e6, "Vom Objekt (bzw. der Oberkante) bis zum PML nach oben.")
    c = st.columns(4)
    g["h_obj_nm"] = _num(c[0], "Zelle am Objekt (nm)", f"td_gho_{ver}", g["h_obj_nm"], 5.0, 1.0, 1e5)
    g["h_far_nm"] = _num(c[1], "Zelle fern (nm)", f"td_ghf_{ver}", g["h_far_nm"], 5.0, 5.0, 1e5, "In Medien mit Index n wird die Zelle durch n geteilt (Stapel).")
    g["grade_nm"] = _num(c[2], "Übergangslänge (nm)", f"td_ggr_{ver}", g["grade_nm"], 10.0, 1.0, 1e6, "Abstand vom Objekt, über den die Zelle von „am Objekt“ auf „fern“ wächst.")
    g["order"] = int(c[3].number_input("Geometrieordnung", 1, 2, int(g["order"]), key=f"td_gord_{ver}", help="2: gekrümmte Elemente (die Kugel wird exakt quadratisch approximiert)."))
    if st.button("Maschenweiten vorschlagen", key=f"td_gsug_{ver}", help="Zelle am Objekt ≈ min(r/2,5; λ/(3n)), fern ≈ λ/2,5, PML ≈ λ/2, Innenraum ≈ 1,5 r (aus der kürzesten Wellenlänge)."):
        g.update(ft.suggest_sizes(m))
        _bump()
        st.rerun()
    st.caption(f"PML = {g['pml_nm'] / max(g['h_far_nm'], 1e-9):.1f} ferne Zellen dick (mindestens 2 bis 3 nötig). Gemessen an einer Kugel n = 2, ka = 1: mit PML = 2 Zellen, Zelle fern λ/4 und p = 2 liegt Q_sca "
               "innerhalb 1 % der Mie-Reihe; mit einer Zelle im PML 25 % daneben.")


def _tab_run(ctx, work):
    m, ver = S.td_model, S.td_ver
    t = m["task"]
    if t == "ports":
        pa3.tab_params(m["p3"], ver)
        c = st.columns(1)
        m["max_gb"] = _num(c[0], "Speichergrenze (GB)", f"td_mg_{ver}", m["max_gb"], 2.0, 0.5, 1000.0, "Der Worker bricht ab, wenn die Schätzung der Bibliothek darüber liegt.")
        m["p3"]["max_gb"] = m["max_gb"]
    elif t == "scatter":
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
    if t != "ports":
        st.subheader("Netz und Ordnung")
        kind = "builtin" if t == "bands" else m["mesh_kind"]
        if kind == "gmsh":
            _gmsh_mesh_params(m, ver)
        c = st.columns(5)
        if t == "bands":
            m["n"] = int(c[0].number_input("Zellen je Radius (n, Kugel)", 1, 12, int(m["n"]), key=f"td_n_{ver}"))
        elif kind == "builtin":
            m["n"] = int(c[0].number_input("Zellen je Radius (n)", 1, 12, int(m["n"]), key=f"td_n_{ver}", help="Bei der Kugel: Zellen auf dem Radius; beim Würfel: Zellen auf der halben Kante."))
            m["half_width"] = _num(c[1], "Innenraum (halbe Breite, in r)", f"td_hw_{ver}", m["half_width"], 0.5, 1.0, 20.0, "Die Box um das Objekt bis zum PML.")
            m["outer"] = _num(c[2], "Außenrand (in r)", f"td_ou_{ver}", m["outer"], 0.5, 1.5, 30.0, "Ende des PML; zwischen Innenraum und Außenrand liegt die Schicht.")
        m["order"] = int(c[3].number_input("Ordnung p", 1, 6, int(m["order"]), key=f"td_p_{ver}"))
        if t == "scatter" and kind == "builtin":
            m["quarter"] = c[4].checkbox("Viertelmodell", value=bool(m["quarter"]), key=f"td_q_{ver}", help=ft.SYM[True] + ": nur für die x-polarisierte Welle längs z; Faktor 4 weniger Speicher.")
        if t == "scatter":
            _adaptive_params(m, ver)
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
            S.td_est = ft.run_estimate(ctx["py"], ctx["repo"], folder, ft.job(m), int(ctx["threads"]), timeout=600)
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
            ui.show_fig(ft.fig_mesh(npz, axis, coord * unit, unit, lab, obj_tags=ft.object_cell_tags(m)), "netz.png", "td_dl_mesh")
            st.caption("Gerade Tetraederkanten; gekrümmte Ränder liegen im Löser exakt. Orange: Zellen des Objekts, übrige Farben: weitere Materialien (Stapel, Marken).")
    st.markdown("---")
    stale = est is not None and S.get("td_est_model") is not None and ft.job(S.td_est_model) != ft.job(m)
    if stale:
        st.caption("Die Schätzung gehört zu einer früheren Einstellung: bitte neu schätzen.")
    if st.button("Rechnen", type="primary", key="td_run", disabled=bool(errs)):
        try:
            work.mkdir(parents=True, exist_ok=True)
            jb = ft.job(m)
            (work / "job.json").write_text(json.dumps(jb, ensure_ascii=False), encoding="utf-8")
            for f in list(work.glob("field_*.npz")) + list(work.glob("mesh_*.npz")) + [work / "results.json"]:
                if f.exists():
                    f.unlink()
            S.td_proc = fr.start_worker(ctx["py"], ctx["repo"], work, int(ctx["threads"]), script=ft.worker_script(jb))
            S.td_dir, S.td_cancel, S.td_run_model = str(work), False, copy.deepcopy(m)
            st.rerun()
        except Exception as exc:
            st.error(f"Start fehlgeschlagen: {exc}")
    if S.get("td_msg"):
        (st.success if S.td_msg[0] == "ok" else st.error)(S.td_msg[1] + (" Die Ergebnisse stehen im Reiter 3." if S.td_msg[0] == "ok" else ""))
    if (work / "log.txt").is_file():
        with st.expander("Protokoll der letzten Rechnung"):
            log = fr.read_log(work, tail=40)
            st.code(log)
            if "PML is under-resolved" in log or "under-resolved" in log:
                st.warning("Die Bibliothek meldet ein unteraufgelöstes PML: das PML braucht mindestens zwei bis drei Zellen Dicke (bei Ordnung 1 mehr) oder ein feineres Netz im PML-Gebiet; "
                           "die Querschnitte sind dann nur Richtwerte.")


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
    elif task == "ports":
        pa3.results(res, work, model)
    else:
        _res_bands(res, model)


def _field_view(npz_path, quarter, model, key_prefix, title):
    npz = np.load(npz_path)
    c = st.columns(2)
    sel = c[0].radio("Ebene", ["xz", "xy"], format_func={"xz": "x–z (y = 0, Welle längs z)", "xy": "x–y (z = 0)"}.get, horizontal=True, key=f"{key_prefix}_pl")
    ui.show_fig(ft.fig_field(npz, sel, quarter, 1e-9, "nm", title=title, outline=[(x, y, s) for x, y, s in ft.object_outline(model, sel)]), f"feld_{sel}.png", f"{key_prefix}_dl")


def _res_scatter(res, work, model, meta):
    pts = res["points"]
    for p in pts:
        if "error" in p:
            st.error(f"λ = {p['lam_nm']:g} nm: {p['error']}")
    ok = [p for p in pts if "q_sca" in p]
    if not ok:
        return
    steps = [p for p in ok if p.get("steps")]
    sub = st.tabs(["Wirkungsquerschnitte", "Felder", "hp-Verfeinerung", "Netz", "Tabelle"])
    with sub[0]:
        if len(ok) > 1:
            ui.show_fig(ft.fig_efficiency(ok), "effizienz.png", "td_dl_eff")
        df = ft.scatter_dataframe(ok)
        ui.table_show(df.round(5))
        if "mie_q_sca" in ok[0]:
            worst = max(abs(p["q_sca"] - p["mie_q_sca"]) / max(abs(p["mie_q_sca"]), 1e-30) for p in ok) * 100
            st.caption(f"Größte Abweichung Q_sca zur Mie-Reihe: {worst:.1f} %. Das ist das Maß für die Netzgüte (Q_abs bei verlustfreier Kugel: Fehler des Flussintegrals).")
        else:
            st.caption("Keine analytische Referenz für diese Geometrie. Konvergenz prüfen, indem Netz und p verfeinert werden (oder die hp-Verfeinerung nutzen).")
        if "stack_R" in ok[0]:
            st.caption("R, T, A des Schichtstapels ohne Objekt (analytisch) stehen in der Tabelle; Q ist auf die Projektion des Objekts auf die x-y-Ebene bezogen, die Querschnitte σ auf die "
                       "Intensität der einfallenden Welle (Poynting-Fluss von |E| = 1 V/m im Einfallsmedium).")
    with sub[1]:
        withf = [p for p in ok if p.get("field_file")]
        if not withf:
            st.info("Keine Feldschnitte gespeichert.")
        else:
            i = st.selectbox("Wellenlänge", range(len(withf)), format_func=lambda i: f"{withf[i]['lam_nm']:g} nm", key="td_sc_lam")
            _field_view(work / withf[i]["field_file"], bool(meta.get("quarter")), model, "td_sc", f"|E|²/|E₀|² bei {withf[i]['lam_nm']:g} nm")
            st.caption("Gestrichelt: Umriss des Objekts. Im Viertelmodell wird das berechnete Viertel an den Symmetrieebenen gespiegelt; die Welle läuft in +z.")
    with sub[2]:
        if not steps:
            st.info("Die hp-adaptive Verfeinerung war nicht eingeschaltet (Reiter 2).")
        else:
            j = st.selectbox("Wellenlänge", range(len(steps)), format_func=lambda j: f"{steps[j]['lam_nm']:g} nm", key="td_ad_lam") if len(steps) > 1 else 0
            ps = steps[j]
            ui.show_fig(ft.fig_convergence(ps["steps"]), "hp_konvergenz.png", "td_dl_conv")
            ui.table_show(ft.steps_dataframe(ps["steps"]).round(5))
            st.caption(f"Abbruch: {ps.get('stop_reason', '—')}. η ist der residuale Fehlerschätzer der Bibliothek (Summe der Elementindikatoren); er fällt mit den Schritten, aber Q_sca "
                       "konvergiert nicht zwingend monoton: bei grobem PML dominiert dessen Diskretisierungsfehler, den der Schätzer nur teilweise sieht. Dörfler-Markierung, "
                       "Entscheidung h/p nach der Vorhersage (Melenk–Wohlmuth).")
    with sub[3]:
        npz_path = work / "mesh_scatter.npz"
        if not npz_path.is_file():
            st.info("Kein Netz gespeichert.")
        else:
            npz = np.load(npz_path)
            by = "tag"
            if "orders" in npz.files and steps:
                by = "order" if st.radio("Färbung", ["Material", "Ordnung p"], horizontal=True, key="td_res_mesh_by") == "Ordnung p" else "tag"
            axis = {"x": 0, "y": 1, "z": 2}[st.radio("Schnittebene senkrecht zu", ["x", "y", "z"], index=1, horizontal=True, key="td_res_mesh_axis")]
            lo, hi = float(npz["vertices"][:, axis].min() * 1e9), float(npz["vertices"][:, axis].max() * 1e9)
            coord = st.slider("Lage der Ebene (nm)", lo, hi, min(max(0.0, lo), hi) if lo <= 0 <= hi else (lo + hi) / 2, (hi - lo) / 100 or 0.01, key=f"td_res_mesh_c_{axis}")
            ui.show_fig(ft.fig_mesh(npz, axis, coord * 1e-9, 1e-9, "nm", obj_tags=ft.object_cell_tags(model), by=by), "netz_ergebnis.png", "td_dl_res_mesh")
    with sub[4]:
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
