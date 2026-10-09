# Architektur

```
 Streamlit-App (fem_app.py)                          hpfem-Python (MSYS2 o. ä.)
 ┌──────────────────────────────┐   job.json          ┌─────────────────────────┐
 │ Modell (Dict, JSON)          │   mesh.msh          │ fem_worker.py           │
 │ fem_geometry: Vorschau,      │ ──────────────────► │  hpfem.grating (0.4+)   │
 │   Prüfung, Gmsh-Netz         │                     │  hpfem.ConicalScattering│
 │ fem_materials: ε(λ)          │   subprocess        │  Scattering2D, hp-Netz  │
 │ fem_run: Job, Prozess, Log   │ ──────────────────► │                         │
 │ fem_post: Tabellen, Bilder   │ ◄────────────────── │ results.json, maps_*.npz │
 └──────────────────────────────┘   log.txt (PROGRESS)│ tri_*.npz, hpmesh_*.npz │
                                                      └─────────────────────────┘
```

## Module (`fem_gui/`)

| Datei | Aufgabe |
|---|---|
| `fem_app.py` | Streamlit-Seite mit fünf Reitern (Modell, Netz, Rechnung, Ergebnisse, Info) und Seitenleiste. Prüft beim Start, dass alle Module zur gleichen Version gehören. |
| `fem_geometry.py` | Modell (Schichten, Formen, Beleuchtung, Gebiet), Vorlagen, Validierung, PML-Auslegung, Gmsh-Netz (physikalische Gruppen für den hpfem-Reader), Vorschau und Netzbilder. |
| `fem_materials.py` | Optische Konstanten ohne hpfem: Bibliothek (CSV aus `python/hpfem/data` und Sellmeier), n+ik, ε, Drude, Tabelle. Gleiche Werte wie `hpfem.materials`. |
| `fem_run.py` | Durchlaufwerte, Job-Datei, Start des Workers als Hintergrundprozess (PYTHONPATH nur für einen Quell-Build mit gebautem Modul), Abbruchdatei, Prüflauf (`--check`), Bibliotheksprüfung (Version, Funktionen), Log- und Fortschrittsauswertung. |
| `fem_worker.py` | Läuft im hpfem-Python. Liest `job.json` und `mesh.msh`, rechnet je Durchlaufpunkt (Streuung oder Resonanzen), schreibt Ergebnisse nach jedem Punkt. Mit `--check` nur Prüfungen und Speicherschätzung. |
| `fem_ui.py` | Gemeinsame Streamlit-Bausteine beider Modi (Materialeditor, Tabellen, Bilder mit Download, CSV). |
| `fem_axi.py` | Rotationskörper ohne hpfem: Modell (Teile im Querschnitt r ≥ 0, z; Substrat; Emitter; Resonanzsuche), Vorlagen (Mikrosäule, Kugel, Au-Nanokugel, Mikroscheibe), DBR-Generator, Prüfungen, Layout mit PML und Messebenen, Gmsh-Meridiannetz (Achse Tag 90, Wand Tag 91, Emitterkasten 101 + Materialindex), Abbildungen. |
| `fem_axi_app.py` | Seite des Modus „Rotationskörper“ (fünf Reiter), von `fem_app.py` aufgerufen. |
| `fem_axi_worker.py` | Läuft im hpfem-Python: `AxisymmetricResonance` (Moden der Ordnung m) bzw. `AxisymmetricScattering` mit `axisymmetric_gaussian_dipole` (Purcell-Faktor über `axisymmetric_poynting_flux` durch die Fläche um den Emitterkasten, β durch die Messebenen); Felder über `FieldExporter2D` (ASCII-VTU) als `mode_<k>.npz` / `field_<i>.npz`. |
| `fem_post.py` | Ergebnistabelle, abgeleitete Felder (|E|, Q, |H|, S), Ableitungen, Resonanzen, Bilanz, Zeit, alle Abbildungen. |

## Schnittstelle App ↔ Worker

* Der App-Prozess schreibt `<Arbeitsordner>/run/job.json` (Modell, Schichtlage, Beleuchtung, Durchlauf, Löser, Karten, Konvergenzstudie) und `mesh.msh` (Gmsh 4.1, ASCII).
* `job.json` enthält außerdem `task` (`scattering` oder `resonances`) und `resonance` (`num_modes`, `krylov_dimension`); `solver.engine` ist `grating`, `grating_hp`, `conical`, `inplane` oder `inplane_hp`, dazu `jacobian`, `scalar`, `check` und unter `adaptive` `estimator` (`residual`/`dwr`) und `mirror_periodic`.
* Der Worker schreibt `log.txt` mit den Zeilen `PROGRESS i/n`, `STEP s/S`, `PHASE name`, `ESTIMATE …`, `DIAG severity code: text`, `WARN …`, `ERROR …`, `CANCELLED` und `DONE`, danach `results.json` (nach jedem Punkt aktualisiert; bei `grating` zusätzlich `diagnostics`, `timing`, `flux_balance`, `A_exact`, `amplitudes`, `jacobian`), `maps_<i>.npz` (Raster, mit `H` und `S`), `tri_<i>.npz` (Elemente, exakt, mit `values_H` und `values_S`), `hpmesh_<i>.npz` (Endnetz der hp-Adaptivität) und bei Resonanzen `mode_<i>_<m>.npz`.
* Abbruch: die App legt `<Rechenordner>/cancel` an; der Worker fragt die Datei zwischen den Phasen eines Lösungsschritts (Fortschritts-Callback von hpfem) und zwischen den Punkten ab und endet mit `CANCELLED`.
* Ein erneutes Ausführen der Streamlit-Seite beendet den Worker nicht: die App findet ihn über das Prozessobjekt und die Logdatei wieder.

## Koordinaten und Konventionen

x entlang der Periode, y senkrecht zu den Schichten, z entlang der invarianten Richtung. Einheiten: nm im Modell, SI im Löser. Zeitabhängigkeit exp(−iωt), Verlust bedeutet Im ε > 0. Einfallende ebene Welle mit |E₀| = 1 V/m. Einzelheiten stehen im Reiter „Info“ der App.

## Kompatibilität mit hp-FEM

Das Gmsh-Netz trägt die Ränder als `hpfem.box_tag` (1 links, 2 rechts, 3 unten, 4 oben), damit `hpfem.grating` es direkt nimmt; die Materialflächen sind 1 bis N.

Die Löser `grating` und `grating_hp`, Resonanzen und der Prüflauf brauchen hp-FEM ab 0.4 (`hpfem.grating`), die Ableitungen den Stand mit `grating.jacobian` (M16 S1). Die App fragt die Bibliothek beim Start ab (`fem_run.probe_library`) und bietet nur an, was vorhanden ist. Der klassische Löser nutzt neuere Funktionen, wenn vorhanden (`AdaptiveMesh.set_periodic`, `problem.sample`, `problem.triangulate`, `absorbed_power_by_tag`, `sample(quantity="H"/"S")`) und fällt sonst auf Schleife und Rasterintegration zurück. Das Protokoll nennt dann den Grund.
