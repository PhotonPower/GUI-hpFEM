# Architektur

```
 Streamlit-App (fem_app.py)                          hpfem-Python (MSYS2 o. ä.)
 ┌──────────────────────────────┐   job.json          ┌─────────────────────────┐
 │ Modell (Dict, JSON)          │   mesh.msh          │ fem_worker.py           │
 │ fem_geometry: Vorschau,      │ ──────────────────► │  hpfem.ConicalScattering │
 │   Prüfung, Gmsh-Netz         │                     │  hpfem.Scattering2D      │
 │ fem_materials: ε(λ)          │   subprocess        │  AdaptiveMesh2D (hp)     │
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
| `fem_run.py` | Durchlaufwerte, Job-Datei, Start des Workers als Hintergrundprozess, Log- und Fortschrittsauswertung. |
| `fem_worker.py` | Läuft im hpfem-Python. Liest `job.json` und `mesh.msh`, rechnet je Durchlaufpunkt, schreibt Ergebnisse nach jedem Punkt. |
| `fem_post.py` | Ergebnistabelle, abgeleitete Felder (|E|, Q), alle Abbildungen. |

## Schnittstelle App ↔ Worker

* Der App-Prozess schreibt `<Arbeitsordner>/run/job.json` (Modell, Schichtlage, Beleuchtung, Durchlauf, Löser, Karten, Konvergenzstudie) und `mesh.msh` (Gmsh 4.1, ASCII).
* Der Worker schreibt `log.txt` mit den Zeilen `PROGRESS i/n`, `STEP s/S`, `WARN …`, `ERROR …` und `DONE`, danach `results.json` (nach jedem Punkt aktualisiert), `maps_<i>.npz` (Raster), `tri_<i>.npz` (Elemente, exakt) und `hpmesh_<i>.npz` (Endnetz der hp-Adaptivität).
* Ein erneutes Ausführen der Streamlit-Seite beendet den Worker nicht: die App findet ihn über das Prozessobjekt und die Logdatei wieder.

## Koordinaten und Konventionen

x entlang der Periode, y senkrecht zu den Schichten, z entlang der invarianten Richtung. Einheiten: nm im Modell, SI im Löser. Zeitabhängigkeit exp(−iωt), Verlust bedeutet Im ε > 0. Einfallende ebene Welle mit |E₀| = 1 V/m. Einzelheiten stehen im Reiter „Info“ der App.

## Kompatibilität mit hp-FEM

Der Worker nutzt neuere Funktionen, wenn vorhanden (`AdaptiveMesh.set_periodic`, `problem.sample`, `problem.triangulate`, `absorbed_power_by_tag`) und fällt sonst auf Schleife und Rasterintegration zurück. Das Protokoll nennt dann den Grund.
