# GUI-hpFEM: FEM-Modellwerkstatt

[![CI](https://github.com/PhotonPower/GUI-hpFEM/actions/workflows/ci.yml/badge.svg)](https://github.com/PhotonPower/GUI-hpFEM/actions/workflows/ci.yml)
[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-blue.svg)](LICENSE)

Streamlit-Oberfläche für die Bibliothek [**hp-FEM**](https://github.com/PhotonPower/hp-FEM) (hp-Finite-Elemente für die Nano-Optik).
Periodische Struktur eingeben, mit Gmsh vernetzen, mit hp-FEM rechnen, Spektren, Beugungsordnungen und Feldkarten ansehen.
Es sind keine BEM- oder RCWA-Daten nötig.

## Funktionen

* **Modell:** Materialbibliothek (Si, Ag, Au, Al, GaAs, MAPbI3, SiO2, TiO2, Wasser, Luft) sowie n+ik, ε, Drude und eigene Tabellen; Schichtaufbau; Formen (Rechteck, Trapez, Kreis, Ellipse, Polygon); TE/TM, θ, φ, λ und Durchläufe; Vorlagen und Prüfmeldungen.
* **Netz:** Gmsh-Vernetzung nach Wellenlänge und Eindringtiefe, periodische Ränder, gekrümmte Elemente, PML-Auslegung.
* **Rechnung:** konischer Löser (TE/TM/Azimut), In-Ebenen-Löser gleichmäßig oder **hp-adaptiv**, Konvergenzstudie in p, Hintergrundprozess mit Fortschritt und Abbruch.
* **Ergebnisse:** Spektren, Beugungsordnungen (R, T, A), Feldkarten (exakt auf den Elementen), Schnitte, Absorption je Material, hp-Adaptivität, CSV-Export.

Ausführliche Bedienung: [docs/ANLEITUNG.md](docs/ANLEITUNG.md). Aufbau des Codes: [docs/ARCHITEKTUR.md](docs/ARCHITEKTUR.md).

## Voraussetzungen

Die App und die Rechnung laufen in **zwei getrennten Python-Umgebungen**:

| Teil | Python | Pakete |
|---|---|---|
| App (Oberfläche, Gmsh, Auswertung) | beliebiges Python ≥ 3.10 | `pip install -r requirements.txt` |
| Rechnung (Worker) | das Python, in dem `hpfem` gebaut ist (unter Windows z. B. MSYS2) | `hpfem` aus [hp-FEM](https://github.com/PhotonPower/hp-FEM), Version mit `ConicalScattering` (empfohlen: M15 mit `AdaptiveMesh.set_periodic`, `problem.sample`, `problem.triangulate`) |

Die App startet den Worker als eigenen Prozess; sie selbst braucht `hpfem` nicht.

## Installation und Start

```bash
git clone https://github.com/PhotonPower/GUI-hpFEM.git
git clone https://github.com/PhotonPower/hp-FEM.git      # daneben, Python-Modul bauen (siehe dessen README)
cd GUI-hpFEM
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run fem_gui/fem_app.py
```

Unter Windows genügt ein Doppelklick auf `start_windows.bat`, unter Linux/macOS `./start.sh`.

In der Seitenleiste einstellen:

* **Python mit hpfem**, z. B. `C:/msys64/ucrt64/bin/python3.exe`
* **hp-FEM-Ordner** (enthält `python/`). Vorbelegt wird `../hp-FEM`, oder der Ordner aus der Umgebungsvariable `HPFEM_REPO`.
* Arbeitsordner und Threads.

Mit der Vorlage **„Dünnschicht SiO2 auf Si“** ist ein erster Test des Lösers möglich (R und T gegen den ebenen Stapel).

## Entwicklung

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

Die Tests brauchen weder `hpfem` noch ein funktionierendes Gmsh. Siehe [CONTRIBUTING.md](CONTRIBUTING.md) und [CHANGELOG.md](CHANGELOG.md).

## Lizenz

MIT, siehe [LICENSE](LICENSE). Die optischen Daten der Bibliothek stammen aus der refractiveindex.info-Datenbank (CC0) über hp-FEM.
