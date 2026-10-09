# GUI-hpFEM: FEM-Modellwerkstatt

[![CI](https://github.com/PhotonPower/GUI-hpFEM/actions/workflows/ci.yml/badge.svg)](https://github.com/PhotonPower/GUI-hpFEM/actions/workflows/ci.yml)
[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-blue.svg)](LICENSE)

Streamlit-Oberfläche für die Bibliothek [**hp-FEM**](https://github.com/PhotonPower/hp-FEM) (hp-Finite-Elemente für die Nano-Optik).
Periodische Struktur eingeben, mit Gmsh vernetzen, mit hp-FEM rechnen, Spektren, Beugungsordnungen und Feldkarten ansehen.
Es sind keine BEM- oder RCWA-Daten nötig.

## Funktionen

* **Modell:** Materialbibliothek (Si, Ag, Au, Al, GaAs, MAPbI3, SiO2, TiO2, Wasser, Luft) sowie n+ik, ε, Drude und eigene Tabellen; Schichtaufbau; Formen (Rechteck, Trapez, Kreis, Ellipse, Polygon); TE/TM, θ, φ, λ und Durchläufe; Vorlagen und Prüfmeldungen.
* **Netz:** Gmsh-Vernetzung nach Wellenlänge und Eindringtiefe, periodische Ränder, gekrümmte Elemente, PML-Auslegung.
* **Rechnung:** `hpfem.grating` (TE/TM/konisch) mit den Prüfungen der Bibliothek, skalarem E_z-Pfad, Flussbilanz und **Ableitungen** (Jacobi-Matrix nach ε, λ, θ, φ); **hp-Adaptivität** für TE, TM und konischen Einfall (Residuen- oder zielorientierter DWR-Schätzer); **Resonanzen und Bänder** der offenen Zelle; klassischer konischer Löser und In-Ebenen-Löser; Konvergenzstudie in p; Prüfen mit Speicherschätzung vor dem Rechnen; Hintergrundprozess mit Phasenanzeige und Abbruch zwischen den Phasen.
* **Isolierte Strukturen (neu):** einzelner Draht, Graben, Schlitz oder Stufe im Schichtstapel mit PML links und rechts (statt Bloch-Rändern): Streu-, Absorptions- und Extinktionsbreite (Mie-Vergleich für Zylinder), Fernfeld, Detektoren für den Energiefluss (z. B. Slit-Groove-Benchmark), Feldkarten.
* **Rotationskörper (neu):** Resonatoren und Emitter mit Rotationssymmetrie (Mikrosäule mit Bragg-Spiegeln und Quantenpunkt, Kugeln, Nanopartikel, Scheiben, Ringe) in der Meridianebene: **Resonanzen** (komplexe Frequenz, Güte Q, Modenfelder), **Emitter** auf der Achse (Purcell-Faktor, Abstrahlung nach oben/unten über der Wellenlänge, auch um die Resonanz zentriert, Modenzerlegung per Riesz-Projektion) und **Streuung ebener Wellen** an Partikeln (Streu-, Absorptions- und Extinktionsquerschnitt mit Mie-Vergleich, Streudiagramm, Nahfeld).
* **Ergebnisse:** Spektren, Beugungsordnungen (R, T, A), Feldkarten (exakt auf den Elementen) für E, H und den Poynting-Vektor, Schnitte, Absorption je Material, Energie- und Flussbilanz, Zeit je Phase, Ableitungen, Resonanzen/Bänder mit Modenfeldern, hp-Adaptivität, CSV-Export.

Ausführliche Bedienung: [docs/ANLEITUNG.md](docs/ANLEITUNG.md). Aufbau des Codes: [docs/ARCHITEKTUR.md](docs/ARCHITEKTUR.md).

## Voraussetzungen

Die App startet die Rechnung (Worker) als eigenen Prozess in einem Python mit `hpfem`. Das kann dasselbe Python sein wie das der App (empfohlen) oder ein anderes:

| Teil | Python | Pakete |
|---|---|---|
| App (Oberfläche, Gmsh, Auswertung) | beliebiges Python ≥ 3.10 | `pip install -r requirements.txt` |
| Rechnung (Worker) | ein Python mit `hpfem` | [hp-FEM](https://github.com/PhotonPower/hp-FEM) **ab 0.4** empfohlen (`hpfem.grating`, Resonanzen, Prüfungen; Ableitungen ab dem Stand M16 auf `main`). Ältere Versionen mit `ConicalScattering` gehen mit den klassischen Lösern. |

## Installation und Start (Windows, eine Umgebung für App und Rechnung)

```bash
git clone https://github.com/PhotonPower/GUI-hpFEM.git
git clone https://github.com/PhotonPower/hp-FEM.git hp-FEM-lib     # daneben
cd GUI-hpFEM
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

hpfem in dieselbe venv: entweder das fertige Wheel des Releases (z. B. `hpfem-0.4.0-cp311-cp311-win_amd64.whl` von der Release-Seite von hp-FEM) mit `pip install <wheel>`, oder den neuesten Stand von `main` selbst bauen (MSYS2 UCRT64 mit `gcc`, `cmake` und `ninja` aus pip, wie der Wheels-Workflow von hp-FEM):

```bash
# in Git Bash, im Ordner hp-FEM-lib
export PATH="$PWD/../GUI-hpFEM/.venv/Scripts:/c/msys64/ucrt64/bin:$PATH" CC=gcc CXX=g++ CMAKE_GENERATOR=Ninja
python -m pip install ninja cmake
python -m pip wheel . -w dist --no-deps --config-settings=cmake.define.HPFEM_ENABLE_MUMPS=OFF --config-settings=cmake.define.HPFEM_STATIC_RUNTIME=ON
python -m pip install dist/hpfem-*.whl
```

Start: Doppelklick auf `start_windows.bat` (nimmt `.venv`, wenn vorhanden), unter Linux/macOS `./start.sh`, oder `streamlit run fem_gui/fem_app.py`.

In der Seitenleiste:

* **Python mit hpfem**: vorbelegt mit dem Python der App, wenn es hpfem hat; sonst z. B. `C:/msys64/ucrt64/bin/python3.exe`. Darunter meldet die App Version, Löser-Backends, Threads und (unter „Bibliothek: Details“) die vorhandenen Funktionen.
* **hp-FEM-Ordner** (optional): vorbelegt aus `HPFEM_REPO`, `../hp-FEM-lib` oder `../hp-FEM`. Er kommt nur dann in den PYTHONPATH des Workers, wenn dort ein gebautes Modul liegt (`python/hpfem/_hpfem*.pyd`); sonst würde er das installierte hpfem verdecken.
* Arbeitsordner und Threads.

**Hinweis zu pip:** Nennt eine globale `pip.ini` einen nicht erreichbaren zusätzlichen Index (z. B. `pypi.ngc.nvidia.com`), bricht pip mit Verbindungsfehlern ab. Abhilfe nur für die venv: `.venv\pip.ini` mit

```ini
[global]
index-url = https://pypi.org/simple
extra-index-url =
```

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
