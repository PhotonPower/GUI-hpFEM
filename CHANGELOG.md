# Änderungen

## Unveröffentlicht

* Neu: **Erweiterte Studien** (Modus „Studie“, Aufgabe Optimierung): Nebenbedingungen an Messgrößen (`optimize.constraints`, `hpfem.opt.OutcomeConstraint`,
  Verfahren bayesian) und an Parameter (`optimize.linear_constraints`, `hpfem.opt.LinearConstraint` auf dem Entwurfsraum, alle Verfahren), Mehrzieloptimierung
  (Verfahren `pareto`, `hpfem.opt.pareto_optimize` mit `objectives`, `maximize`, `ref_point`) und Mehrstufen-Optimierung (`multi_fidelity`, Stufen
  `fidelities`), beide mit BoTorch (`pip install "hpfem[opt-bo]"`; `FEATS["opt_bo"]`). Formular „Nebenbedingungen und Mehrziel“ in Reiter 1 (schreibt ins
  JSON, mit Prüfung), neue Vorlage „Optimierung mit Nebenbedingungen“. Neuer Studien-Worker `fem_study_worker.py`: der Job-Runner `hpfem.run` selbst (gleiche
  Ereignisse, `results.json` und Studienspeicher), nur mit ersetztem Optimierungsschritt (hpfem 0.5.x); nur Jobs mit den neuen Schlüsseln laufen darüber,
  alle anderen wie bisher über `python -m hpfem.run`. Verlauf und Pareto-Reiter markieren Auswertungen, die eine Nebenbedingung verletzen; die Front einer
  Mehrzieloptimierung steht mit Referenzpunkt als Tabelle, Diagramm und CSV im Ergebnis. Integer- und kategoriale Entwurfsvariablen gibt es nicht: der
  Evaluator `GratingEvaluator` kennt nur Geometrie- und Materialparameter.
* Neu: Modus **„Bandstruktur (photonischer Kristall)“** (Seitenleiste; hpfem 0.5, `BandStructure2D`, Floquet-Bloch): Bänder f = ω a / (2π c) eines 2D-Kristalls
  entlang Γ–X–M–Γ (quadratisch) bzw. Γ–M–K–Γ (hexagonal), Polarisation mit E in der Ebene (H_z; die Polarisation E_z deckt der Löser nicht ab). Zelle
  maßstabsfrei in Einheiten der Gitterkonstante, Formen (Scheibe, Rechteck, Ring, Polygon, Priorität nach Reihenfolge) werden im Gitter wiederholt (periodische
  Ränder im Gmsh-Netz), nur reelle konstante ε. Ergebnis: Bandstrukturdiagramm mit Lücken über den ganzen Pfad (Δf/Mitte in %), optional dem leeren Gitter
  |k + G| als Referenz, Tabellen und CSV, Modenfelder an den Eckpunkten. Der zweite Nullwert bei Γ (konstantes Feld) wird entfernt. Die Krylov-Dimension wird fest
  gesetzt (die Automatik der Bibliothek stockt bei Γ). Vorlagen: quadratisches Stabgitter (ε = 12, r = 0,2 a; keine Lücke für E in der Ebene), leere Gitter als
  Referenz, Dreiecksgitter aus Luftlöchern (ε = 13, r = 0,4 a). Neue Module `fem_bands.py`, `fem_bands_app.py`, `fem_bands_worker.py`.
* Neu: Aufgabe **„Hohlraum“** im Rotationskörper-Modus (hpfem 0.5, `AxisymmetricCavity`): Eigenmoden eines geschlossenen Zylinders mit PEC-Wand (0 ≤ r ≤ R,
  0 ≤ z ≤ h), gefüllt mit der „Umgebung“, Teile als Einsätze, verlustfrei (einachsige Kristalle mit Achse längs z, radial oder azimutal gehen). Je Azimutordnung
  m die niedrigsten Moden: k₀, λ = 2π/k₀, f = c/λ (Tabelle, CSV, Diagramm der Frequenzen über m), Modenfelder auf dem Meridianschnitt. Für den leeren Zylinder
  steht die analytische TM/TE-Lösung daneben. Eigenes Netz ohne PML (`fem_axi.build_cavity_mesh`), Vorlage „Zylinderhohlraum R = 1 µm, h = 1,5 µm“, Abschnitt
  „Hohlraum“ im Reiter 1. Gemessen mit dem groben Netz der Vorlage (3 Elemente je Wellenlänge, p = 3): größte Abweichung von der analytischen Wellenlänge 7·10⁻⁴
  über 12 Moden (m = 0, 1, 2), TM₀₁₀ auf 1·10⁻⁷.
* Neu: **Drude-Lorentz-Materialien** und **selbstkonsistente Resonanzen**. Neuer Materialtyp „Drude-Lorentz (Fit)“ (ε∞, ħωp, ħγ und Lorentz-Oszillatoren in eV,
  `hpfem.materials.DrudeLorentz`), in allen Modi. Im Materialeditor passt „An Messdaten anpassen“ ein Drude-Lorentz-Modell an Bibliotheks-, Tabellen- und
  Drude-Materialien über einen Wellenlängenbereich an (`hpfem.materials.fit_drude_lorentz`, Vergleich Re/Im ε, größter relativer Fehler; als neues Material
  oder als Ersatz). Aufgabe „Resonanzen“: Haken „Dispersion selbstkonsistent“ reicht die dispersiven Modelle an `hpfem.grating.resonances` und führt jede Mode
  mit `grating.refine_resonance` nach (ε bei der eigenen komplexen Frequenz); die Tabelle zeigt zusätzlich λ und Q der eingefrorenen Rechnung. Neues Modul
  `fem_disp.py` (läuft im hpfem-Python). Verkippte optische Achsen gab es im periodischen Modus schon (einachsig θ, φ; zweiachsig Euler-Winkel; voller Tensor).
* Neu: **Photovoltaik** im periodischen Modus (hpfem.pv; Reiter 3 Expander „Photovoltaik“, Reiter 4 Unterreiter „Photovoltaik“): je Streupunkt die Generationsrate
  im gewählten Absorber (absorbierte Leistung je Zelle / (ħω · Fläche), reine Nachbereitung ohne zusätzlichen Lösungsschritt) und ihr seitlich gemitteltes
  Tiefenprofil G(y) (`hpfem.pv.depth_profile`, ohne PML). Bei einem Wellenlängendurchlauf gewichtet die App die Punkte mit einem Spektrum (Schwarzkörper 5778 K als
  dokumentierte Näherung oder eigenes CSV, kein Normspektrum eingebaut) und zeigt J_sc (Sammelwirkungsgrad 1, obere Grenze), Absorption im Absorber, spektral
  gewichtetes G(y) und CSV. Neue Module `fem_pv.py` (rein), `fem_pv_app.py` (Oberfläche), `pv_profile` im Worker.
* Neu: Warnung in der Seitenleiste, wenn der Quell-Build im hp-FEM-Ordner älter ist als das im gewählten Python installierte hpfem (der Quell-Build
  kommt in den `PYTHONPATH` und verdeckt die neuere Version). `fem_run.older_source_build`, Eintrag `older_source` in der Bibliotheksprüfung.
* Neu: **Emitter in der Gitterzelle** (hp-FEM 0.5, M17; Reiter 1 „Emitter in der Zelle“, Aufgabe „Emitter“ im Reiter 3): ein Gauß-Dipol in der periodischen
  Zelle. Stufe B (einzelner Dipol per Array-Scan, `grating.dipole_emission`): Purcell-Faktor je Orientierung, Anteile nach oben, unten und nicht abgestrahlt
  (absorbiert oder geführt) und Leistung in Sammelaperturen (`grating.emission_cone`); Stufe A (Bloch-Array, `grating.emit`, Leistung je Ordnung); Stufe C
  (Abstrahlmuster dP/dΩ, `grating.emission_pattern`, Aperturen aus dem Raster). Vorab schätzt `grating.emission_cost` Zahl der Lösungen, Speicher und
  Rechenzeit (`COST` im Protokoll, „Nur die Kosten schätzen“); die Suche nach geführten Moden („Pole“) lässt sich abschalten. Das Netz wird um den Emitter
  verfeinert (σ-abhängig), die Position wird gegen Grenzflächen und Zellrand geprüft (6σ in einem verlustfreien Medium), Vorschau und Netz zeigen ihn.
  Neue Module `fem_emit.py` (Modell, Prüfung, Tabellen, Abbildungen), `fem_emit_app.py` (Oberfläche), `fem_emit_worker.py` (Rechnung im Worker).
* Neu: Modus **„Wellenleiter und Ports“** (Seitenleiste; ersetzt den Modus „Emitter über Schichtstapel“, dessen halbanalytische Rechnung die Bibliothek weiter
  bietet): Aufgabe **Ports** (`WaveguidePort`, `s_parameters`): Bauteil in der Chipebene (Effektivindex-Modell, H_z) zwischen modalen Ports bei x = 0 und x = L,
  Formen Streifen, Rechteck, Scheibe, Ring und Polygon mit Priorität, Gmsh-Netz mit Verfeinerung an Grenzflächen; Ergebnis: leistungsnormierte S-Matrix (|S|², Phase),
  Transmission, Reflexion und Verlust des Anregungskanals über der Wellenlänge, effektive Indizes der Portmoden, Feldkarte einer Anregung, CSV. Aufgabe **Moden**
  (`PropagatingMode`): effektive Indizes und Felder E_t, E_z der geführten Moden eines Querschnitts in einem PEC-Kasten. Vorlagen: Streifen, Stufe 450 → 900 nm,
  Querschnitt 450 × 220 nm. Module `fem_wg.py`, `fem_wg_app.py`, `fem_wg_worker.py`.
* Neu: **Emitter außerhalb der Achse** bei Rotationskörpern (Modus „Rotationskörper“, Emitter, „außerhalb der Achse“): Abstände ρ, alle Azimutordnungen,
  `dataports.axisymmetric_emitter_response(radii=…)`. Ergebnis: Purcell-Tensor (F_ρ, F_φ, F_z, A_xz), abgestrahlter Anteil, Detektoranteile (optional mit NA) über ρ,
  Tabelle und JSON-Umschlag. Das Netz bekommt einen Kasten (6σ) je Abstand; der Worker schiebt das Netz so, dass die Oberkante des Stapels bei z = 0 liegt.
* Neu: **Kristalle bei Rotationskörpern** (hp-FEM M19 S6): der Tensor wird in (r, φ, z) gelesen (x→r, y→φ). Erlaubt sind die optische Achse längs der Symmetrieachse, radial oder
  azimutal, zweiachsig ohne Drehung und diagonale Tensoren; eine verkippte Achse wird mit Hinweis abgelehnt. Resonanzen, Streuung und Emitter auf der Achse; Umgebung und Emitterwirt
  bleiben isotrop. Emitter außerhalb der Achse gehen in hp-FEM 0.5 nicht mit Kristallen (die Bibliothek bricht ab, die App meldet es vorab).
* Neu: **Kristall-Vorlagen** in der Materialauswahl aller Modi mit Kristallunterstützung (Saphir, Quarz, Kalkspat, LiNbO3, Rutil, MgF2; konstante Hauptindizes bei λ0).
* Neu: Materialien **Ti und Cr** (Johnson & Christy 1974, Haftschichten unter Gold) in der Bibliothek und in der Auswahl „aus der Bibliothek hinzufügen“.
* Geändert: Die Bibliotheksprüfung (`fem_run.probe_library`) meldet zusätzlich `emitter`, `dataports` und `offaxis`; die App bietet nur an, was vorhanden ist.

* Neu: **Studien** (hp-FEM M16 S9b, Seitenleiste „Art des Modells“ → „Studie“): Optimierung, Rekonstruktion von Parametern aus Messwerten und
  Unsicherheits-/Sensitivitätsanalyse mit dem Job-Runner der Bibliothek (`python -m hpfem.run`, Schema 2, Aufgaben `optimize`, `reconstruct`,
  `uq`). Job-Dokument aus Vorlagen (Lamellenzelle) oder Datei als JSON bearbeiten, im Hintergrund starten, abbrechen und fortsetzen (der
  Studienspeicher `*.study.jsonl` bleibt). Ansichten: Verlauf mit dem bisher besten Wert und den Gründen fehlgeschlagener Auswertungen, Parameter
  je Auswertung, abgetastete Punkte, Pareto-Front zweier Messgrößen, Optimum, Anpassung mit Laplace-Fehlern, Korrelationen und emcee-Posterior,
  Streuung der Messgrößen und Sobol'-Balken; Verlauf, Front, Ergebnisse und Speicher als Download.
* Neu: **Kristalle (anisotrope Materialien)** im Materialeditor (hp-FEM M19, ADR-0015): einachsig (n_o, n_e, Achse über θ gegen die Normale und φ
  gegen x), zweiachsig (drei Hauptindizes, Euler-Winkel z-x-z) oder voller symmetrischer Tensor, im Laborsystem (x quer zu den Linien, y längs, z die
  Normale); in Formen, Schichten und Substrat, das Einfallsmedium bleibt isotrop. Nur mit dem Löser `hpfem.grating` (Schichten oder Substrat aus
  Kristall: `AnisotropicStack`).
* Neu: **Jones- und Mueller-Matrizen** je ausbreitungsfähiger Ordnung (`hpfem.grating.jones`, Option „Jones- und Mueller-Matrizen berechnen“):
  Reiter „Jones / Mueller“ mit der Mueller-Matrix (Einzelpunkt als Karte, Durchlauf als 4×4 Kurven, normiert auf m_00) und der Jones-Tabelle
  (Betrag, Phase, Effizienz); bei einem Kristallsubstrat sind die Zeilen der transmittierten Matrix die beiden Moden. Gegen die analytische
  Jones-Matrix einer flachen Kristallplatte auf Kristallsubstrat: 3·10⁻¹⁶.

## 0.3.0 (2026-10-09)

* Neu: **Radial unendliche Schichten** bei Rotationskörpern (durch die PML bis zur Wand): planare Spiegel, Membranen, Schichtwellenleiter; DBR-Generator
  mit planarem unterem Spiegel; Vorlagen „GaAs-Membran mit Quantenpunkt“ und „Mikrosäule auf planarem unterem Spiegel“. Gegen die exakte
  Sommerfeld-Lösung: 1,4·10⁻³ (Dipol in der Membranebene) und 4·10⁻⁴ (axial) mit numerischer Normierung.
* Neu: Option „Purcell-Normierung numerisch auf demselben Netz“.

* Neu: **Isolierte Strukturen** im Modus „periodische Struktur“ (seitliche Ränder „isoliert“): PML links und rechts, Schichtstapel als Hintergrund,
  Streu-/Absorptions-/Extinktionsbreite (Mie-Vergleich für Zylinder auf 1e-4 bis 9e-4), Fernfeld, Detektoren für den Energiefluss, Messbox;
  Vorlagen „Isolierter Zylinder (Mie-Test)“ und „Slit-Groove-Benchmark“ (S/S₀ = 2,19923 bei p = 4 gegen 2,198826, +1,9e-4).
* Geändert: Kartenansichten in `fem_app` als Funktionen (`maps_view`, `cuts_view`); „Elemente pro Eindringtiefe“ ab 0,2.

* Neu: Modus **„Rotationskörper: Resonator, Emitter“** (Seitenleiste „Art des Modells“) auf dem zylindersymmetrischen Löser von hp-FEM
  (`AxisymmetricResonance`, `AxisymmetricScattering`): Teile im Querschnitt (Zylinder/Ring, Kegelstumpf, Kugel, Ellipsoid, Torus, Polygon), Substrat,
  Generator für Mikrosäulen mit Bragg-Spiegeln, Gmsh-Meridiannetz mit Achse, PML und Emitterkasten.
* Neu: **Resonanzen** der Ordnung m (λ_res, Q, Modenfelder E_r, E_φ, E_z) und **Emitter** auf der Achse (Purcell-Faktor, β nach oben, Anteil nach
  unten über der Wellenlänge, optional um die gefundene Resonanz zentriert, Feldkarte).
* Neu: **Streuung ebener Wellen** an Rotationskörpern (beliebiger Einfallswinkel, S/P): σ_sca, σ_abs, σ_ext über λ, Streudiagramm, Nahfeld in der
  Einfallsebene, Mie-Reihe als Referenz für Kugeln (gemessen: Au-Kugel und Si-Kugel unter 45° auf 10⁻⁴ bis 10⁻³).
* Neu: **Modenzerlegung** des Purcell-Spektrums (Riesz-Projektion): Anteil der Resonanz und Hintergrund.
* Neu: Vorlagen Mikrosäule GaAs/AlAs (wie `examples/micropillar_qd`), dielektrische Kugel, Gold-Nanokugel mit Emitter, Au-Kugel in Wasser
  (Streuung), Si-Nanokugel (Mie-Resonanzen, schräger Einfall), Mikroscheibe.
* Behoben: Splitterzellen zwischen DBR-Schichten (Rundung der Grenzflächen); Boolesche Toleranz in Gmsh.
* Geändert: Materialeditor und Hilfsfunktionen in `fem_ui.py` (von beiden Modi genutzt); `fem_run.start_worker(..., script=)`.

## 0.2.0 (2026-10-09)

Anpassung an hp-FEM 0.4 und den Stand M16 auf `main` (getestet mit einem Build von `main`, Commit 1c50a5b, Python 3.11, Windows).

* Neu: Löser **„hpfem.grating“** (Ein-Aufruf-Schnittstelle der Bibliothek) als Standard: Prüfungen der Bibliothek vor jedem Punkt, skalarer E_z-Pfad
  (TE bei φ = 0), exakte Absorption, Flussbilanz, Amplituden der Ordnungen, Rechenzeit je Phase.
* Neu: **hp-Adaptivität für TE, TM und konischen Einfall** („hpfem.grating, hp-adaptiv“) mit Residuen- oder zielorientiertem **DWR-Schätzer** für R₀;
  die periodischen Ränder dürfen mitverfeinert werden.
* Neu: **Ableitungen** (`hpfem.grating.jacobian`): dR/dε, dT/dε (Re, Im) je Material, dR/dλ, dR/dθ, dR/dφ; Reiter „Ableitungen“ mit Diagramm, Tabelle, CSV.
* Neu: Aufgabe **„Resonanzen“** (`hpfem.grating.resonances`): komplexe Eigenfrequenzen, λ_res, Q, Bandstruktur über θ/φ, Modenfelder.
* Neu: **„Mit hpfem prüfen“**: Diagnosen der Bibliothek und Speicherschätzung ohne Rechnung (`fem_worker.py --check`); streifende Ordnungen an allen
  Durchlaufpunkten.
* Neu: **Phasenanzeige** im Fortschritt (Assemblierung, Faktorisierung, Lösen …), Speicherschätzung und Meldungen während der Rechnung;
  **Abbruch zwischen den Phasen** über eine Abbruchdatei statt hartem Beenden („Sofort beenden“ bleibt).
* Neu: Feldkarten für **|H|, H-Komponenten und den Poynting-Vektor** (Raster und Elemente).
* Neu: Reiter „Prüfung, Bilanz, Zeit“ (Meldungen, R + T + A, Flussbilanz, Zeit je Phase).
* Neu: Seitenleiste fragt die Bibliothek ab (Version, Backends, Threads, Funktionen) und bietet nur an, was vorhanden ist.
* Geändert: Rand-Tags des Gmsh-Netzes 1 bis 4 (`hpfem.box_tag`) statt 101 bis 104; ältere Netze neu erzeugen.
* Geändert: PYTHONPATH des Workers nur noch für einen Quellordner mit gebautem Modul (ein Quellordner ohne Modul verdeckte ein installiertes Wheel);
  „Python mit hpfem“ ist mit dem Python der App vorbelegt, wenn es hpfem hat; Materialdaten auch aus dem installierten hpfem.
* Geändert: `start_windows.bat` nimmt `.venv`, wenn vorhanden. README: Einrichtung mit einer venv, Bau von hpfem mit MSYS2-GCC, Hinweis zu `pip.ini`.
* Tests für die neuen Teile (ohne hpfem).

## 0.1.0 (2026-10-07)

* Erste Aufnahme des Prototyps ins Repository (Streamlit-App, Geometrie/Gmsh, Worker, Auswertung, Anleitung).
* Neu: `fem_materials.py` (Materialbibliothek ohne hpfem-Import; Werte wie `hpfem.materials`).
* Neu: Vorbelegung des hp-FEM-Ordners (`HPFEM_REPO`, `../hp-FEM`), Tests, CI, Startskripte, Dokumentation.

* CI-Workflow (GitHub Actions) aktiviert.
