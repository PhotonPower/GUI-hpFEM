# FEM-Modellwerkstatt: Kurzanleitung

Periodische Struktur eingeben, mit Gmsh vernetzen, mit hp-FEM rechnen, Schaubilder und Karten ansehen. Keine BEM- oder RCWA-Daten nötig.

## Installation
Siehe README im Hauptordner. Die sechs Dateien (`fem_app.py`, `fem_geometry.py`, `fem_materials.py`, `fem_run.py`, `fem_post.py`, `fem_worker.py`) liegen zusammen im Ordner `fem_gui/`.

Für die App (Windows-Python oder MSYS2-Python):
```
pip install -r requirements.txt
```
(Im MSYS2-Python alternativ `pacman -S mingw-w64-ucrt-x86_64-gmsh`; prüfen mit `python -c "import gmsh"`.)

Die Rechnung selbst läuft in einem Python mit `hpfem`. Am einfachsten ist ein Python für beides: hpfem ab 0.4 gibt es als Windows-Wheel für
python.org-CPython 3.10 bis 3.13 (`pip install hpfem-0.4.0-cp311-cp311-win_amd64.whl` aus den Releases von hp-FEM, oder selbst gebaut, siehe README).
Die App startet den Worker als eigenen Prozess. Start:
```
streamlit run fem_gui/fem_app.py
```
Seitenleiste: **Python mit hpfem** (vorbelegt mit dem Python der App, wenn es hpfem hat; sonst z. B. `C:/msys64/ucrt64/bin/python3.exe`), **hp-FEM-Ordner**
(optional: nur nötig für einen Quell-Build, dessen `python/` dann in den PYTHONPATH kommt), Arbeitsordner, Threads. Die Seitenleiste meldet Version,
Löser-Backends und Threads der gefundenen Bibliothek; unter „Bibliothek: Details“ steht, welche Funktionen sie hat.
Empfohlen ist hp-FEM ab 0.4 (Ein-Aufruf-Schnittstelle `hpfem.grating`, Prüfungen, Resonanzen; Ableitungen ab dem Stand M16). Mit älteren Versionen bietet
die App nur den klassischen konischen Löser und den In-Ebenen-Löser an.

## Ablauf (fünf Reiter)
1. **Modell:** Materialien (Bibliothek mit Si, Ag, Au, Al, Ti, Cr, GaAs, MAPbI3, SiO2, TiO2, Wasser, Luft; n + ik; ε; Drude; eigene Tabelle), Schichtaufbau
   (Einfallsmedium, Schichten, Substrat), Formen (Rechteck, Trapez, Kreis, Ellipse, Polygon; spätere überdecken frühere; Formen über dem Zellrand setzen sich
   periodisch fort), Beleuchtung (TE/TM, θ, φ, λ, Durchlauf über λ, θ oder φ) und Rechengebiet. Rechts steht die Vorschau, darunter die Prüfmeldungen.
   Mit **Vorlage laden** (Seitenleiste) starten, **Vorschlag aus der Wellenlänge übernehmen** setzt Höhen, PML und Substrattiefe.
2. **Netz:** Gmsh-Vernetzung mit Elementgröße nach Wellenlänge und Eindringtiefe, Verfeinerung an Grenzflächen, periodischem Rand, gekrümmten Elementen für
   Kreise und Ellipsen. Zeigt Statistik, Winkelqualität, Freiheitsgrade (Schätzung) und das Netz.
3. **Rechnung:** Aufgabe (Streuung oder Resonanzen), Löser, Polynomordnung p, PML-Zielfehler (angepasst an den größten Beugungswinkel), Ordnungen,
   Ableitungen, Feldkarten (welche Punkte, Auflösung, H und Poynting-Vektor), optional Konvergenzstudie in p. **„Mit hpfem prüfen“** lässt die Bibliothek
   das Modell vorab prüfen und schätzt den Speicher. Die Rechnung läuft im Hintergrund, die Seite zeigt Fortschritt mit der aktuellen Phase des Lösers
   (Assemblierung, Faktorisierung, Lösen …), Speicherschätzung, Meldungen und Protokoll; „Abbrechen“ hält zwischen zwei Phasen an.
4. **Ergebnisse:** Spektrum und Beugungsordnungen (R, T, A, je Ordnung, ebener Stapel als Referenz), Feldkarten (|E|, |E|², Komponenten Re/Im/Betrag, absorbierte
   Leistungsdichte, |H|, Poynting-Vektor; 1 bis 3 Perioden; Geometrie und Netz einzeichenbar), Schnitte, Absorption je Material, Prüfung/Bilanz/Zeit,
   Ableitungen, Konvergenz in p, hp-Adaptivität, Tabelle und CSV-Export, Archiv. Bei Resonanzen: Moden (λ_res, Q), Bänder und Modenfelder.
5. **Info:** Koordinaten, Konventionen, Grenzen.

## Löser wählen und hp-Adaptivität testen (Reiter 3)
Fünf Möglichkeiten:
* **hpfem.grating** (empfohlen; TE, TM, Azimut): die Ein-Aufruf-Schnittstelle der Bibliothek, feste Ordnung p auf dem vorhandenen Netz. Dazu: Prüfungen der
  Bibliothek vor jedem Punkt (Fehler stoppen den Punkt, abschaltbar), skalarer E_z-Pfad bei TE und φ = 0 (gleiches Ergebnis mit etwa einem Drittel der
  Freiheitsgrade), exakte Absorption, Flussbilanz, Zeit je Phase und auf Wunsch die **Ableitungen** (siehe unten).
* **hpfem.grating, hp-adaptiv** (TE, TM, Azimut): hp-Schleife mit dem konischen Löser. Fehlerschätzer **Residuum** (das Feld überall) oder **DWR**
  (zielorientiert: nur was die spiegelnde Reflexion R₀ beeinflusst, mit Schätzung ihres Fehlers; die Schleife stoppt, wenn er unter die Toleranz fällt).
  DWR ist **experimentell**: hp-FEM hat ihn bisher nur am konstruierten Eckproblem verifiziert, nicht an Gittern, und am Ag-Gitter unten konvergierte er
  im Test nicht. Für verlässliche Ergebnisse das Residuum nehmen.
  Die periodischen Ränder dürfen mitverfeinert werden; „Verfeinerung an den periodischen Rändern spiegeln“ hält sie gleich.
* **Konischer Löser, klassisch** (TE, TM, Azimut): der bisherige eigene Aufbau der App, auch für hpfem vor 0.4.
* **In-Ebenen-Löser, gleichmäßig** (nur TM, φ = 0): derselbe Fall mit dem älteren Löser. Auf demselben Netz gerechnet vergleicht das beide Löser: R, T und die
  Ordnungen müssen übereinstimmen.
* **In-Ebenen-Löser, hp-adaptiv** (nur TM, φ = 0): Schleife aus Rechnen, Fehlerschätzer je Dreieck, Dörfler-Markierung und h- oder p-Verfeinerung nach der Glattheit.
  Einstellbar: Startordnung p₀, höchste Schrittzahl, höchste Freiheitsgrade, Dörfler-Anteil, Abbruch bei Änderung von R und T unter einer Schwelle (in zwei aufeinanderfolgenden
  Schritten), und ob bei einem Durchlauf nur am ersten Punkt adaptiert und das Endnetz für alle weiteren Punkte benutzt wird. Dreiecke am periodischen Rand und in der PML werden nie
  markiert. Ergebnisse im Reiter 4 „hp-Adaptivität“: Tabelle der Schritte, Konvergenzdiagramme (R, T und η über den Freiheitsgraden) und das Endnetz, nach der Polynomordnung gefärbt.

**Vorschlag für den Test:** Vorlage „Ag-Lamellengitter (TM, 50°: Test der hp-Adaptivität)“. Reiter 2: 2 Elemente pro Wellenlänge, 1 Element pro Eindringtiefe (etwa 450 bis 900
Dreiecke), gerade Elemente. Dann dreimal rechnen und im Reiter 4 vergleichen: (1) In-Ebenen-Löser hp-adaptiv mit p₀ = 3, 12 bis 20 Schritten, höchstens 150 000 Freiheitsgraden;
(2) In-Ebenen-Löser gleichmäßig mit p = 4 und 5 auf einem feineren Netz; (3) konischer Löser. RCWA-Referenz für die Silberdaten dieser Bibliothek (ε = −4,698 + 0,217 i bei 405 nm,
1/N-extrapoliert, Unsicherheit etwa 1e-4): **R₀ = 0,7800, R₋₁ = 0,0801, absorbiert 0,140.** Gleichmäßige Netze blieben in den früheren Tests bei Abweichungen von etwa 4e-3 bis 9e-3 stehen,
die hp-adaptive Rechnung erreichte etwa 2e-5 (R₋₁) bei etwa 66 000 Freiheitsgraden.
Gekrümmte Netze (Kreise, Ellipsen) mit hp-Verfeinerung sind ungetestet; die App warnt.

**Gemessen mit hp-FEM `main` (1c50a5b, 9.10.2026)** auf dem Startnetz mit 1154 Dreiecken (2 Elemente pro Wellenlänge, gerade Elemente), PML-Zielfehler 1e-3:

| Löser | Freiheitsgrade | ΔR₀ | ΔR₋₁ | Zeit |
|---|---|---|---|---|
| hpfem.grating, gleichmäßig, p = 5 | 46 k | −2,8·10⁻³ | +4,0·10⁻³ | 4 s |
| hpfem.grating, hp-adaptiv, Residuum, p₀ = 3, 16 Schritte | 81 k | +3·10⁻⁷ | −3,4·10⁻⁴ | 70 s |
| hpfem.grating, hp-adaptiv, DWR (experimentell), 16 Schritte | 21 k | −1,5·10⁻³ | +5,0·10⁻³ | 67 s |

Die Residuen-Adaptivität des konischen Lösers erreicht also das Niveau der Referenz; mit mehr Schritten bzw. Startordnung 4 wird R₋₁ noch genauer
(hp-FEM, validation.md E: ΔR₋₁ = −5·10⁻⁶ bei 88 k Freiheitsgraden).


## Isolierte Strukturen (Reiter 1: „Seitliche Ränder: isoliert“)
Statt einer periodischen Zelle eine **einzelne** Struktur (Draht, Graben, Schlitz, Stufe, Partikelquerschnitt) im Schichtstapel: links und rechts
absorbieren PML-Schichten, die Schichten laufen hindurch, dahinter liegt eine Metallwand. Die „Periode“ heißt dann **Breite des Innengebiets**
(Struktur plus Abstand); die Formen müssen darin liegen. Gerechnet wird mit dem konischen Löser (TE, TM, konischer Einfall) und dem Schichtstapel als
analytischem Hintergrund (Streufeld-Formulierung, wie im Slit-Groove-Benchmark der Bibliothek).

Ergebnisse (Reiter 4):
* **Querschnitte je Länge** („Breiten“, nm): σ_sca aus dem Fluss des Streufelds durch die geschlossene Messbox (grün in der Vorschau) und der Anteil
  nach oben; in **homogener Umgebung** (gleiches Material oben und unten, keine Schichten, PML unten) zusätzlich σ_abs und σ_ext (hpfem
  `conical_cross_sections`) und das Fernfeld dσ/dφ. Für einen einzelnen Kreiszylinder bei φ = 0 die Mie-Reihe als Referenz.
* **Detektoren** (Reiter 1, Rechengebiet: eine Zeile je Detektor „Name; y; x von; x bis“): Energiefluss des Gesamtfelds nach unten durch die
  Strecke, absolut (W/m für |E₀| = 1 V/m) und normiert auf den einfallenden Fluss durch dieselbe Breite.
* Feldkarten, Schnitte, Konvergenz in p wie im periodischen Modus.

Mit Schichtstapel enthält die volumetrische Absorption den ebenen Stapel (über die ganze Breite) und ist kein Querschnitt; die App zeigt dann nur
Streubreite, Anteil nach oben und die Detektoren.

**Gemessen (9.10.2026):**

| Vorlage | Ergebnis |
|---|---|
| Isolierter Zylinder n = 1,5, r = 200 nm, 500–1000 nm, TE und TM (p = 4, 3 Elemente/λ) | σ_sca und σ_ext gegen Mie auf 1·10⁻⁴ bis 9·10⁻⁴; eigene Streubreite gegen die der Bibliothek auf 1,4·10⁻⁴ |
| Slit-Groove-Benchmark (Ag-Film mit Schlitz und Rille, TM, 852 nm, Substrat ε = 2,25; 3 Elemente/λ, 0,25 Elemente je Eindringtiefe) | S/S₀ = 2,20081 (p = 3, 250 k Freiheitsgrade), 2,19923 (p = 4, 433 k); Referenz 2,198826 (Burger et al. 2013): +9·10⁻⁴ bzw. +1,9·10⁻⁴ |

Für den Slit-Groove-Benchmark beide Vorlagen rechnen („… S“ mit Rille und „… ohne Rille (Referenz S₀)“) und die Detektorwerte „P nach unten“
teilen. In Reiter 2 „Elemente pro Eindringtiefe“ auf etwa 0,25 stellen (die hohe Ordnung p löst den Skin-Effekt auf; 1 Element je Eindringtiefe
ergäbe über die 10 µm Filmbreite unnötig viele Dreiecke).

## Rotationskörper: Hohlraum (Reiter 3, Aufgabe „Hohlraum“)
Die Aufgabe „Hohlraum“ rechnet die Eigenmoden eines **geschlossenen** Zylinders mit idealer Leiterwand (PEC) ohne PML (hpfem: `AxisymmetricCavity`). Reiter 1,
Expander „Hohlraum“: Radius R, Höhe h, Bezugswellenlänge (dort werden die Permittivitäten ausgewertet und die Elementgröße bestimmt: die kleinste gesuchte
Wellenlänge wählen), Azimutordnungen m (z. B. 0, 1, 2) und die Zahl der niedrigsten Moden je Ordnung. Gefüllt wird der Hohlraum mit dem Material „Umgebung“, die
Teile des Modells liegen als Einsätze darin (z. B. eine dielektrische Scheibe; sie müssen in 0 ≤ r ≤ R, 0 ≤ z ≤ h liegen). Substrat, Schichten, PML und Emitter gelten
nicht. Alle Materialien müssen verlustfrei sein; einachsige Kristalle gehen, wenn die optische Achse längs z, radial oder azimutal liegt. Reiter 2 erzeugt dafür ein eigenes
Netz (geschlossenes Rechteck der Meridianebene), Reiter 3 rechnet je Ordnung ein reelles Eigenwertproblem (Sekundenbruchteile bis Sekunden), Reiter 4 zeigt Tabelle (m,
Mode, λ, f = c/λ, k₀; CSV), das Diagramm der Frequenzen über m und die Modenfelder (|E|, E_r, E_φ, E_z).
**Test mit der Vorlage „Zylinderhohlraum R = 1 µm, h = 1,5 µm“** (leer, Luft): die Ergebnistabelle ordnet jeder Mode die analytische zu (TM_mnp: k = √((j_mn/R)² + (pπ/h)²),
TE_mnp: k = √((j′_mn/R)² + (pπ/h)²), p ≥ 1). **Gemessen** (3 Elemente je Wellenlänge, p = 3, hpfem 0.5): TM₀₁₀ bei 2613,5 nm (analytisch 2613,5 nm, Abweichung 1·10⁻⁷),
TM₀₁₁ 1970,8 nm, TE₁₁₁ 2253,8 nm; über die 12 niedrigsten Moden (m = 0, 1, 2) ist die größte Abweichung 7·10⁻⁴ (die Paare TE₀ₙₚ/TM₁ₙₚ sind entartet, dort kann der Name
der Partnermode erscheinen). Ein einachsiger Kristall mit der Achse längs z skaliert TM₀₁₀ mit n_e, ein isotropes Dielektrikum alle Wellenlängen mit n.

## Rotationskörper: Resonatoren und Emitter (Seitenleiste „Art des Modells“)
Für Strukturen mit Rotationssymmetrie um die z-Achse rechnet die App jede Azimutordnung m als 2D-Problem in der Meridianebene (r ≥ 0, z). Ablauf wie
im periodischen Modus in fünf Reitern:
1. **Modell:** Materialien, Umgebung (verlustfrei) und optional ein Substrat (Halbraum z < 0); Teile im Querschnitt: Zylinder/Scheibe/Ring, Kegelstumpf,
   Kugel, Rotationsellipsoid, Torus, Polygon (später überdeckt früher). **Generator „Mikrosäule mit Bragg-Spiegeln“**: Entwurfswellenlänge, Radius, Paare
   oben/unten, Materialien, Kavitätslänge; setzt den Emitter in die Kavitätsmitte. **Resonanzsuche:** Zielwellenlänge, Azimutordnung m, Anzahl Moden.
   **Emitter:** Position z auf der Achse, Richtung (senkrecht zur Achse: m = ±1; entlang: m = 0), Ausdehnung σ, Spektrum fest oder „um die Resonanz“
   (± Linienbreiten λ/Q). **Rechengebiet:** Abstand zur PML und PML-Dicke („Vorschlag“: 0,75 λ und 1,5 λ). Rechts: Querschnitt (gespiegelt), PML,
   Messebenen, Emitter.
2. **Netz:** Gmsh in der Meridianebene, Verfeinerung an Grenzflächen und um den Emitter (Kasten 4σ), gekrümmte Elemente für runde Teile.
3. **Rechnung:** Aufgabe „Resonanzen“ oder „Emitter“, Polynomordnung p (2 für Übersichten, 3 für genaue Werte), Felder speichern.
4. **Ergebnisse:** Resonanzen: λ_res und Q (Diagramm, Tabelle, CSV), Modenfelder (|E|, Komponenten E_r, E_φ, E_z; als Schnitt durch die Achse
   gespiegelt). Emitter: Purcell-Faktor und Anteile nach oben (β) und unten über der Wellenlänge, die gefundene Resonanz, das Feld in der Mitte des
   Spektrums.

**Test mit der Vorlage „Mikrosäule … (schnell)“** (wie das Beispiel `examples/micropillar_qd` von hp-FEM, GaAs/AlAs mit n = 3,53 / 2,95, r = 0,75 µm,
6/10 Paare): Reiter 2 mit 4 Elementen pro Wellenlänge, Reiter 3 „Resonanzen“ mit p = 2: Grundmode (m = 1) bei etwa **930,5 nm, Q ≈ 170** (p = 3: 930,6 nm,
Q ≈ 172). Dann „Emitter“ mit Spektrum „um die Resonanz“: das Maximum des Purcell-Faktors liegt auf der Resonanz, die Breite ist etwa λ/Q.
**Gemessen (hpfem `main` 1c50a5b, 9.10.2026)**, Purcell-Faktor und β jeweils genau an der eigenen Resonanz:

| Rechnung | λ_res | Q | F_P | β oben |
|---|---|---|---|---|
| GUI, 4 Elemente/λ, p = 2 | 930,44 nm | 172,1 | 2,05 | 0,291 |
| GUI, 4 Elemente/λ, p = 3 | 930,65 nm | 172,1 | 1,97 | 0,295 |
| Beispiel von hp-FEM, strukturiertes Netz 0,05 λ, p = 3 | 930,65 nm | 172,1 | 1,99 | 0,291 |
| Beispiel von hp-FEM, schnelle Einstellung (0,1 λ, p = 2) | 930,15 nm | 171 | 2,33 | 0,24 |

Resonanz und Q konvergieren schnell, der Purcell-Faktor langsamer: **für genaue F_P und β mit p = 3 rechnen** (p = 2 liegt etwa 4 % darüber,
p = 3 auf 1 % am feinsten Vergleichswert). Der Wert 2,33 der README von hp-FEM gehört zur groben Schnelleinstellung und ist nicht konvergiert.
Mit Modenzerlegung (p = 2, 7 Wellenlängen): Summe der Moden 1,98 gegen 2,05 direkt an der Resonanz, Anteil der Grundmode 1,06, Hintergrund 0,92.

**Schichten (radial unendlich).** Planare Schichten (Material, Unterkante z, Dicke) laufen wie das Substrat über den ganzen Radius durch die PML
bis zur Wand: planare Bragg-Spiegel unter einer Säule, Membranen, Schichtwellenleiter. In ihnen geführte Leistung wird in der PML absorbiert und
zählt beim Emitter zum seitlichen Anteil. Der DBR-Generator kann den unteren Spiegel als solche Schichten anlegen („nur die Kavität und der obere
Spiegel sind geätzt“). Nicht für die Streuung ebener Wellen (geschichteter Hintergrund fehlt im zylindersymmetrischen Löser).

**Gemessen (10.10.2026), GaAs-Membran 200 nm in Luft, Emitter in der Mitte, 950 nm, p = 3,** gegen die exakte Sommerfeld-Lösung des planaren
Stapels (eigene Referenz, selbst geprüft am Spiegeldipol und an zwei perfekten Platten):

| Dipol | Sommerfeld | FEM, Normierung analytisch | FEM, Normierung numerisch |
|---|---|---|---|
| in der Membranebene (m = ±1), σ = 10 nm | 0,71513 | 0,71108 (−5,7·10⁻³) | 0,71412 (−1,4·10⁻³) |
| entlang der Achse (m = 0), σ = 10 nm | 1,09394 | 1,09257 (−1,3·10⁻³) | 1,09350 (−4,0·10⁻⁴) |

Unabhängig von PML-Dicke (1,5 λ / 3 λ) und Abstand (0,75 λ / 1,5 λ) bis auf 5 Stellen: die PML absorbiert die geführten Moden der Membran sauber.
Die verbleibende Abweichung kommt von der schmalen Gauß-Quelle; die Option **„Purcell-Normierung numerisch auf demselben Netz“** (Reiter 3) rechnet
P_bulk mit derselben Quelle im homogenen Emittermaterial und kürzt diesen Fehler heraus (doppelte Rechenzeit).

**Streuung an Partikeln** (Aufgabe „Streuung“): ebene Welle aus der Umgebung unter dem Winkel θ gegen +z (Richtung (sin θ, 0, cos θ)), S
(E entlang y) oder P (E in der Einfallsebene), Wellenlängenbereich und höchste Azimutordnung |m| (Faustregel k·R + 4; die Summe stoppt früher,
wenn ±m weniger als 10⁻⁵ der Streuleistung trägt). Ergebnisse: σ_sca, σ_abs, σ_ext über λ (rechts als Effizienz σ/πR²), für eine einzelne Kugel
mit der Mie-Reihe als gestrichelter Referenz; Streudiagramm dσ/dΩ in der Einfallsebene und senkrecht dazu; Nahfeld (Gesamt- oder Streufeld,
kartesische Komponenten) in der Einfallsebene. Nur ohne Substrat.

**Gemessen (9.10.2026, p = 3, 6 Elemente/λ):**

| Vorlage | Abweichung zur Mie-Reihe |
|---|---|
| Au-Kugel r = 40 nm in Wasser, 450–700 nm, axial | σ_sca ≤ 3·10⁻⁴, σ_ext ≤ 1,4·10⁻³, σ_abs ≤ 6·10⁻³ (relativ) |
| Si-Kugel r = 75 nm in Luft, θ = 45°, P, 450–800 nm (Ordnungen bis ±3) | σ_sca ≤ 1·10⁻⁴, σ_ext ≤ 1,3·10⁻³; σ_abs als kleine Differenz absolut ≤ 7·10⁻⁶ µm² |

**Modenzerlegung** (Emitter, Spektrum „um die Resonanz“, Haken „Modenzerlegung“): das Purcell-Spektrum als Summe über die Quasi-Normalmoden der
Resonanzsuche (Riesz-Projektion, PML bei der Zielwellenlänge eingefroren), mit dem Anteil der Resonanz (Lorentz-Kurve) und dem Hintergrund. Die
Summe weicht von der direkten Rechnung um wenige Prozent ab (eingefrorene PML). Kostet etwa 16 Lösungen je Pol und 48 für den Hintergrund.

**Emitter auf der Achse** koppeln nur an m = 0 (axial) und m = ±1 (senkrecht). Flüstergalerie-Moden (große m) einer Scheibe sind deshalb nur als
Resonanzen zugänglich (Vorlage „Mikroscheibe“, m = 12 bei etwa 975 nm).

**Kristalle (anisotrope Teile).** Rotationskörper dürfen anisotrope Teile enthalten (Materialeditor, Typ „Kristall“, auch als Vorlage: Saphir, Quarz, Kalkspat, LiNbO3, Rutil, MgF2 mit konstanten Indizes bei λ0). Die Symmetrieachse ist z, im Tensor gilt x → r, y → φ. Erlaubt sind die optische Achse längs z, radial oder azimutal und diagonale Tensoren; verkippte Achsen lehnt die App mit Hinweis ab. Umgebung und Emitterwirt müssen isotrop sein, Substrat und Schichten nur bei Streuung und Emitter; Resonanzen erlauben Kristall in Schichten (nicht nachgerechnet). Emitter außerhalb der Achse gehen in hp-FEM 0.5 nicht mit Kristallen. Gemessen (Kugel r = 300 nm, p = 3): Tensor ∝ I gibt dieselben Moden und σ_sca wie isotrop (bitgleich); n_o = 2,4, n_e = 2,7 verschiebt die Mode m = 1 von 696,6 nm auf 674,1 nm.

## Emitter (hp-FEM 0.5)

Zwei Wege, einen Dipol zu rechnen. Alle Faktoren sind auf den freien Dipol im selben Medium bezogen (frei = 1).

**1. In der Gitterzelle (Modus „Periodische Struktur“).** Reiter 1, Abschnitt „Emitter in der Zelle“: Haken setzen, Position x, y (y = 0 an der Oberkante des
Stapels) und Breite σ der Gauß-Quelle. Die App prüft, dass im Umkreis von 6σ nur ein verlustfreies Medium liegt und die Quelle 6σ vom Zellrand entfernt ist; die Vorschau
zeigt Stern und 6σ-Kreis, das Netz wird dort auf etwa 2σ verfeinert (Reiter 2 neu erzeugen). Reiter 3, Aufgabe **Emitter** (nur mit Emitter im Modell):

* **Stufe B** (einzelner Dipol, Array-Scan `grating.dipole_emission`): Purcell-Faktor je Orientierung (x, y, z, isotrop), Anteile nach oben, unten und „nicht abgestrahlt“
  (absorbiert, wenn die Struktur Verlust hat, sonst in geführten Moden) und die Leistung in Sammelaperturen (NA, Kegel um die Normale, `grating.emission_cone`).
* **Stufe A** (Bloch-Array bei festem kx, β, `grating.emit`): Leistung je Beugungsordnung nach oben und unten.
* **Stufe C** (Abstrahlmuster, `grating.emission_pattern`): dP/dΩ auf einem Raster aus θ, φ; Aperturen daraus.

Der Array-Scan löst viele Zellenprobleme (komplexe Bloch-Phasen). **Die Kosten stehen vorher im Protokoll** (`COST`: Zahl der Lösungen, Freiheitsgrade und Speicher je Faktorisierung,
gemessene Zeit je Lösung und Gesamtzeit) und unter Ergebnisse; „Nur die Kosten schätzen“ rechnet nichts anderes. Mit Stützstellen 6/4, Winkelnetz 16×32 und je einer
Faktorisierung von etwa 10 000 Freiheitsgraden sind das einige Hundert bis Tausende Lösungen je Wellenlänge. Die Suche nach geführten Moden und Oberflächenplasmonen („Pole“,
Einstellung „automatisch“) vervielfacht die Zahl, sobald ein Material einen größeren Index als die Halbräume hat oder ein Metall vorkommt: an der TiO₂-Metaoberfläche der
Vorlage (Stützstellen 4/3, Winkelnetz 8×16) 2996 statt 693 Zellenprobleme je Wellenlänge und eine geschätzte Zeit von 8,7 h für zwei Wellenlängen. Nur wenn es sicher keine geführten Moden gibt, darf „aus“
gewählt werden; sonst ist das Ergebnis falsch. Die Quelle sitzt in einem verlustfreien Medium (auch im Inneren einer Form möglich, solange 6σ in einem Material liegen);
Kristalle und isolierte Strukturen gibt es hier nicht.

**2. Rotationskörper, auch außerhalb der Achse (Modus „Rotationskörper“, Emitter).** Mit dem Haken „Emitter außerhalb der Achse“ und einer Liste von Abständen ρ rechnet die
Bibliothek (`dataports.axisymmetric_emitter_response(radii=…)`) die Gauß-Quelle mit allen Azimutordnungen. Ergebnis je Wellenlänge und Abstand: Purcell-Tensor
(F_ρ radial, F_φ azimutal, F_z axial, dazu der Kreuzterm A_xz), abgestrahlter Anteil, Detektoranteile (optional ein Bereich mit NA oben), Tabelle und JSON-Umschlag. Alle drei
Dipolrichtungen kommen zugleich; die Richtung „axial/senkrecht“ der Achsen-Rechnung gilt hier nicht. Das Netz bekommt einen Kasten von ±6σ um jede Position, die Kästen müssen
in einem Material liegen. Nur „fester Bereich“ als Spektrum (keine Resonanzzentrierung). Schlägt die Prüfung der Energiebilanz der Bibliothek an (A_rad > A_tot um mehr als 1 %),
die Polynomordnung erhöhen oder das Netz verfeinern. Die Position wird in Schritten von 0,1 nm verschoben, wenn der Punktsucher der Bibliothek sie nicht findet (steht im Protokoll).

**Gemessen (hpfem 0.5.0, 20.11.2026, p = 3):**

* *Planare Bibliotheksrechnung gegen Gitterzelle* (`dataports.planar_emitter_response`, Sommerfeld/Fresnel; früher ein eigener Modus der App). Flache Zelle (Luft/Glas, Periode 400 nm, Dipol 400 nm über dem Glas, σ = 10 nm, isotrop, λ = 600 nm, p = 3, 358 Dreiecke). Planar (Referenz): F = 0,9855, nach oben 0,5522,
  nach unten 0,4478, NA 0,5: 0,0700 der Gesamtleistung. **Stufe C** der Zelle (θ = 0° bis 30° in 5°-Schritten, φ = 0°): das Muster dP/dΩ stimmt mit dem planaren Fernfeld auf vier Stellen
  überein (Gewinn 1,2217 bei 0°, 0,8684 bei 30°), die Leistung in der Apertur NA 0,5 ist 0,0687 von P_bulk gegen 0,0690 (planar, auf den freien Dipol bezogen): 0,4 %. Das Raster der Stufe C
  muss über den Rand der Apertur hinausreichen (der Rand von NA 0,5 in Luft liegt bei 29,9993°); der Rand wird interpoliert. Dauer: 148 s für 14 ebene Wellen.
  **Stufe B** braucht auch bei dieser kleinen Zelle mit Stützstellen 4/3 und Winkelnetz 8×16 576 Lösungen (Schätzung des Programms: 336 Zellenprobleme und 400 ebene Wellen); mit sehr groben
  Einstellungen (Stützstellen 2/2, Winkelnetz 4×4, p = 2) schließt die Bilanz an der TiO₂-Metaoberfläche nur auf etwa 9 % (Anteile oben + unten = 1,09): für Zahlen erst die
  Stützstellen erhöhen und die Kanäle gegen die Summe prüfen.
* *Rotationskörper.* Glaskugel (r = 300 nm) in Luft, axialer Dipol 150 nm über dem Pol, σ = 8 nm: der Weg über die Achse (Larmor-Normierung) gibt F_P = 1,067 (700 nm) und 1,119 (800 nm),
  der Weg der Bibliothek bei ρ = 0 F_z = 1,0687 und 1,1195: Abweichung 0,2 % und 0,04 %. Auf der Achse gilt F_ρ = F_φ und A_xz = 0; bei ρ = 200 nm liegt F_z (700 nm) bei 1,007.

## 3D-Strukturen (Seitenleiste „Art des Modells“ → „3D-Strukturen“)
Der Modus rechnet mit den dreidimensionalen Lösern der Bibliothek ein Objekt (Kugel oder Würfel, im Ursprung) in Luft. Reiter 1: Vorlage, Aufgabe, Form, Radius (bei
dem Würfel die halbe Kante), Material des Objekts (alle Materialtypen der App ohne Kristalle). Reiter 2: Wellenlängen (Streuung), Zielwellenlänge und Modenzahl (Resonanzen) bzw.
Pfad und Bänder (Bandstruktur), Netz (Zellen je Radius n, Innenraum und Außenrand in Radien, dazwischen das PML; n mal Innenraum und n mal Außenrand müssen ganzzahlig sein),
Ordnung p und Speichergrenze. **„Netz prüfen und Speicher schätzen“** vernetzt das Modell und zeigt Zellen, Unbekannte, den Speicher der Bibliothek (`hpfem.estimate_memory`) und einen
Zeitrichtwert, dazu einen Schnitt durch das Netz; erst danach lohnt „Rechnen“. Reiter 3: Streuung: Q_sca, Q_abs, Q_ext (Kugel mit der Mie-Reihe und der Abweichung in %), |E|² auf der
x–z- und x–y-Ebene; Resonanzen: Tabelle λ, Q, Residuum und Feldschnitt je Mode; Bandstruktur: Diagramm f = ω a / (2π c) entlang Γ–X–M–R–Γ mit Lücken und dem leeren Gitter.

Grenzen: Der Wheel hat nur den direkten Löser SparseLU (ein Thread); 18 000 Unbekannte brauchen etwa 70 s je Faktorisierung, 50 000 über zwei Minuten und mehrere GB. Das Viertelmodell
(Haken „Viertelmodell“, nur für die x-polarisierte Welle längs z) spart einen Faktor 4 an Speicher. Mit n = 2, p = 2 liegt Q_sca der Kugel etwa 3 % neben der Mie-Reihe; die Warnung
des Lösers, das PML sei unteraufgelöst, erscheint bei diesen groben Netzen im Protokoll (so auch in der Validierung der Bibliothek). Resonanzen frieren das Material bei der Zielwellenlänge
ein und sind, wie die Bandstruktur, nicht gegen analytische Lösungen validiert; ein k-Punkt der Bandstruktur dauert schon bei sehr kleinen Netzen Sekunden bis eine Minute.

### 3D-Strukturen: Schichtstapel, Gmsh-Netze, hp-Adaptivität, Wellenleiter

**Netz** (Reiter 1, Streuung und Resonanzen): *Bibliotheksnetz* (wie oben), *Gmsh-Geometrie* (Kugel, Zylinder mit Achse z oder Quader in einer Box mit PML-Rand) oder *Eigene Gmsh-Datei*.
Bei der Gmsh-Geometrie stehen in Reiter 2 Innenraum (halbe Breite), PML-Dicke, Abstand nach oben, Zelle am Objekt, Zelle fern, Übergangslänge und Geometrieordnung; „Maschenweiten vorschlagen“
setzt Richtwerte aus der kürzesten Wellenlänge. **Das PML braucht mindestens zwei bis drei ferne Zellen Dicke** (gemessen: Kugel n = 2, ka = 1, p = 2: PML = 2 Zellen: Q_sca 0,3 % neben der Mie-Reihe;
eine Zelle: 24 % daneben); die Anzeige „PML = x ferne Zellen“ und die Warnung aus dem Protokoll helfen beim Prüfen. Voraussetzung: das Python-Modul `gmsh` im hpfem-Python.

**Schichtstapel** (Streuung mit Gmsh-Geometrie): Haken „Objekt über einem Schichtstapel“, Einfallsmedium (verlustfrei), Schichten (Material, Dicke; Hinzufügen und Entfernen), Substrat, Abstand
des Objekts zur Oberkante (0 = aufliegend; Kugeln brauchen Abstand), Tiefe des Substrats bis zum PML, Einfallswinkel, Polarisation s oder p, Azimut. Die Hintergrundlösung ist der analytische Stapel (`LayerStack3D`),
das Objekt streut nur die Abweichung. Die Tabelle zeigt R, T, A des Stapels ohne Objekt; σ sind auf die Intensität der einfallenden Welle bezogen, Q auf die Projektion des Objekts auf die x-y-Ebene.
Ein Stapel aus Luft liefert dieselbe Kugel wie ohne Stapel (Q_sca +1,7 % gegen Mie bei 21 000 Unbekannten); R einer Luft/Glas-Grenzfläche bei 30° (p) stimmt mit der Fresnel-Formel überein.
Das PML ist für den Index 1 ausgelegt: bei Substrat und Schichten mit höherem Index sind die Querschnitte Richtwerte, Substrat und Schichten sollten bis in den PML-Bereich reichen (Tiefe einstellen).

**Eigene Datei:** Datei hochladen (Gmsh 4.1 ASCII), Einheit der Koordinaten wählen, „Datei lesen“ (zeigt Zellmarken mit Zellzahl, Flächenmarken, Ausdehnung), jeder Zellmarke ein Material zuweisen, die
Zellmarken des Streukörpers ankreuzen, PML-Dicke einstellen (das PML liegt in der äußersten Schicht der Box). Randflächen ohne Marke werden zu den Seiten der Box (PEC). Streuung, Resonanzen und die
hp-Verfeinerung laufen wie bei den eingebauten Netzen; Q bezieht sich auf die Projektion des Objekts (aus der Oberfläche mit geraden Facetten, bei gekrümmten Körpern ungenau, σ ist davon unabhängig).

**hp-Adaptivität** (Reiter 2, Streuung): Haken „hp-adaptive Verfeinerung“, Schritte, Dörfler-Anteil θ, Obergrenze Unbekannte, optional η/η₀. Je Schritt wird gelöst, der residuale Fehlerschätzer
ausgewertet, markiert und h- oder p-verfeinert; die Startordnung p ist die kleinste. Reiter 3 „hp-Verfeinerung“: Diagramm η und Q_sca gegen Unbekannte, Tabelle je Schritt; Reiter „Netz“: Schnitt, auf Wunsch
nach Ordnung gefärbt. η fällt, Q_sca konvergiert bei grobem PML nicht monoton (Kugel n = 2, ka = 2: η 0,133 → 0,042, Q_sca 4,51 → 4,44, Mie 4,77, in 16 s).

**Wellenleiter und Ports** (Seitenleiste „3D-Wellenleiter und Ports“ oder Modus „3D-Strukturen“ → Aufgabe „Wellenleiter und Ports (S-Parameter)“)**:** Bauteil Streifenwellenleiter, Richtkoppler oder eigene Gmsh-Datei. Kern (Breite, Höhe), Länge, Rand, Materialien Kern und Mantel
(dispersiv auswertbar), beim Koppler Spalt und Portart (vier Halbflächen-Ports oder ein Port mit zwei Moden je Stirnfläche); Reiter 2: Wellenlängen, Zelle (und Zelle längs z, gröber spart viel), Ordnung,
Anregung (Port und Mode), Feld der Anregung. Ergebnis: Transmission, Reflexion, Verlust (Leistungsbilanz, Fehler der Portnormierung), S-Matrix (|S|², Spalte = Anregung), n_eff der Portmoden mit feiner 2D-Referenz,
beim Koppler cross, bar, Kopplungslänge nach der Theorie gekoppelter Moden (Netz der Rechnung und feines Referenznetz) sowie |E|² im Längs- und Querschnitt. Eigene Datei: Flächenmarken als PEC-Wand oder Port
(mit Modenzahl) zuweisen (z. B. das Ringnetz von `examples/ring_resonator_3d/ring_mesh.py`; den Ring selbst baut das Bauteil „Ringresonator“, siehe unten).
**Grenzen:** 3D-Wellenleiter sind teuer. n_eff des 450 × 220 nm Si/SiO₂-Streifens ist mit p = 1 und 150-nm-Zellen um 9 % zu klein (2,14 gegen 2,36); mit p = 2 und 10 000 Unbekannten dauert die Rechnung 3 Minuten.
Ein Richtkoppler mit p = 1 und 150-nm-Zellen koppelt viel zu stark (Kopplungslänge 10 µm gegen 25 µm auf dem feinen Netz), cross ist bei kurzen Kopplern sehr empfindlich auf den Phasenfehler. Die Vorlagen
zeigen also die Mechanik, keine konvergierten Bauteile; Netz und Ordnung muss man prüfen (die Tabelle zeigt dafür die Referenzen). Halbflächen-Ports verfälschen cross zusätzlich um einige Prozent.

**Ringresonator (Bauteil „Ringresonator (Add-Drop, Gmsh)“):** Add-Drop-Filter nach `examples/ring_resonator_3d` der Bibliothek: ein Ring zwischen zwei geraden Zuleitungen gleichen Querschnitts, vier Ports
(1 Eingang, 2 Through, 3 Drop, 4 Add), Anregung an Port 1. Reiter 1: Kernbreite und -höhe, Ringradius (Mittellinie), Spalt, Ränder (längs der Zuleitungen, neben den Zuleitungen, oben und unten),
Materialien Kern/Mantel. Reiter 2: Durchlauf **fest** (von, bis, Zahl) oder **automatisch** (Zielwellenlänge; neun Wellenlängen im Suchfenster um die Resonanz des isolierten Rings, dann neun innerhalb ±2
Halbwertsbreiten des angepassten Pols; das Material ist dabei bei der Zielwellenlänge eingefroren), Zelle am Kern und fern, Staffelweg, **halbes Modell** (spart etwa ein Drittel der Unbekannten, die Mittelebene
z = h/2 ist eine magnetische Wand, die S-Parameter sind die des ganzen Bauteils), Ordnung p am Kern und fern mit Reichweite, Löser, Referenz und FSR, Feld beim stärksten Drop. Vor „Rechnen“ unbedingt
„Netz prüfen und Speicher schätzen“: das Netz wird mit Gmsh erzeugt (einige Sekunden), die Schätzung der Bibliothek steht daneben (Speicher, Zeitrichtwert, Warnung bei über 60 % des freien Speichers, harte Grenze).
Ergebnis: Spektrum von Through, Drop, Add und Reflexion mit der angepassten Resonanz (strichpunktiert) und der des isolierten Rings (gepunktet); Tabelle mit Lage, Halbwertsbreite, belasteter Güte, größtem
Drop, kleinstem Through, Rest der Anpassung, Referenz-λ, FSR aus dem isolierten Ring und FSR = λ²/(n_g·2πR) aus dem Gruppenindex der geraden Zuleitung; S-Matrix der Anregung (|S|², Phase); |E|² in der Kernmitte; CSV.
Die Resonanz ist der gemeinsame komplexe Pol von Through und Drop (kleinste Quadrate, mindestens vier Wellenlängen; bei großem Rest der Anpassung warnt die Seite).
**Vorlagen:** „Test“ (R = 1,5 µm, 150-nm-Zellen, p = 1, halbes Modell: etwa 17 000 Unbekannte, 0,4 GB, einige Sekunden je Wellenlänge) prüft nur die Rechenkette; mit p = 1 ist der Resonator nicht aufgelöst
(Bibliothek: n_eff 7 % zu klein, Scheinresonanzen alle paar Nanometer). „mittel“ (R = 2 µm, p = 2 am Kern): rund 80 000 Unbekannte, nach der Schätzung 7,8 GB mit SparseLU, 18 Wellenlängen bei einem Thread
sind viel Zeit (nicht gelaufen). „Bibliotheksbeispiel“ (100-nm-Zellen, p = 3/2): 0,4 bis 1,7 Millionen Unbekannte; die Bibliothek rechnet das mit MUMPS auf einem Server, das Wheel hat nur SparseLU
(nicht gelaufen). **Grenzen:** das 3D-Spektrum und die Güte sind hier nicht gegen die Bibliothek validiert; jede Wellenlänge ist eine volle Lösung (keine Faktorisierung wird geteilt, weil die Matrix von ω abhängt);
bei kleinem Radius weicht die FSR-Schätzung aus der geraden Zuleitung stark vom isolierten Ring ab (R = 1,5 µm: 62 gegen 85 nm), für R = 2 µm nennt die Bibliothek 45 bis 48 nm, die Schätzung aus n_g = 4,15 ergibt rechnerisch 46,5 nm (hier nicht nachgerechnet); die Referenz nimmt
die tiefste Mode je Azimutordnung als die TE-ähnliche Grundmode; verlustfreie Materialien und PEC-Wände: Energiebilanz Σ|S|² = 1 prüft die Portnormierung, nicht das Netz.

## Wellenleiter und Ports (Seitenleiste „Art des Modells“ → „Wellenleiter und Ports“)

Ein zweidimensionales Bauteil in der Ebene (x, y), vier Reiter wie bei den anderen Modi. Reiter 1: Materialien (isotrop), Hintergrund (Mantel), Gebiet (Länge L in x, Breite B in y) und
**Formen** (Streifen, Rechteck, Scheibe, Ring, Polygon; spätere Formen überdecken frühere, Pfeile ändern die Reihenfolge). Reiter 2: Wellenlänge oder Durchlauf, Netz (Zellen je Wellenlänge,
Faktor an Grenzflächen), Ordnung, Anregungskanal und Feldkarten; das Netz erzeugt die App mit Gmsh, die Rechnung läuft im Worker (`fem_wg_worker.py`).

**Aufgabe Ports** (`WaveguidePort`, `s_parameters`). Effektivindex-Modell: die Chipebene mit Polarisation H_z, das Feld (E_x, E_y) liegt in der Ebene. Die Ports sitzen auf den Rändern x = 0 (links)
und x = L (rechts); die Wände y = ±B/2 sind ideale Leiter und müssen mehrere Abklinglängen von den Wellenleitern entfernt sein (die Prüfung warnt unter 300 nm). Die Portmoden sind die geführten
Moden des Querschnitts der Portlinie (1D-p-FEM); je Port wählt man, wie viele Moden entwickelt werden (geführte zuerst). Ergebnis (Reiter 3): **Spektrum** (Transmission, Reflexion und Verlust des
Anregungskanals), **S-Matrix** (|S|² oder Phase, Spalte = Anregung, alle ausbreitungsfähigen Kanäle), **Portmoden** (n_eff über der Wellenlänge) und **Feld** (|E|, Re E_x, Re E_y der Anregung); CSV-Export.
Ein verlustfreies Bauteil muss Σ|S|² = 1 liefern; weicht der Verlust davon ab, ist das Netz zu grob.

**Aufgabe Moden** (`PropagatingMode`). Der Querschnitt (Breite in x, Höhe in y) liegt in einem PEC-Kasten; gesucht werden die geführten Moden eines geraden Wellenleiters (vektorielles Eigenproblem,
keine Geistermoden): n_eff und β je Mode, Felder |E_t|, |E_z|, Re E_x, Re E_y. Verlustfreie Materialien.

**Gemessen (hpfem 0.5.0, 20.11.2026, p = 3).** Gerader Streifen 450 nm, n = 2,4 in 1,44, λ = 1550 nm, Gebiet 3000 × 3000 nm (1250 Dreiecke, 2,5 s): T = 1,0000, R < 10⁻⁶, der Portmode hat
n_eff = 1,98555 gegen 1,98555 aus der Dispersionsgleichung des Schichtwellenleiters (TM, Abweichung unter 2·10⁻⁴), S reziprok und unitär. Stufe 450 → 900 nm: R = 0,0036, T = 0,9964. Querschnitt
450 × 220 nm, n = 3,48 in 1,44 (2148 Dreiecke, 29 s): n_eff = 2,356, 1,731, 1,447, 1,367 (der Grundmode liegt im üblichen Bereich von 2,3 bis 2,4 für einen Siliziumdraht in Quarz).

**Grenzen.** Gerade Elemente an den Rändern der Formen (Merkmale unter etwa einem Drittel der Zellgröße werden nicht aufgelöst, Gitterzähne und kleine Spalte brauchen „Faktor an Grenzflächen“ ≈ 0,5),
isotrope Materialien, nur Ports links und rechts (keine Ports an den oberen und unteren Wänden); dreidimensionale Ports (`PortModes3D`) gibt es im Modus „3D-Strukturen“. Resonante Bauteile (Ringe, Koppler) sind
nicht vorbereitet; sie brauchen feine Netze (Spalt und Steg aufgelöst) und viele Wellenlängen. Eine Vorlage dafür ist nicht mitgeliefert, weil ich eine Resonanz hier nicht nachgerechnet habe.

## Wärme und Mechanik (Seitenleiste „Art des Modells“ → „Wärme und Mechanik (2D und 3D)“)
Der Modus koppelt drei hpfem-Löser in zwei Dimensionen (Ergebnisse je Meter in der unendlichen Richtung); über den Schalter „Raumdimension“ oben gibt es dieselbe Kette in 3D (siehe unten). Reiter 1 wählt den Fall.

* **Nanodraht: Optik → Wärme → Spannung.** Materialien (optische Daten wie sonst, dazu je Material Wärmeleitfähigkeit κ, E-Modul, Querkontraktion ν, Ausdehnung α und
  das thermo-optische dε/dT), Draht und Matrix (die Matrix muss verlustfrei sein), Wellenlänge, Feldamplitude E₀ (die App zeigt die Intensität), Radius und Gebiet. Das
  Gebiet ist ein Quadrat mit der PML außen; **dort sitzt auch die Wärmesenke T = T_ref**. Mit „selbstkonsistent“ rechnet die Bibliothek die Optik so oft neu, bis sich
  die Temperatur um weniger als die Toleranz ändert (Reiter 3, Unterreiter „Konvergenz“). Mit „Wärmespannung berechnen“ folgt aus dem Temperaturfeld eine Rechnung der
  Elastizität in ebener Dehnung (Rand eingespannt oder frei; Schrumpfung des Drahts optional).
* **Block** und **Zylinder.** Abmessungen und Zellzahl, Randbedingungen je Seite (frei, eingespannt, Symmetrie, Zug/Druck mit Spannung in MPa; positiv = Zug), ΔT am Anfang
  und Ende (linear dazwischen) und Schrumpfung des ganzen Körpers. Beim Zylinder ist die Achse r = 0 fest (u_r = 0), x ist r und y ist z.

Reiter 2 setzt Ordnung und Rasterauflösung der Karten, Reiter 3 zeigt Kennzahlen mit den Prüfungen (Block und Zylinder mit Symmetrie links und unten gegen die geschlossene Lösung;
Draht ohne Rückkopplung gegen die Linienquelle), Feldkarten, Schnitte und das verformte Netz mit wählbarer Überhöhung.

Wichtig: Die Temperatur gilt für das gewählte Gebiet (in 2D wächst sie logarithmisch mit dem Abstand der Senke). Temperatur und Schrumpfung wirken als Eigendehnung in der
Ebene (ε*_zz = 0), die Von-Mises-Spannung nimmt σ_zz = ν(σ_xx + σ_yy). Es gibt nur den stationären Fall, keine Konvektion und Strahlung, keine temperaturabhängigen Materialdaten
außer ε(T), und die Optik rechnet auf dem unverformten Netz. Die Materialvorgaben sind grobe Richtwerte; die Vorlage „selbstkonsistent“ trägt ein Beispiel-dε/dT, keinen Messwert.

### 3D (Schalter „Raumdimension“ = 3D)
Zwei Fälle mit `Scattering3D`, `Thermal3D`, `ThermoOptical3D` und `Elasticity3D` (Ergebnisse in W, J, K, nm, MPa).

* **Kugel: Optik → Wärme → Spannung.** Materialien wie in 2D (optische Daten, κ, E, ν, α, dε/dT), Kugel und Matrix (verlustfrei), Wellenlänge, E₀, Radius, Innenbreite und Außenbreite in Radien
  (der Rest ist die PML; **an den Außenflächen der Box sitzt die Wärmesenke**), Zellen je Radius. Die Welle läuft in +z, E längs x. Das **Viertelmodell** rechnet nur x > 0, y > 0 (Faktor 4 für Leistung und Energie,
  die Felder werden gespiegelt) und spart etwa einen Faktor 4 an Speicher. „Selbstkonsistent“ löst die Optik je Iteration neu (bei der Vorlage 5 Iterationen, rund 65 s). „Wärmespannung“ rechnet `Elasticity3D`
  (Außenflächen eingespannt oder frei, Schrumpfung der Kugel optional).
* **Quader.** Breite, Tiefe, Höhe und Zellen je Richtung; Randbedingungen je Fläche (frei, eingespannt, Symmetrie, Zug/Druck in MPa); ΔT bei z = 0 und z = H (linear) und Schrumpfung bei z = 0 und z = H
  (linear; „Schrumpfblock auf dem Substrat“ ist das Viertelmodell aus Beispiel R7 der Bibliothek) **oder** eine **Wärmequelle** q (mW/µm³) mit Wärmesenken auf den gewählten Flächen (`Thermal3D`; die übrigen
  Flächen sind adiabatisch), deren Temperatur die Wärmedehnung treibt.

Reiter 2: Polynomordnung, Auflösung der Schnitte, **Speichergrenze** und der Knopf „Netz prüfen und Speicher schätzen“ (Zellen, Unbekannte, `estimate_memory`, Zeitrichtwerte; rot, wenn die Schätzung die Grenze
überschreitet, gelb ab 60 % des freien Hauptspeichers). Reiter 3: Kennzahlen mit den Prüfungen (Quader gegen die geschlossene Lösung, Wärmequelle gegen die eindimensionale Lösung, Kugel gegen die Schätzung
ΔT ≈ P/(4πκ)(1/R − 1/R_s)), Schnitte senkrecht zu x, y oder z mit Schieber, Linienschnitte, das verformte Raster eines Schnitts und die Konvergenz der Schleife.

Wichtig: 3D ist teuer (nur SparseLU, ein Thread; 15 000 Unbekannte der Mechanik brauchen etwa 50 s und 0,6 GB; die Zeitangabe der App ist eine Faustformel). Das Standardnetz ist grob (die Bibliothek meldet eine
unteraufgelöste PML); Konvergenz prüfen. Die Temperatur gilt für das gewählte Gebiet. Wie in 2D: stationär, keine Konvektion und Strahlung, Materialdaten außer ε(T) unabhängig von T, Optik auf dem unverformten
Netz, Maxima sind Rasterwerte, Materialvorgaben sind grobe Richtwerte, die Kette ist nicht gegen Literaturwerte für konkrete Partikel validiert.

## Zeitbereich (Seitenleiste „Art des Modells“ → „Zeitbereich“)
Der Modus löst die Wellengleichung im Zeitbereich mit `hpfem.TimeDomain2D/3D` (hp-Elemente im Raum, implizite Newmark-Regel in der Zeit). Reiter 1: Vorlage, Dimension (2D-Schnitt
oder 3D-Kasten mit Kugel), Materialien (konstante, reelle ε bei λ₀, dazu eine Leitfähigkeit σ in S/m je Material), Gebiet und Formen (2D wie im Wellenleiter-Modus), Rand je Seite
(PEC, absorbierend, PMC). Reiter 2: Zeitsignal (modulierter Gauß-Puls mit Träger λ₀ und Breite τ, Gauß-Puls, Dauerstrich mit weichem Einschalten), Quelle (Position, Gauß-Radius,
Polarisation, Amplitude J₀), Sonden (eine Zeile je Punkt), Zeit (Schritte je Periode der höchsten Frequenz, Δt, Endzeit, höchste Frequenz, Zahl der Feldbilder, Newmark γ und β) und
Netz. Die Kennzahlen (Träger, höchste Frequenz und kürzeste Wellenlänge, Δt, Schritte, Frequenzauflösung 1/T) und das Quellspektrum stehen darunter. Reiter 3 rechnet im eigenen
Prozess (Fortschritt in Blöcken von Schritten, Abbruch möglich). Reiter 4: Feldfilm (Schieber und „Als Film abspielen“), Sonden E(t), Spektrum (FFT, Frequenz oder Wellenlänge,
Hann-Fenster, „Übertragung“ = Sonde durch Quelle), Energie, Tabellen mit CSV.

Wichtig: Die Trapezregel (β = ¼, γ = ½) ist bedingungslos stabil und erhält die Energie; der Zeitschritt bestimmt nur die Genauigkeit (Phasenfehler etwa (ω Δt)²/12 je Bogenmaß).
Der absorbierende Rand ist erster Ordnung (nur senkrechter Einfall vollständig, keine PML). Materialien sind nicht dispersiv (Metalle mit ε < 0 gehen nicht). Das Netz richtet sich
nach der kürzesten Wellenlänge des Pulses (aus Träger und Breite), ein sehr kurzer Puls braucht also ein feines Netz. Ein Rechenschritt kostet bei vielen Unbekannten mehr als bei
wenigen (die Energie wird dann nicht in jedem Schritt berechnet). 3D: nur `box_with_ball` (Würfel mit Kugel), Ordnung und Zellen je Radius klein halten. Die Spektren sind nicht gegen
eine Rechnung im Frequenzbereich geprüft; sie gelten im Band der Quelle und wenn das Signal bis zum Rechenende abgeklungen ist (sonst Hann-Fenster oder längere Rechnung).

## Bandstruktur (Seitenleiste „Art des Modells“ → „Bandstruktur“)
Der Modus rechnet die Bänder eines zweidimensionalen photonischen Kristalls mit `hpfem.BandStructure2D` (Floquet-Bloch-Eigenproblem der Einheitszelle). Reiter 1: Vorlage,
Gitter (quadratisch oder hexagonal), Materialien mit reellem ε, Formen in der Zelle (Scheibe, Rechteck, Ring, Polygon; spätere überdecken frühere, Angaben relativ zur Zellmitte
in Einheiten der Gitterkonstante a, im ganzen Gitter wiederholt). Reiter 2: Punkte je Pfadsegment, Zahl der Bänder, Netz (Zellen je Wellenlänge, Ordnung, größte Frequenz f_max),
optional die Modenfelder an den Eckpunkten des Pfads. Reiter 3: das Bandstrukturdiagramm f = ω a / (2π c) = a / λ entlang Γ–X–M–Γ bzw. Γ–M–K–Γ, schraffiert die Lücken über den
ganzen Pfad mit Δf/Mitte in %, die Tabelle (CSV), das leere Gitter |k + G| als graue Referenz und die Felder |E|² oder Re E_x, Re E_y eines Bandes an einem Eckpunkt.

Wichtig: Der Löser rechnet nur die Polarisation mit E in der Ebene (H_z aus der Ebene; in Joannopoulos et al. „TE“). Die bekannte TM-Lücke (E_z) quadratischer Stabgitter erscheint
deshalb nicht; die Vorlage mit Stäben (ε = 12, r = 0,2 a) zeigt entsprechend keine Lücke. Lochgitter öffnen Lücken für diese Polarisation. Es gibt keinen Lichtkegel: die Struktur ist in z
unendlich ausgedehnt, alle Moden sind gebunden. Es sind nur konstante, verlustfreie Materialien möglich. Das Band bei f = 0 beginnt bei Γ; der zweite Nullwert, den der Löser dort
liefert (konstantes Feld), wird entfernt. Die Ableitungen der Bänder (Gruppengeschwindigkeit, Empfindlichkeit gegen ε und Form) hat die Bibliothek (`group_velocity`,
`band_permittivity_derivative`, `band_shape_derivative`), die Oberfläche zeigt sie noch nicht.

Gemessen (kurze Läufe, keine Validierungsstudie): das leere Gitter trifft die Bänder |k + G| (quadratisch X: 0,5000 / 0,5000 / 1,118, M: 0,7071; Dreiecksgitter M: 0,5774, K: 0,6667), das Dreiecksgitter
aus Luftlöchern (ε = 13, r = 0,4 a, 380 Zellen) hat eine Lücke zwischen Band 1 und Band 2 von f = 0,2356 bis 0,3948 (Δf/Mitte ≈ 50 %); die Werte an M und K ändern sich zwischen p = 3 und p = 5 in den
ersten vier Stellen nicht. Einen Vergleich mit Literaturwerten habe ich nicht gemacht.

## Studien: Optimierung, Rekonstruktion, UQ (Seitenleiste „Art des Modells“ → „Studie“)
Eine Studie rechnet eine Gitterzelle für viele Parameterwerte mit dem Job-Runner der Bibliothek (`python -m hpfem.run`). Die Zelle wird im
Job-Dokument beschrieben (JSON, Schema 2: Zelle, Netz, Materialien, Stapel, Parameter mit Grenzen, Konfigurationen aus Wellenlänge, Winkeln,
Polarisation und Ordnungen), nicht im Modell der anderen Modi. Das Netz wird für jeden Parameterwert verformt, nicht neu erzeugt.

1. **Reiter 1 „Studie“:** eine Vorlage wählen (Lamellenzelle mit Stegbreite und ε als Parametern, etwa eine Sekunde je Auswertung) oder eine
   Job-Datei laden und das JSON bearbeiten; Fehler zeigt die Seite sofort. „Studie starten“ rechnet im Ordner `<Arbeitsordner>/study/<Name>`.
   „Abbrechen“ hält nach der laufenden Auswertung an; liegt im Ordner schon ein Studienspeicher, setzt „Studie fortsetzen“ dort fort (bereits
   gerechnete Punkte kommen aus dem Speicher). Ältere Studien öffnet „Vorhandene Studie öffnen“.
2. **Verlauf:** eine Messgröße über den Auswertungen mit dem bisher besten Wert, die Parameter je Auswertung mit Grenzen und Ergebnis, die
   abgetasteten Punkte in zwei frei wählbaren Größen. Fehlgeschlagene Auswertungen stehen mit Grund da; „MeshQualityError“ heißt, dass das
   verformte Netz an diesen Werten ungültig ist (Grenzen enger setzen).
3. **Pareto-Front:** die nicht dominierten Auswertungen zweier Messgrößen (je Maximum oder Minimum) — der Kompromiss zwischen zwei Zielen aus
   allen gerechneten Punkten.
4. **Ergebnis:** Optimum; bei der Rekonstruktion die angepassten Parameter mit Unsicherheiten der Laplace-Näherung, Korrelationen, Residuen und
   mit `posterior` die emcee-Randverteilungen (68 %- und 95 %-Bereich); bei UQ Mittel, σ und Quantile jeder Messgröße und die Sobol'-Indizes
   (erste Ordnung und total) als Balken.

**Nebenbedingungen, Mehrziel, Mehrstufen (Optimierung).** Der Job-Runner der Bibliothek kennt nur ein Ziel ohne Nebenbedingungen. Das Formular
„Nebenbedingungen und Mehrziel“ in Reiter 1 (sichtbar bei der Aufgabe Optimierung) schreibt die Erweiterungen ins Job-Dokument; solche Studien laufen über
den Studien-Worker der App (`fem_study_worker.py`, der Runner mit ersetztem Optimierungsschritt, hpfem 0.5.x), alle anderen wie bisher über `hpfem.run`.
- **Nebenbedingungen an Messgrößen** (`optimize.constraints`, Messgröße zwischen `lower` und `upper`, nur Verfahren bayesian): ein eigener Gauß-Prozess
  je Nebenbedingung, die Suche gewichtet die erwartete Verbesserung mit der Wahrscheinlichkeit der Zulässigkeit.
- **Nebenbedingungen an Parameter** (`optimize.linear_constraints`, Koeffizienten in Job-Einheiten, z. B. `width: 1, eps: 20` zwischen von und bis): unzulässige
  Punkte werden nie gerechnet, für alle Verfahren.
- **Mehrziel** (Verfahren `pareto`, mindestens zwei `objectives`, `maximize` je Ziel, optional `ref_point`): Bayes-Optimierung mit BoTorch
  (`pip install "hpfem[opt-bo]"`, installiert auch PyTorch). Ohne BoTorch lässt sich die Option wählen, aber nicht übernehmen oder starten. Das Ergebnis ist
  die Pareto-Front (Tabelle, CSV, Diagramm mit Referenzpunkt in Reiter 3 und 4). `raw_samples` und `num_restarts` im Block `optimize` steuern den
  Aufwand je Schritt (Vorgabe der Bibliothek 512 und 10: bei vielen Kernen Minuten je Schritt).
- **Mehrstufen** (Verfahren `multi_fidelity`, `fidelities` z. B. `[{"order": 2}, {"order": 3}]`, billigste zuerst, optional `costs`, `max_cost`): grobe
  Rechnungen suchen, die letzte Stufe bestätigt; Ergebnis ist die beste Auswertung der Zielstufe.
Auswertungen, die eine Nebenbedingung an Messgrößen verletzen, sind in Verlauf und Pareto-Reiter rote Kreuze und zählen nicht als „bestes“. Die Vorlage
„Optimierung mit Nebenbedingungen“ (acht Auswertungen, etwa 25 s) zeigt den Aufbau. Ganzzahlige oder kategoriale Entwurfsvariablen gibt es nicht
(der Evaluator der Gitterzelle kennt nur Geometrie- und Materialparameter).

Die Vorlagen sind mit der Bibliothek durchgerechnet: die Rekonstruktion findet aus synthetischen Daten (Breite 210 nm, ε 2,3, Rauschen 0,002)
210,5 ± 3,6 nm und 2,303 ± 0,014.

## Photovoltaik (Reiter 3 → Expander „Photovoltaik“, Reiter 4 → „Photovoltaik“)
Für Streuung im periodischen Modus (nicht isoliert, Löser mit exakter Absorption). Im Reiter 3 „Generationsrate im Absorber berechnen“ einschalten, den
Absorber wählen (das Material, dessen Absorption Ladungsträger erzeugt) und die Stützstellen des Tiefenprofils. Jeder Punkt liefert dann zusätzlich die
Generationsrate G = absorbierte Leistung der Zelle / (ħω · Zellfläche) im Absorber, seitlich über die Periode gemittelt als Profil G(y) für die einfallende
Intensität 1 W/m² (ohne PML; kein zusätzlicher Lösungsschritt). Unter den Ergebnissen gewichtet der Unterreiter „Photovoltaik“ die Punkte eines
Wellenlängendurchlaufs mit einem Spektrum: Schwarzkörper 5778 K (auf 1000 W/m² über 280–4000 nm skaliert, eine Näherung ohne Absorptionslinien) oder ein eigenes
CSV (`λ [nm]; Irradianz [W/(m² nm)]`, z. B. das Normspektrum AM1.5; es ist keines eingebaut). Ausgabe: J_sc = q Σ A_Absorber Φ (Φ = I λ/(h c), Bandbreiten aus den
Mittelpunkten der Punkte; jedes absorbierte Photon erzeugt ein Paar, alle werden gesammelt: obere Grenze, keine Ladungsträgersimulation), der Wert bei A = 1 im
gleichen Bereich, die Absorption über λ, das spektral gewichtete G(y) und beides als CSV. Der Durchlauf sollte das Spektrum dicht genug abdecken (die
Wellenlängen außerhalb des Spektrums tragen nichts bei, die Randbänder bekommen den vollen Punktabstand wie in `hpfem.pv.spectral_weights`).

## Ableitungen (Reiter 3 → Reiter 4 „Ableitungen“)
Mit dem Löser „hpfem.grating“ und „Ableitungen berechnen“ berechnet die Bibliothek (`hpfem.grating.jacobian`, M16) die Jacobi-Matrix aller
Beugungseffizienzen nach Re ε und Im ε jedes Materials in der Zelle (Formen und Schichten, nicht Einfallsmedium und Substrat), nach der Wellenlänge
(je nm, bei festem ε), nach θ und bei vektorieller Rechnung nach φ (je Grad). Das kostet einen zusätzlichen Lösungsschritt auf der behaltenen
Faktorisierung, aber keine weitere Rechnung. Der Reiter zeigt dR/dp und dT/dp der Summen (Balken bei einem Punkt, Kurven über einem Durchlauf) und die
ganze Matrix als Tabelle und CSV. Anwendung: Toleranzen (ΔR ≈ Σ dR/dp · Δp), Empfindlichkeit auf Materialdaten, Startwerte für Fits und Optimierung.

## Dispersive Materialien: Drude-Lorentz-Anpassung
Der Materialtyp „Drude-Lorentz (Fit)“ beschreibt ε(ω) = ε∞ − ωp²/(ω² + iγω) + Σ f_j ωp²/(ω_j² − ω² − iγ_j ω) mit Energien in eV (Konvention exp(−iωt), Verlust
Im ε > 0). Im Materialeditor eines Bibliotheks-, Tabellen- oder Drude-Materials öffnet „An Messdaten anpassen (Drude-Lorentz)“ die Anpassung: Wellenlängenbereich
und Zahl der Oszillatoren wählen, „Anpassen“ rechnet im Python mit hpfem (einige Sekunden), das Diagramm zeigt Daten und Fit, darunter steht der größte relative
Fehler von ε auf den Stützstellen. Die Anpassung ist nur im gewählten Bereich brauchbar; mehr Oszillatoren passen enger an, können aber überanpassen. Der
Hauptnutzen ist die Aufgabe „Resonanzen“ mit dem Haken „Dispersion selbstkonsistent“: Ohne ihn wertet hpfem ε bei der Zielwellenlänge aus und friert es ein, mit
ihm wird jede Mode nachgeführt (ε bei der komplexen Eigenfrequenz der Mode, Newton-Verfahren). Das verschiebt λ_res und Q bei Metallen und nahe Absorptionskanten
merklich; die Tabelle zeigt beide Werte. Tabellen werden dabei bei Re ω ausgewertet (nicht analytisch fortgesetzt), ein Drude-Lorentz-Material ist genauer. Kann eine
Mode nicht nachgeführt werden, bleibt die eingefrorene stehen und das Protokoll meldet eine Warnung. Verkippte optische Achsen von Kristallen gibt es im periodischen
Modus schon (Kristallmaterial: einachsig mit θ, φ, zweiachsig mit Euler-Winkeln, voller Tensor).

## Resonanzen und Bänder (Reiter 3, Aufgabe „Resonanzen“)
`hpfem.grating.resonances` sucht die Eigenmoden der offenen Zelle (PML oben und unten, Bloch-periodisch) nahe der Wellenlänge der Beleuchtung, bei der
Bloch-Wellenzahl kx = k₀ n sin θ cos φ (und β = k₀ n sin θ sin φ entlang der Linien). Ergebnis je Mode: komplexe Frequenz ω,
Resonanzwellenlänge λ_res = 2πc / Re ω und Güte Q = Re ω / (−2 Im ω). Ein Durchlauf über θ oder φ ergibt die Bandstruktur (λ_res über kx·P/2π, Farbe Q).
Mit „Feldkarten speichern“ werden die Modenfelder der gewählten Punkte gespeichert (auf max |E| = 1 normiert). Moden mit Im ω > 0 (Q < 0) oder extrem
großem Q sind meist PML- oder Kastenmoden; das Modenfeld zeigt es.

## Formen verschieben und zentrieren (Reiter 1)
Unter der Periode: **„Alle Formen in x verschieben um (nm)“** mit „Verschieben“, und **„Zentrieren“**, das die Mitte der x-Ausdehnung aller Formen in die Zellmitte P/2 legt. Die Schichten
hängen nicht von x ab; Formen, die über den Zellrand ragen, setzen sich periodisch fort. Mit **„Beim Ändern der Periode die Formen mitzentrieren“** wandern die Formen um die halbe Änderung
der Periode mit, die Struktur bleibt dann mittig.

## Neu mit hp-FEM M15 (Stand 7.10.2026): schnellere Karten, exakte Elemente, exakte Absorption
Der Worker nutzt, wenn der Build es kann, `problem.sample` (alle Kartenpunkte in einem Aufruf, parallel), `problem.triangulate` und `absorbed_power_by_tag`; bei einem älteren Build
rechnet er wie vorher (Punktschleife, Rasterintegration) und das Protokoll zeigt „field map … by loop“. Dafür hpfem aktualisieren (`git pull`, Python-Modul neu bauen und kopieren).
* **Feldkarten, „Elemente (exakt)“:** das Feld auf den unterteilten Dreiecken des Netzes. Sprünge der Normalkomponente an Materialgrenzen bleiben scharf, das Material ist je Dreieck bekannt,
  die absorbierte Leistungsdichte Q hat deshalb keine Pixelfehler. Umschalten auf „Raster“ ist möglich (nur dort lässt sich das Netz einzeichnen; Schnitte nutzen das Raster).
  Einstellungen in Reiter 3 unter Ausgabe: „Elementdarstellung speichern“ und die Unterteilung je Dreieck (3 bis 4 genügt für p = 4 bis 5).
* **Absorption je Material, exakt:** Volumenquadratur der Joule-Wärme Q = ½ ω ε₀ Im ε |E|² auf dem Netz (nur das physikalische Gebiet, PML-Zellen ausgenommen), bezogen auf die einfallende
  Leistung je Periode. Sie steht für **jeden Punkt eines Durchlaufs** zur Verfügung, ohne dass Karten gespeichert werden: Diagramm „Absorption je Material über den Durchlauf“, Tabelle im Reiter 4
  „Absorption“ und Spalten `A[Material]` in der Ergebnistabelle. Die Summe wird mit der Energiebilanz verglichen (Verhältnis nahe 1); die Rasterwerte bleiben als Vergleich, wenn Karten gespeichert sind.

## Wichtig für sinnvolle Ergebnisse
* **Netz zur Ordnung:** bei Polynomordnung p genügen etwa 12/p Elemente pro Wellenlänge im Material (p = 4: 3). Mehr Elemente erhöhen die Freiheitsgrade stark.
  Die App zeigt eine Schätzung und warnt über 400 000.
* **PML auslegen:** Die Bibliothek verlangt, dass das gedehnte Feld in der PML aufgelöst ist: k₀·n·|s|·h ≤ Grenze (3 für p ≥ 4, sonst 0,75·p), mit |s| = √(1 + σ̂²) und
  σ̂ = −(m+1)·ln R₀ / (2·k₀·n_ref·d). Ein strenger PML-Zielfehler bei großem Beugungswinkel (die PML wird für den größten Winkel der propagierenden Ordnungen ausgelegt)
  macht R₀ winzig und σ̂ groß; eine dünne PML braucht dann extrem feine Zellen, sonst meldet der Solver „the PML is under-resolved“. Die App zeigt die Auslegung im Reiter 3
  („PML-Auslegung“: Dicke, σ̂, |s|, nötige und vorhandene Zellgröße) und warnt vor der Rechnung. **„Vorschlag aus der Wellenlänge“ (Reiter 1)** legt die PML-Dicke aus
  Polynomordnung, PML-Zielfehler und „Elemente pro Wellenlänge in der PML“ (Reiter 2, Voreinstellung 6) aus; das PML-Netz wird entsprechend fein. Nach einer Änderung von p oder des
  Zielfehlers den Vorschlag erneut anwenden. Voreinstellung des PML-Zielfehlers: 1e-3; für genaue Vergleiche (z. B. hp-Test) 1e-4 oder kleiner, dann wird die PML dicker.
* **Periodische Ränder bei hp-Verfeinerung:** Seit hp-FEM M15 F16 Stufe 1 (Stand 7.10.2026) deklariert der Worker die Randpaare mit `AdaptiveMesh.set_periodic`, die Bibliothek spiegelt dann jede Verfeinerung auf die Gegenseite. Bei einem älteren Build gilt die Reparatur unten (das Protokoll zeigt „NOTE: this hpfem has no AdaptiveMesh.set_periodic“). Hintergrund: Der Löser verlangt, dass linke und rechte Seite Facette für Facette gleich fein sind. Die Verfeinerung eines Nachbarn kann eine Randzelle
  nur auf einer Seite mitverfeinern; die App gleicht die Seiten nach jedem Schritt an (die gröbere Seite wird nachverfeinert). Gelingt das nicht, endet die Schleife mit dem Ergebnis des
  letzten vollständigen Schritts und einem Hinweis im Reiter „hp-Adaptivität“.
* **Unterer Rand:** ein verlustfreies (oder fast verlustfreies) Substrat braucht unten eine PML. Eine Metallwand ist nur sicher, wenn das Feld bis dahin
  auf e⁻⁶ abgeklungen ist (sechs Eindringtiefen; bei Si und 405 nm sind das 1480 nm). Der Vorschlag aus der Wellenlänge beachtet das.
* **Runde Formen** nicht tangential an eine Grenzfläche legen (Kreis auf Substrat: 1 bis 2 nm einsinken lassen), sonst entstehen verzerrte Dreiecke. Die App warnt.
* **Bibliotheksdaten** gelten nur in ihrem Wellenlängenbereich (z. B. TiO2 430 bis 1530 nm); außerhalb meldet die App einen Fehler vor der Rechnung.
* **Absorption je Material** wird aus dem Feld auf dem Kartenraster integriert (Fehler der Größe eines Pixels an scharfen Materialgrenzen). Das Verhältnis zur
  Energiebilanz zeigt, ob die Kartenauflösung reicht; die normierte Spalte verteilt die verlässliche Energiebilanz nach dem Feld.
* **T und A:** bei verlustbehaftetem Substrat ist nur R messbar; A inkl. Substrat = 1 − R. Bei verlustfreiem Substrat sind T (Ordnungen im Substrat) und A = 1 − R − T verfügbar.

## Neu mit hp-FEM 0.4 und M16 (Stand 9.10.2026)
* **Löser „hpfem.grating“**, hp-Adaptivität für TE, TM und konischen Einfall (Residuum oder DWR), Resonanzen und Bänder, Ableitungen: siehe oben.
* **Prüfen vor dem Rechnen:** „Mit hpfem prüfen“ (Reiter 3) ruft `hpfem.grating.validate` am ersten und letzten Punkt des Durchlaufs auf (Netz,
  periodische Ränder, PML-Dicke und -Auflösung, Elemente je Wellenlänge für p, Materialbereich der Tabellen, verlustbehaftetes Einfallsmedium,
  PEC-Wand zu nah im verlustbehafteten Substrat) und prüft an allen Punkten auf streifende Ordnungen; dazu die Speicherschätzung (Freiheitsgrade,
  Matrix und Faktoren). Dieselben Meldungen erscheinen während der Rechnung und im Reiter 4 „Prüfung, Bilanz, Zeit“.
* **Fortschritt und Abbruch:** die Fortschrittsanzeige nennt die Phase des Lösers; „Abbrechen“ legt die Datei `cancel` in den Rechenordner, der Worker
  hält nach der laufenden Phase an (bei großen Problemen also nicht erst nach dem ganzen Punkt). „Sofort beenden“ beendet den Prozess hart.
* **Magnetfeld und Energiefluss:** die Feldkarten enthalten |H|, die Komponenten von H und den zeitgemittelten Poynting-Vektor S (Komponenten mit
  Vorzeichen, Betrag) in Raster- und Elementdarstellung.
* **Energiebilanz:** R + T + A exakt, die Absorption der Bibliothek (alle Zellen einschließlich PML; bei verlustbehaftetem Substrat mit PML darunter
  deshalb größer als die des physikalischen Gebiets) und der relative Rest der Flussbilanz durch die PML-Grenzen als unabhängiges Fehlermaß.
* **Rand-Tags:** das Gmsh-Netz trägt die Ränder jetzt als 1 (links), 2 (rechts), 3 (unten), 4 (oben), wie `hpfem.box_tag`. Netze älterer App-Versionen
  (101 bis 104) neu erzeugen.

## Erster Test des Solvers
1. Vorlage **„Dünnschicht SiO2 auf Si (ebener Stapel …)“**: ohne Struktur. R und T müssen mit den Linien „ebener Stapel“ im Spektrum übereinstimmen.
2. Vorlage **„Si-Lamellengitter (Projektfall …)“**, TM, 50°, 405 nm: R0 und R−1 nahe den Referenzwerten des Projekts (R0 ≈ 0,143, R−1 ≈ 0,142). Die Bibliotheksdaten für Si
   weichen leicht von den BEM-Daten ab (ε = 29,87 + 2,86 i statt 29,63 + 2,77 i), kleine Unterschiede sind deshalb zu erwarten.
3. Konvergenzstudie in p anschalten: R, T, A sollten sich mit p schnell einem Wert nähern.

## Bei Problemen
* „Gmsh fehlt“ in der Seitenleiste: `pip install gmsh` in dem Python, das die App startet.
* Rechnung bricht ab: Protokoll im Reiter 3 öffnen (letzte Zeilen) und die Zeile mit `ERROR` lesen; der Ordner `fem_work/run/` enthält `log.txt`, `job.json` und `mesh.msh`.
* Rechnung läuft nicht an: „Python mit hpfem“ und „hp-FEM-Ordner“ in der Seitenleiste prüfen (`python -c "import hpfem"` im gewählten Python). Meldet die
  Seitenleiste `No module named 'hpfem._hpfem'`, zeigt der hp-FEM-Ordner auf einen Quellordner ohne gebautes Modul: Feld leeren oder hpfem bauen.
* pip bricht mit Verbindungsfehlern zu `pypi.ngc.nvidia.com` ab: eine globale `pip.ini` nennt einen zusätzlichen Index. In der venv eine eigene
  `pip.ini` mit `[global]` / `extra-index-url =` anlegen (siehe README).
