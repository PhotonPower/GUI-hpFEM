"""Optische Materialien der FEM-Modellwerkstatt (ohne hpfem importieren zu müssen).

Ein Material ist ein JSON-fähiges Dict ("spec") mit dem Schlüssel ``type``:

* ``library``: ``{"type": "library", "name": "Si"}``, Namen siehe ``LIBRARY`` (gleiche Daten wie ``hpfem.materials``)
* ``index``:   ``{"type": "index", "n": 1.5, "k": 0.0}``
* ``eps``:     ``{"type": "eps", "re": 2.25, "im": 0.0}``
* ``drude``:   ``{"type": "drude", "eps_inf": 1.0, "omega_p_eV": 9.0, "gamma_eV": 0.07}``
* ``table``:   ``{"type": "table", "rows": [[lambda_nm, n, k], ...]}`` (linear interpoliert, keine Extrapolation)
* ``crystal``: anisotroper Kristall (hp-FEM M19, ADR-0015), konstante Hauptindizes, Tensor im Laborsystem
  (x quer zu den Linien, y längs der Linien, z die Normale nach oben):

  - ``{"type": "crystal", "kind": "uniaxial", "n_o": .., "k_o": .., "n_e": .., "k_e": .., "theta_deg": .., "phi_deg": ..}``:
    optische Achse a = (sin θ cos φ, sin θ sin φ, cos θ) (θ gegen die Normale, φ gegen x), ε = ε_o I + (ε_e − ε_o) a aᵀ
  - ``{"type": "crystal", "kind": "biaxial", "n": [nx, ny, nz], "k": [..], "euler_deg": [α, β, γ]}``:
    ε = R diag(ε1, ε2, ε3) Rᵀ mit R = R_z(α) R_x(β) R_z(γ) (Hauptachsen gedreht, z-x-z)
  - ``{"type": "crystal", "kind": "tensor", "re": 3×3, "im": 3×3}``: voller symmetrischer Tensor

  ``eps_at`` gibt für Kristalle den Hauptwert mit dem größten Realteil (Netzweiten, Schätzungen); den Tensor liefert ``tensor_at``.

Konvention exp(-i omega t): ε = (n + ik)², Verlust bedeutet Im ε > 0. Wellenlängen überall in nm (Vakuum).

Die gemessenen Tabellen (Si, Ag, Au, Al, Ti, Cr, GaAs, MAPbI3) liegen als CSV in ``<hp-FEM>/python/hpfem/data``. Das Modul sucht
diesen Ordner über ``set_repo()``, die Umgebungsvariable ``HPFEM_DATA``, das Arbeitsverzeichnis und den Ordner der App.
Sellmeier-Materialien (SiO2, TiO2, Wasser) und Konstanten (Luft, Vakuum) sind eingebaut.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HC_EV_NM = 1239.841984  # h*c in eV nm

_TABULATED = ("Si", "Au", "Ag", "Al", "GaAs", "MAPbI3", "Ti", "Cr")
# (A, B, C, lam2_numerator, (lo, hi) in um) wie in hpfem.materials
_SELLMEIER = {
    "SiO2": (0.0, (0.6961663, 0.4079426, 0.8974794), (0.0684043**2, 0.1162414**2, 9.896161**2), True, (0.21, 6.7)),
    "TiO2": (5.913, (0.2441,), (0.0803,), False, (0.43, 1.53)),
    "water": (0.0, (5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1),
              (5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1), True, (0.2, 2.0)),
}
_CONSTANT = {"vacuum": 1.0, "air": 1.000293**2}

LIBRARY = ["Si", "Ag", "Au", "Al", "Ti", "Cr", "GaAs", "MAPbI3", "SiO2", "TiO2", "water", "air", "vacuum"]
CRYSTAL_KINDS = ("uniaxial", "biaxial", "tensor")

# Kristall-Vorlagen: konstante Hauptindizes bei der genannten Wellenlänge λ0 (keine Dispersion!), optische Achse längs z
# (c-Schnitt bzw. Symmetrieachse). Literaturwerte (Malitson & Dodge 1972 für Saphir, hpfem.materials.get("Al2O3"); die übrigen aus
# gängigen Handbüchern, auf drei bis vier Stellen). Wellenlänge in nm.
CRYSTAL_PRESETS = {
    "Saphir Al2O3 (c-Schnitt, 589 nm)": dict(lam_nm=589.3, n_o=1.7681, n_e=1.7600),
    "Quarz SiO2 (einachsig +, 589 nm)": dict(lam_nm=589.3, n_o=1.5443, n_e=1.5534),
    "Kalkspat CaCO3 (einachsig −, 589 nm)": dict(lam_nm=589.3, n_o=1.6584, n_e=1.4864),
    "Lithiumniobat LiNbO3 (633 nm)": dict(lam_nm=633.0, n_o=2.2864, n_e=2.2028),
    "Rutil TiO2 (einachsig +, 589 nm)": dict(lam_nm=589.3, n_o=2.613, n_e=2.909),
    "Magnesiumfluorid MgF2 (589 nm)": dict(lam_nm=589.3, n_o=1.3777, n_e=1.3895),
}


def crystal_preset(name: str, theta_deg: float = 0.0, phi_deg: float = 0.0) -> dict:
    """Spec (Typ ``crystal``, einachsig) der Vorlage ``name``; die Achse liegt standardmäßig längs z."""
    v = CRYSTAL_PRESETS[name]
    return {"type": "crystal", "kind": "uniaxial", "n_o": v["n_o"], "k_o": 0.0, "n_e": v["n_e"], "k_e": 0.0, "theta_deg": float(theta_deg),
            "phi_deg": float(phi_deg)}

_data_dir: Path | None = None
_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def set_repo(repo) -> None:
    """Nennt den hp-FEM-Ordner (der ``python/`` enthält); leert den Zwischenspeicher."""
    global _data_dir
    _data_dir = Path(repo) / "python" / "hpfem" / "data" if repo else None
    _cache.clear()


def data_dir() -> Path | None:
    cands = []
    if _data_dir is not None:
        cands.append(_data_dir)
    if os.environ.get("HPFEM_DATA"):
        cands.append(Path(os.environ["HPFEM_DATA"]))
    here = Path(__file__).resolve().parent
    for base in (Path.cwd(), here, here.parent):
        cands.append(base / "python" / "hpfem" / "data")
    cands.append(_installed_data())
    return next((c for c in cands if c is not None and (c / "Si.csv").is_file()), None)


def _installed_data() -> Path | None:
    """data/ of an hpfem installed in this Python (wheel), found without importing hpfem."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("hpfem")
        if spec is not None and spec.submodule_search_locations:
            return Path(list(spec.submodule_search_locations)[0]) / "data"
    except Exception:
        pass
    return None


def library_available() -> bool:
    return data_dir() is not None


def _table(name: str):
    if name not in _cache:
        d = data_dir()
        if d is None:
            raise FileNotFoundError(f"Materialdaten für {name} nicht gefunden (python/hpfem/data im hp-FEM-Ordner)")
        rows = [[float(v) for v in line.split(",")] for line in (d / f"{name}.csv").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
        a = np.array(rows)
        _cache[name] = (a[:, 0] * 1000.0, a[:, 1], a[:, 2])  # um -> nm
    return _cache[name]


def _check_range(name, lam, lo, hi):
    if lam < lo * (1 - 1e-12) or lam > hi * (1 + 1e-12):
        raise ValueError(f"{name}: Wellenlänge {lam:g} nm außerhalb des Gültigkeitsbereichs {lo:.0f}–{hi:.0f} nm")


def valid_range(spec):
    """(kürzeste, längste) gültige Wellenlänge in nm oder None (unbegrenzt)."""
    t = spec["type"]
    if t == "library":
        n = spec["name"]
        if n in _TABULATED:
            try:
                w = _table(n)[0]
                return float(w[0]), float(w[-1])
            except FileNotFoundError:
                return None
        if n in _SELLMEIER:
            lo, hi = _SELLMEIER[n][4]
            return lo * 1000.0, hi * 1000.0
        return None
    if t == "table":
        w = [r[0] for r in spec["rows"]]
        return float(min(w)), float(max(w))
    return None


def is_crystal(spec) -> bool:
    """True für einen anisotropen Kristall (Typ ``crystal``)."""
    return spec.get("type") == "crystal"


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def tensor_at(spec, lam_nm: float) -> np.ndarray:
    """Komplexer 3×3-Tensor der relativen Permittivität im Laborsystem (x quer, y längs der Linien, z die Normale); isotrope Materialien
    geben ε·I. Kristalle sind nicht dispersiv (``lam_nm`` wird dann nicht gebraucht)."""
    if not is_crystal(spec):
        return eps_at(spec, lam_nm) * np.eye(3, dtype=complex)
    kind = spec.get("kind")
    if kind == "uniaxial":
        e_o = complex(spec["n_o"], spec["k_o"]) ** 2
        e_e = complex(spec["n_e"], spec["k_e"]) ** 2
        t, p = np.radians(spec["theta_deg"]), np.radians(spec["phi_deg"])
        a = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
        return e_o * np.eye(3, dtype=complex) + (e_e - e_o) * np.outer(a, a)
    if kind == "biaxial":
        e = [complex(n, k) ** 2 for n, k in zip(spec["n"], spec["k"])]
        al, be, ga = np.radians(spec["euler_deg"])
        r = _rot_z(al) @ _rot_x(be) @ _rot_z(ga)
        return r @ np.diag(e) @ r.T
    if kind == "tensor":
        m = np.array(spec["re"], dtype=float) + 1j * np.array(spec["im"], dtype=float)
        if m.shape != (3, 3):
            raise ValueError("Kristall: der Tensor braucht 3×3 Einträge")
        if not np.allclose(m, m.T, atol=1e-12):
            raise ValueError("Kristall: der Tensor muss symmetrisch sein (ε = εᵀ, keine gyrotropen Medien)")
        return m
    raise KeyError(f"unbekannte Kristallart '{kind}'")


AXI_HELP = ("Rotationskörper: der Tensor wird in den Zylinderkomponenten (r, φ, z) gelesen, z ist die Symmetrieachse. Erlaubt sind "
            "einachsige Kristalle mit der optischen Achse längs z (θ = 0°), radial (θ = 90°, φ = 0°) oder azimutal (θ = 90°, φ = 90°), "
            "zweiachsige Kristalle ohne Drehung (Euler 0°: n1 = n_r, n2 = n_φ, n3 = n_z) und diagonale Tensoren. Eine gegen die Achse "
            "verkippte optische Achse bricht die Rotationssymmetrie und wird abgelehnt.")


def axi_diagonal(spec, lam_nm: float = 500.0):
    """(ε_r, ε_φ, ε_z) eines Materials für den Rotationskörper-Löser; ``ValueError`` mit Hinweis, wenn der Tensor in (r, φ, z) nicht
    diagonal ist (verkippte Achse)."""
    t = tensor_at(spec, lam_nm)
    scale = float(np.abs(t).max()) or 1.0
    off = max(abs(t[i, j]) for i in range(3) for j in range(3) if i != j)
    if off > 1e-9 * scale:
        raise ValueError("Der Tensor ist in (r, φ, z) nicht diagonal (optische Achse gegen die Symmetrieachse verkippt). " + AXI_HELP)
    return np.diag(t).copy()


def principal_values(spec, lam_nm: float) -> np.ndarray:
    """Hauptwerte des Tensors (Eigenwerte), nach dem Realteil sortiert."""
    w = np.linalg.eigvals(tensor_at(spec, lam_nm))
    return w[np.argsort(w.real)]


def eps_at(spec, lam_nm: float) -> complex:
    """Komplexe relative Permittivität bei der Vakuumwellenlänge ``lam_nm`` [nm]; für einen Kristall der Hauptwert mit dem größten
    Realteil (für Netzweiten und Schätzungen, der Löser bekommt ``tensor_at``)."""
    lam = float(lam_nm)
    t = spec["type"]
    if t == "crystal":
        return complex(principal_values(spec, lam)[-1])
    if t == "eps":
        return complex(spec["re"], spec["im"])
    if t == "index":
        return complex(spec["n"], spec["k"]) ** 2
    if t == "drude":
        e = HC_EV_NM / lam
        return complex(spec["eps_inf"] - spec["omega_p_eV"] ** 2 / (e**2 + 1j * spec["gamma_eV"] * e))
    if t == "table":
        rows = sorted(spec["rows"])
        w, n, k = (np.array(c, dtype=float) for c in zip(*rows))
        _check_range("Tabelle", lam, w[0], w[-1])
        return complex((np.interp(lam, w, n) + 1j * np.interp(lam, w, k)) ** 2)
    if t == "library":
        name = spec["name"]
        if name in _CONSTANT:
            return complex(_CONSTANT[name])
        if name in _SELLMEIER:
            A, B, C, num, (lo, hi) = _SELLMEIER[name]
            _check_range(name, lam, lo * 1000.0, hi * 1000.0)
            l2 = (lam / 1000.0) ** 2
            n2 = 1.0 + A if num else A
            for b, c in zip(B, C):
                n2 += (b * l2 if num else b) / (l2 - c)
            return complex(n2)
        if name in _TABULATED:
            w, n, k = _table(name)
            _check_range(name, lam, w[0], w[-1])
            return complex((np.interp(lam, w, n) + 1j * np.interp(lam, w, k)) ** 2)
        raise KeyError(f"unbekanntes Bibliotheksmaterial '{name}'")
    raise KeyError(f"unbekannter Materialtyp '{t}'")


def nk(spec, lam_nm: float):
    """(n, k) mit n + ik = sqrt(ε), k ≥ 0."""
    r = np.sqrt(eps_at(spec, lam_nm) + 0j)
    if r.imag < 0:
        r = -r
    return float(r.real), float(r.imag)


def describe(spec) -> str:
    t = spec["type"]
    if t == "library":
        return f"Bibliothek {spec['name']}"
    if t == "index":
        return f"n = {spec['n']:g}, k = {spec['k']:g}"
    if t == "eps":
        return f"ε = {spec['re']:g} {spec['im']:+g} i"
    if t == "drude":
        return f"Drude (ε∞ = {spec['eps_inf']:g}, ħωp = {spec['omega_p_eV']:g} eV, ħγ = {spec['gamma_eV']:g} eV)"
    if t == "table":
        return f"Tabelle ({len(spec['rows'])} Zeilen)"
    if t == "crystal":
        kind = spec.get("kind")
        if kind == "uniaxial":
            return (f"einachsig (n_o = {spec['n_o']:g}, n_e = {spec['n_e']:g}, Achse θ = {spec['theta_deg']:g}°, "
                    f"φ = {spec['phi_deg']:g}°)")
        if kind == "biaxial":
            return f"zweiachsig (n = {', '.join(f'{v:g}' for v in spec['n'])}; Euler {', '.join(f'{v:g}°' for v in spec['euler_deg'])})"
        return "Kristall (voller Tensor)"
    return t
