import json

import pytest

import fem_geometry as fg
import fem_materials as fm


def test_presets_validate_and_layout():
    for name, model in fg.presets().items():
        res = fg.validate(model, model["incidence"]["wavelength_nm"])
        assert isinstance(res, list), name
        assert all(level in ("error", "warning", "info", "ok") or isinstance(level, str) for level, _ in res), name
        lay = fg.layout(model)
        assert lay["y_max"] > lay["y_min"], name


def test_models_are_json_serialisable():
    for model in fg.presets().values():
        json.dumps(model)


def test_preset_materials_evaluate_when_data_present():
    if not fm.library_available():
        return
    for model in fg.presets().values():
        lam = model["incidence"]["wavelength_nm"]
        for spec in model["materials"].values():
            fm.eps_at(spec, lam)


def test_load_model_normalises_foreign_materials():
    import json

    import fem_geometry as fg
    raw = {"name": "fremd", "period_nm": 500.0, "materials": {"Luft": {"n": 1.0}, "Glas": {"n": 1.5, "k": 0.0}, "Abs": {"n": 2.0, "k": 0.1}},
           "cover": "Luft", "substrate": "Glas", "layers": [{"material": "Abs", "thickness_nm": 150.0}], "shapes": []}
    m = fg.load_model(json.dumps(raw))
    assert m["materials"]["Glas"] == {"type": "index", "n": 1.5, "k": 0.0}
    assert m["materials"]["Abs"] == {"type": "index", "n": 2.0, "k": 0.1}
    assert not [x for x in fg.validate(m) if x[0] == "error"]


def test_load_model_rejects_unreadable_material_and_fourier_grating():
    import json

    import fem_geometry as fg
    with pytest.raises(ValueError, match="Material 'X'"):
        fg.load_model(json.dumps({"materials": {"X": {"foo": 1}}}))
    with pytest.raises(ValueError, match="nicht definiert"):
        fg.load_model(json.dumps({"materials": {"Luft": {"n": 1.0}}, "cover": "Luft", "substrate": "Glas"}))
    fgjson = {"dimension": "2D", "solver": "rcwa", "period_um": [0.5, 0.5], "materials": {"Luft": {"n": 1.0}, "Glas": {"n": 1.5}},
              "layers": [{"thickness_um": 0.15, "contour2d": {}}]}
    with pytest.raises(ValueError, match="FourierGrating"):
        fg.load_model(json.dumps(fgjson))
