import json

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
