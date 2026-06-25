from pathlib import Path

from roa_processor.io.parse_filename import parse_filename


def test_parse_filename():
    path = Path(
        "prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7-N20-500mW-146791s_0312026-06-21_A-000_out.txt"
    )
    meta = parse_filename(path)
    assert meta.sample_name == "prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7"
    assert meta.save_interval_cycles == 20
    assert meta.nominal_laser_power_mw == 500
    assert meta.camera_exposure_s == 1.46791
    assert meta.experiment_number == "031"
    assert meta.date == "2026-06-21"
    assert meta.camera == "A"
    assert meta.block_index == 0
