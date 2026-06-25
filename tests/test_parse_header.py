from roa_processor.io.parse_header import parse_header


def test_parse_header():
    header = (
        "#1/cm sum, dif: SCP DCPI DCPII SCPc "
        "# Gain 9.4 e/ADc # Power at sample 321 mW "
        "# Cycles 20 # Total times [s] 470.69 470.69 470.69 470.69"
    )
    meta = parse_header(header)
    assert meta.gain_e_per_adc == 9.4
    assert meta.power_at_sample_mw == 321
    assert meta.cumulative_cycles == 20
    assert meta.total_time_scp_s == 470.69
