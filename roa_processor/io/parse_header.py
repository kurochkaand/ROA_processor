from __future__ import annotations

import re

from roa_processor.models import HeaderMetadata


def _search_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _search_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def parse_header(header: str) -> HeaderMetadata:
    """
    Parse header such as:

    #1/cm sum, dif: SCP DCPI DCPII SCPc # Gain 9.4 e/ADc
    # Power at sample 321 mW # Cycles 20
    # Total times [s] 470.69 470.69 470.69 470.69
    """
    gain = _search_float(r"Gain\s+([0-9.+\-Ee]+)\s*e/ADc", header)
    power = _search_float(r"Power\s+at\s+sample\s+([0-9.+\-Ee]+)\s*mW", header)
    cycles = _search_int(r"Cycles\s+(\d+)", header)

    total_times: tuple[float, ...] = ()
    total_match = re.search(r"Total\s+times\s+\[s\]\s+(.+)$", header)
    if total_match:
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?", total_match.group(1))
        total_times = tuple(float(x) for x in numbers)

    return HeaderMetadata(
        gain_e_per_adc=gain,
        power_at_sample_mw=power,
        cumulative_cycles=cycles,
        total_times_s=total_times,
        raw_header=header.strip(),
    )
