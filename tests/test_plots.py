import matplotlib
import numpy as np

matplotlib.use("Agg")

from roa_processor.models import IsolatedExperiment
from roa_processor.plotting import plots


def test_isolated_raman_plot_highlights_first_and_last_blocks(monkeypatch):
    isolated = IsolatedExperiment(
        wavenumber=np.array([100.0, 200.0, 300.0]),
        raman_raw=np.array(
            [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [1.2, 2.2, 3.2],
            ]
        ),
        roa_raw=np.zeros((3, 3)),
        raman_norm=np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
            ]
        ),
        roa_norm=np.zeros((3, 3)),
        delta_times_s=np.array([1.0, 1.0, 1.0]),
        delta_cycles=np.array([1.0, 1.0, 1.0]),
        power_at_sample_mw=np.array([2.0, 2.0, 2.0]),
        block_indices=np.array([0, 1, 2]),
    )
    monkeypatch.setattr(plots, "_save_or_show", lambda path: None)

    plots.plot_isolated_raman_blocks(isolated)

    lines = plots.plt.gca().lines
    assert [line.get_linewidth() for line in lines] == [0.7, 2.0, 2.0]
    assert lines[1].get_label() == "First block 000"
    assert lines[2].get_label() == "Last block 002"
    plots.plt.close()


def test_isolated_roa_spike_removal_comparison_stacks_representative_pairs(monkeypatch):
    n_blocks = 12
    wavenumber = np.array([100.0, 200.0, 300.0])
    roa_before = np.vstack(
        [np.array([0.0, 1.0, 0.0]) + block_number * 0.01 for block_number in range(n_blocks)]
    )
    roa_after = roa_before.copy()
    roa_after[:, 1] -= 0.5
    block_indices = np.arange(100, 100 + n_blocks)
    monkeypatch.setattr(plots, "_save_or_show", lambda path: None)

    plots.plot_isolated_roa_spike_removal_comparison(
        wavenumber,
        roa_before,
        roa_after,
        block_indices,
    )

    ax = plots.plt.gca()
    lines = ax.lines
    assert len(lines) == 20
    assert lines[0].get_label() == "Before spike removal"
    assert lines[1].get_label() == "After spike removal"
    np.testing.assert_allclose(
        lines[1].get_ydata() - lines[0].get_ydata(),
        np.array([0.0, -0.5, 0.0]),
    )
    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "Block 100",
        "Block 101",
        "Block 102",
        "Block 103",
        "Block 104",
        "Block 106",
        "Block 107",
        "Block 108",
        "Block 109",
        "Block 111",
    ]
    plots.plt.close()
