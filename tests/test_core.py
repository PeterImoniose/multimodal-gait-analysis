"""Unit tests on synthetic signals with known answers."""

import numpy as np
import pandas as pd
import pytest

from mgait import config as C, eeg, emg, gait, stats


def synthetic_events(n=40, stride=1.2, stance=0.62, jitter=0.03, seed=0):
    """Alternating left/right contacts with variable stride times."""
    rng = np.random.default_rng(seed)
    st = stride + rng.normal(0, jitter, n)
    hs_l = 1.0 + np.concatenate([[0], np.cumsum(st[:-1])])
    hs_r = hs_l + st / 2
    ev = gait.Events(hs={"L": hs_l, "R": hs_r}, to={"L": hs_l + stance * st, "R": hs_r + stance * st})
    return ev, float(hs_l[-1] + 2 * stride)


def test_contacts_from_mask_closes_gaps_and_drops_short_runs():
    fs = 100.0
    m = np.zeros(500, bool)
    m[50:150] = True
    m[152:200] = True        # 20 ms gap: should merge with the first run
    m[300:305] = True        # 50 ms run: too short
    on, off = gait.contacts_from_mask(m, fs, min_on=0.2, min_off=0.05)
    assert list(on) == [50] and list(off) == [200]


def test_stride_table_recovers_temporal_parameters():
    ev, _ = synthetic_events(jitter=0.0)
    st = gait.stride_table(ev)
    assert len(st) > 70
    assert st["stride_time"].median() == pytest.approx(1.2, abs=1e-9)
    assert st["stance_pct"].median() == pytest.approx(62.0, abs=1e-6)
    # stance 62 % with feet half a stride apart gives two double-support phases of 12 % each
    assert st["double_support_pct"].median() == pytest.approx(24.0, abs=1e-6)
    assert st["qc_ok"].all()


def test_estimate_lag_recovers_known_offset_and_side_swap():
    ev, dur = synthetic_events()
    shifted = gait.Events(hs={k: v + 0.13 for k, v in ev.hs.items()}, to={k: v + 0.13 for k, v in ev.to.items()})
    r = gait.estimate_lag(ev, shifted, dur)
    assert r["lag_s"] == pytest.approx(0.13, abs=0.011)
    swapped = gait.Events(hs={"L": ev.hs["R"], "R": ev.hs["L"]}, to={"L": ev.to["R"], "R": ev.to["L"]})
    a = gait.estimate_lag(ev, swapped, dur)
    b = gait.estimate_lag(ev, swapped, dur, swap_other=True)
    assert b["peak_r"] > a["peak_r"] and abs(b["lag_s"]) < 0.011


def test_match_events_reports_bias():
    ev, _ = synthetic_events()
    late = gait.apply_offsets(ev, hs_offset=-0.02, to_offset=0.01)   # HS 20 ms late, TO 10 ms early
    m = gait.match_events(ev, late)
    assert m.loc[m.kind == "hs", "err"].mean() == pytest.approx(0.02, abs=1e-9)
    assert m.loc[m.kind == "to", "err"].mean() == pytest.approx(-0.01, abs=1e-9)


def test_plate_baseline_ignores_stance_loading():
    rng = np.random.default_rng(1)
    z = np.where(np.arange(10000) % 1000 < 620, 700.0, 0.0) - 80 + rng.normal(0, 2, 10000)
    assert gait._plate_baseline(z) == pytest.approx(-80, abs=1.0)


def test_resample_curve_endpoints():
    x = np.linspace(0, 10, 37)
    r = gait.resample_curve(x)
    assert len(r) == gait.N_POINTS and r[0] == 0 and r[-1] == 10


def test_holm_matches_hand_computation():
    adj = stats.holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_icc_and_bland_altman():
    rng = np.random.default_rng(2)
    a = rng.normal(10, 2, 200)
    assert stats.icc_a1(a, a) == pytest.approx(1.0)
    ba = stats.bland_altman(a, a + 0.5)
    assert ba["bias"] == pytest.approx(0.5) and ba["sd"] == pytest.approx(0.0, abs=1e-12)
    assert stats.icc_a1(a, a + 3.0) < stats.icc_a1(a, a + 0.1)


def test_paired_cluster_test_finds_injected_difference_only():
    rng = np.random.default_rng(3)
    A = rng.normal(0, 1, (30, 101))
    B = A + rng.normal(0, 0.3, (30, 101))
    B[:, 40:60] += 1.0
    r = stats.paired_cluster_test(B, A, n_perm=500)
    sig = [c for c in r["clusters"] if c["p"] < 0.05]
    assert len(sig) == 1 and sig[0]["start"] >= 35 and sig[0]["end"] <= 65


def test_mixed_model_recovers_slope_with_subject_offsets():
    rng = np.random.default_rng(4)
    rows = []
    for s in range(40):
        off = rng.normal(0, 5)
        for v in (0.5, 0.75, 1.0):
            rows.append({"subject": f"S{s}", "speed_ms": v, "y": 10 + off + 8 * v + rng.normal(0, 0.3)})
    d = pd.DataFrame(rows)
    d.loc[0, "y"] = np.nan   # incomplete rows are dropped, not fatal
    r = stats.mixed_model(d, "y ~ speed_ms")
    assert r.params["speed_ms"] == pytest.approx(8, abs=0.5)


def test_synergies_recover_known_structure():
    rng = np.random.default_rng(5)
    W = np.array([[1, 0], [0.8, 0], [0.5, 0.2], [0, 1], [0.1, 0.9], [0, 0.6]])
    phase = np.tile(np.arange(101), 20)
    burst = lambda c: np.exp(-0.5 * ((phase - c) / 8.0) ** 2)
    H = np.vstack([burst(15), burst(55)]) * (1 + 0.1 * rng.normal(size=(2, phase.size)) ** 2)
    X = emg.prepare_matrix(W @ H + 0.01)
    fit = emg.synergies(X, k_max=3)
    assert fit["vaf"][2] > 0.99 and fit["vaf"][1] < fit["vaf"][2]
    assert emg.n_synergies(fit) == 2


def test_harmonic_ratio_detects_step_locked_peaks():
    fs, n, f_step = C.FS_EEG, int(60 * C.FS_EEG), 1.7
    rng = np.random.default_rng(6)
    t = np.arange(n) / fs
    noise = rng.normal(0, 1, (n, 1))
    art = noise + sum(0.8 * np.sin(2 * np.pi * k * f_step * t)[:, None] for k in range(1, 5))
    f1, p1 = eeg.psd(noise)
    f2, p2 = eeg.psd(art)
    assert eeg.harmonic_ratio(f1, p1, f_step)[0] == pytest.approx(1.0, abs=0.35)
    assert eeg.harmonic_ratio(f2, p2, f_step)[0] > 5


def test_symmetry_index():
    from mgait.features import symmetry_index
    assert symmetry_index(1.0, 1.0) == 0
    assert symmetry_index(1.1, 0.9) == pytest.approx(20.0)


@pytest.mark.skipif(not C.SUBJECT_FILE.exists(), reason="raw dataset not available")
def test_subject_table_drops_summary_rows_and_identifiers():
    from mgait import io
    m = io.load_subjects()
    assert len(m) == 59 and "dob" not in m.columns
    assert m.index.str.fullmatch(r"S\d+").all()
