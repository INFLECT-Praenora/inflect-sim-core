import pytest

from viva_simulation_core.analysis.milestones import extract_milestones

STEP = 600.0  # 10 min


def _fixture():
    """13 samples, 10 min apart (0..2 h). SBP dips <100 at 0.5 h (sustained),
    with a one-sample blip below at 0.1 h that must be ignored. MAP stays >65.
    HR crosses >90 at 1 h. SVR halves. Pathogen peaks at 1.5 h then falls to ~0."""
    sbp = [120, 99, 118, 119, 118, 99, 95, 92, 90, 90, 91, 92, 93]
    hr = [72, 74, 76, 78, 80, 85, 91, 95, 100, 101, 99, 98, 97]
    map_ = [90, 85, 88, 89, 88, 84, 80, 78, 76, 75, 76, 77, 78]
    svr = [1000, 990, 950, 900, 850, 800, 700, 600, 520, 500, 520, 540, 560]
    tis = [1.0, 1.0, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.78, 0.75, 0.76, 0.77, 0.78]
    path = [1, 2, 4, 8, 16, 32, 64, 100, 60, 20, 5, 0.1, 0.0]
    return [{'time': i * STEP, 'SBP': sbp[i], 'MAP': map_[i], 'HR': hr[i], 'SVR': svr[i],
             'tissue_integrity': tis[i], 'blood_pathogen': path[i]} for i in range(13)]


def test_milestone_extraction():
    m = extract_milestones(_fixture(), sustain_minutes=20.0)
    assert m.sbp_below_100_h == pytest.approx(50 / 60)   # onset of sustained run at idx 5
    assert m.map_below_65_h is None
    assert m.hr_above_90_h == pytest.approx(1.0)          # idx 6
    assert m.svr_ratio_min == pytest.approx(0.5)
    assert m.svr_ratio_min_h == pytest.approx(1.5)  # idx 9
    assert m.tissue_integrity_min == pytest.approx(0.75)
    assert m.tissue_integrity_min_h == pytest.approx(1.5)
    assert m.peak_blood_pathogen == pytest.approx(100)
    assert m.peak_blood_pathogen_h == pytest.approx(7 * STEP / 3600)
    assert m.resolved is True


def test_sustain_window_and_unresolved():
    # with a 5 min window the blip at idx 1 counts (single sample, zero span)
    m = extract_milestones(_fixture(), sustain_minutes=0.0)
    assert m.sbp_below_100_h == pytest.approx(STEP / 3600)
    # a run that never lasts long enough -> None
    assert extract_milestones(_fixture(), sustain_minutes=200.0).sbp_below_100_h is None
    # rising pathogen at end -> not resolved
    s = _fixture()
    for r in s[-3:]:
        r['blood_pathogen'] = 500.0
    assert extract_milestones(s, sustain_minutes=20.0).resolved is False


def test_missing_observables_are_none():
    m = extract_milestones([{'time': 0.0, 'SBP': 120}, {'time': 60.0, 'SBP': 118}])
    assert m.sbp_below_100_h is None and m.svr_ratio_min is None
    assert m.peak_blood_pathogen is None and m.resolved is None
