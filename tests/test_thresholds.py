import pytest

from model.evaluate.thresholds import calibrate_thresholds
from model.labels import FINE_LABELS
from test_decode import record, logits


def test_thresholds_use_predicted_modify_gate():
    records = [record(['Modify'], [FINE_LABELS[0]]), record(), record()]
    values = logits([[.1,.4,.1],[.1,.8,.1],[.1,.1,.1]], [[.4]+[.1]*4,[.3]+[.1]*4,[.9]+[.1]*4])
    thresholds = calibrate_thresholds(records, values, split='val', language='csharp')
    assert thresholds.coarse[1] == .4
    assert thresholds.fine[0] == .4


def test_threshold_ties_and_zero_support():
    values = logits([[.4,.1,.1]], [[.9]*5])
    thresholds = calibrate_thresholds([record(['Add'])], values, split='val', language='csharp')
    assert thresholds.coarse == (.4,.5,.5)
    assert thresholds.fine == (.5,)*5
    # One-sided Modify scores must never create a gate or a coarse false positive.
    thresholds = calibrate_thresholds([record(['Add'], pre=None)], logits([[.9]*3], [[.9]*5]), split='val', language='csharp')
    assert thresholds.coarse == (.5,)*3


def test_symmetric_tie_chooses_smaller_threshold():
    records = [record(['Add']), record(), record(), record(['Add'])]
    values = logits([[p,.1,.1] for p in [.6,.5,.48,.46]], [[.1]*5]*4)
    assert calibrate_thresholds(records, values, split='val', language='csharp').coarse[0] == .45


def test_fine_calibration_counts_missed_modify():
    records = [record(['Modify'], [FINE_LABELS[0]])]
    # No candidate can recover Modify: every fine candidate has end-to-end F1 0.
    values = logits([[.1,.01,.1]], [[.9]*5])
    result = calibrate_thresholds(records, values, split='val', language='csharp')
    assert result.coarse[1] == .5
    assert result.fine[0] == .5


@pytest.mark.parametrize('split,language', [('test','csharp'), ('val','java'), ('val','cpp')])
def test_cannot_calibrate_test_or_transfer(split, language):
    with pytest.raises(ValueError):
        calibrate_thresholds([record()], logits([[.1]*3], [[.1]*5]), split=split, language=language)
