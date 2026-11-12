import json

import pytest
from sklearn.metrics import average_precision_score

from model.evaluate.decode import Thresholds
from model.evaluate.metrics import evaluate
from model.labels import FINE_LABELS
from test_decode import record, logits


def test_end_to_end_includes_coarse_errors():
    records = [record(['Modify'], [FINE_LABELS[0]]), record()]
    report = evaluate(records, logits([[.1,.1,.1],[.1,.9,.1]], [[.9]+[.1]*4,[.8]+[.1]*4]), Thresholds(), language='csharp')
    end = report.metrics['fine_end_to_end']['per_label'][FINE_LABELS[0]]
    conditional = report.metrics['fine_conditional']['per_label'][FINE_LABELS[0]]
    assert (end['tp'], end['fp'], end['fn'], end['f1']) == (0,1,1,0)
    assert conditional['f1'] == 1
    assert end['pr_auc'] == average_precision_score([1,0], [.09,.72])
    assert report.metrics['fine_end_to_end_macro_f1'] == 0
    assert report.metrics['fine_conditional']['n_samples'] == 1
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize('language', ['csharp', 'java', 'cpp'])
def test_all_languages_use_the_same_five_label_universe(language):
    report = evaluate([record(['Modify'], ['catch_added'])],
                      logits([[.1,.9,.1]], [[.1,.9,.1,.1,.1]]), Thresholds(), language=language)
    assert report.provenance['fine_label_version'] == 'eh-change-5'
    assert tuple(report.metrics['fine_end_to_end']['per_label']) == FINE_LABELS
    assert report.predictions[0].fine_labels == ('catch_added',)
    assert report.diagnostics == {}
    assert report.metrics['fine_common_end_to_end'] == report.metrics['fine_end_to_end']
    assert report.metrics['fine_common_conditional'] == report.metrics['fine_conditional']


def test_finally_only_gold_remains_in_conditional_modify_denominator():
    report = evaluate([record(['Modify'], ['finally_removed'])],
                      logits([[.1,.9,.1]], [[.1]*5]), Thresholds(), language='csharp')
    assert report.counts['n_gold_modify'] == 1
    assert report.metrics['fine_conditional']['n_samples'] == 1
    assert report.metrics['fine_end_to_end']['support'] == 0
    assert report.metrics['subset_match'] == 1


def test_no_modify_and_fixed_denominators():
    report = evaluate([record(['Add'])], logits([[.9,.1,.1]], [[.1]*5]), Thresholds(), language='java')
    assert report.metrics['coarse_macro_f1'] == pytest.approx(1/3)
    assert report.metrics['coarse']['micro_f1'] == 1
    assert report.metrics['fine_conditional']['n_samples'] == 0
    assert report.metrics['fine_conditional']['macro_f1'] == 0
    assert report.metrics['fine_end_to_end']['per_label'][FINE_LABELS[0]]['pr_auc'] is None
    assert report.metrics['none']['f1'] == 0


def test_empty_evaluation_has_finite_zero_metrics():
    import torch
    from model.models.base import PairLogits

    report = evaluate([], PairLogits(torch.empty((0,3)), torch.empty((0,5))), Thresholds(), language='cpp')
    assert report.metrics['coarse_macro_f1'] == 0
    assert report.metrics['subset_match'] == 0
    json.dumps(report.to_dict(), allow_nan=False)


def test_raw_coarse_and_conditional_rankings_and_common_denominator():
    records = [record(['Modify'], [FINE_LABELS[0]]), record(['Modify'])]
    values = logits([[.1,.2,.1],[.1,.9,.1]], [[.9]+[.1]*4,[.8]+[.1]*4])
    report = evaluate(records, values, Thresholds(), language='java')
    assert report.metrics['fine_conditional']['per_label'][FINE_LABELS[0]]['pr_auc'] == 1
    assert report.metrics['fine_common_conditional']['macro_f1'] == pytest.approx((2/3)/5)
    assert report.metrics['fine_conditional']['macro_f1'] == pytest.approx((2/3)/5)
