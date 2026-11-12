import pytest
import torch

from model.data.schema import PairRecord
from model.labels import COARSE_LABELS, FINE_LABELS, encode_labels
from model.models.base import PairLogits
from model.evaluate.decode import Thresholds, decode


def record(labels=(), fine=(), pre='before', post='after'):
    return PairRecord('id', pre, post, encode_labels(labels, fine, has_pre=bool(pre and pre.strip()), has_post=bool(post and post.strip())), None, None, None, 'csharp')


def logits(coarse, fine):
    return PairLogits(torch.logit(torch.tensor(coarse, dtype=torch.float64)), torch.logit(torch.tensor(fine, dtype=torch.float64)))


def test_none_mixed_labels_and_empty_fine_modify():
    out = decode(logits([[.1,.1,.1],[.9,.9,.9],[.1,.9,.1]], [[.9]*5,[.9]*5,[.1]*5]), [record()]*3, Thresholds())
    assert out[0].coarse_labels == ('None',) and out[0].fine_labels == ()
    assert out[1].coarse_labels == COARSE_LABELS and out[1].fine_labels == FINE_LABELS
    assert out[2].coarse_labels == ('Modify',) and out[2].fine_labels == ()
    assert out[0].fine_scores == pytest.approx([.9]*5)


def test_one_sided_allowed_labels():
    out = decode(logits([[.9]*3]*3, [[.9]*5]*3), [record(pre='  '), record(post=None), record()], Thresholds())
    assert out[0].coarse_labels == ('Add',) and out[0].fine_labels == ()
    assert out[1].coarse_labels == ('Delete',) and out[1].fine_labels == ()
    assert out[2].coarse_labels == COARSE_LABELS


def test_validation_and_threshold_serialization():
    assert Thresholds.from_dict(Thresholds().to_dict()) == Thresholds()
    with pytest.raises(ValueError):
        Thresholds(coarse=(.5,))
    with pytest.raises(ValueError):
        decode(logits([[.5]*3], [[.5]*5]), [], Thresholds())


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_low_precision_logits(dtype):
    from model.evaluate.metrics import evaluate
    from model.evaluate.thresholds import calibrate_thresholds

    values = PairLogits(torch.tensor([[-2.,2.,-2.]], dtype=dtype), torch.tensor([[2.]+[-2.]*4], dtype=dtype))
    records = [record(['Modify'], [FINE_LABELS[0]])]
    assert decode(values, records, Thresholds())[0].fine_labels == (FINE_LABELS[0],)
    assert evaluate(records, values, Thresholds(), language='csharp').metrics['fine_end_to_end']['micro_f1'] == 1
    assert calibrate_thresholds(records, values, split='val', language='csharp').fine[0] == .5
