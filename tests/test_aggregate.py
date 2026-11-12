"""Real aggregation must isolate pairs and ignore every padded slot."""

import pytest
import torch

from model.config import ModelConfig
from model.models.aggregate import WindowAggregator, WindowFusion, pad_windows


def small_config(**kwargs):
    return ModelConfig(native_dim=8, projection_dim=16, layers=2, heads=4,
                       ffn_dim=32, dropout=0.1, **kwargs)


def test_padding_and_other_pair_do_not_affect_prediction():
    torch.manual_seed(12)
    aggregate = WindowAggregator(small_config()).eval()
    first = torch.randn(1, 2, 16)
    second = torch.randn(1, 5, 16) * 20
    alone = aggregate(first, torch.ones(1, 2, dtype=torch.bool))
    padded = torch.cat((first, torch.randn(1, 3, 16) * 1000), dim=1)
    batch = torch.cat((padded, second))
    valid = torch.tensor([[True, True, False, False, False], [True] * 5])
    output = aggregate(batch, valid)
    torch.testing.assert_close(output[:1], alone, atol=2e-6, rtol=2e-6)
    changed = batch.clone()
    changed[0, 2:] = -98765
    changed[1] *= -15
    torch.testing.assert_close(aggregate(changed, valid)[:1], alone, atol=2e-6, rtol=2e-6)
    changed[0, 2:] = torch.nan
    torch.testing.assert_close(aggregate(changed, valid)[:1], alone, atol=2e-6, rtol=2e-6)


def test_roles_and_positions_reach_aggregation():
    fusion = WindowFusion(small_config())
    with torch.no_grad():
        fusion.projection.weight.zero_()
        fusion.projection.bias.zero_()
        fusion.role_embedding.weight.zero_()
        fusion.role_embedding.weight[:, 0] = torch.tensor([1., 2., 3.])
        fusion.position_projection.weight.zero_()
        fusion.position_projection.bias.zero_()
        fusion.position_projection.weight[1] = torch.tensor([1., 2., 4., 8.])
    summaries = torch.zeros(3, 8)
    roles = torch.tensor([0, 1, 2])
    positions = torch.tensor([[.1, .2, .3, .4]] * 3)
    features = fusion(summaries, roles, positions)
    torch.testing.assert_close(features[:, 0], torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(features[:, 1], torch.tensor([4.9] * 3))
    swapped = fusion(summaries, roles.flip(0), positions.flip(1))
    torch.testing.assert_close(swapped[:, 0], torch.tensor([3., 2., 1.]))
    torch.testing.assert_close(swapped[:, 1], torch.tensor([2.6] * 3))
    mean = WindowAggregator(small_config(aggregation="mean"))
    # One-window pairs make role/position differences survive mean pooling.
    pooled = mean(features[:, None], torch.ones(3, 1, dtype=torch.bool))
    torch.testing.assert_close(pooled[:, 0], features)
    assert not torch.equal(pooled[0], pooled[1])


def test_mean_ablation_masks_padding():
    aggregate = WindowAggregator(small_config(aggregation="mean"))
    windows = torch.zeros(2, 3, 16, requires_grad=True)
    with torch.no_grad():
        windows[0, 0] = 2
        windows[0, 1] = 4
        windows[0, 2] = 999
        windows[1, 0] = 7
        windows[1, 1:] = -999
    valid = torch.tensor([[True, True, False], [True, False, False]])
    result = aggregate(windows, valid)
    assert result.shape == (2, 8, 16)
    torch.testing.assert_close(result[0], torch.full((8, 16), 3.))
    torch.testing.assert_close(result[1], torch.full((8, 16), 7.))
    poisoned = windows.detach().clone().masked_fill(~valid[:, :, None], torch.nan)
    torch.testing.assert_close(aggregate(poisoned, valid), result)
    result.sum().backward()
    assert torch.count_nonzero(windows.grad[~valid]) == 0
    assert torch.all(windows.grad[valid] != 0)
    assert sum(p.numel() for p in aggregate.parameters()) == 0
    assert sum(p.numel() for p in WindowAggregator(small_config()).parameters()) > 0


def test_interleaved_pair_indices_preserve_all_windows_and_gradients():
    flat = torch.tensor([[10.], [20.], [30.], [40.]], requires_grad=True)
    padded, valid = pad_windows(flat, torch.tensor([1, 0, 1, 1]), 2)
    torch.testing.assert_close(padded, torch.tensor([[[20.], [0.], [0.]], [[10.], [30.], [40.]]]))
    assert valid.tolist() == [[True, False, False], [True, True, True]]
    padded[valid].sum().backward()
    torch.testing.assert_close(flat.grad, torch.ones(4, 1))


@pytest.mark.parametrize("aggregation", ["transformer", "mean"])
def test_empty_pair_rejected_before_attention(aggregation):
    with pytest.raises(ValueError, match="valid window"):
        WindowAggregator(small_config(aggregation=aggregation))(
            torch.zeros(2, 3, 16), torch.tensor([[True] * 3, [False] * 3]))
    with pytest.raises(ValueError, match="window"):
        pad_windows(torch.zeros(1, 16), torch.tensor([0]), 2)


def test_main_fusion_rejects_concat_pair_role():
    with pytest.raises(ValueError, match="PRE.*POST.*DIFF"):
        WindowFusion(small_config())(torch.zeros(1, 8), torch.tensor([3]), torch.zeros(1, 4))
