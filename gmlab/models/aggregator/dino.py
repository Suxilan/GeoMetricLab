"""Bridge DINO token outputs to aggregators that require the CLS token."""

from torch import nn

from ...components import build_component


class DinoAggregation(nn.Module):
    """Consume (CLS [N,C], patches [N,C,H,W], attention) without losing CLS.

    The configured aggregator must accept patches and cls_token and return
    (descriptor [N,D], auxiliary). Use patch_tokens for patch-only aggregators.
    This explicit bridge avoids guessing keyword support in model forward.
    """

    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = build_component(aggregator)
        self.out_channels = self.aggregator.out_channels

    def forward(self, features):
        if not isinstance(features, tuple) or len(features) != 3:
            raise ValueError("Expected DINO (CLS, patches, attention)")
        cls_token, patches, _ = features
        return self.aggregator(patches, cls_token=cls_token)
