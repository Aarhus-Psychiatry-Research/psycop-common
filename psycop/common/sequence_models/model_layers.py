from torch import nn

from .registry import SequenceRegistry


@SequenceRegistry.layers.register("transformer_encoder_layer")
def create_encoder_layer(
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    layer_norm_eps: float = 1e-12,
    norm_first: bool = True,
) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
        norm_first=norm_first,
    )


@SequenceRegistry.layers.register("transformer_encoder")
def create_transformers_encoder(num_layers: int, encoder_layer: nn.Module) -> nn.Module:
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # type: ignore
