from pathlib import Path

from torch import nn

from psycop.common.sequence_models.embedders.BEHRT_embedders import BEHRTEmbedder
from psycop.common.sequence_models.registry import SequenceRegistry
from psycop.common.sequence_models.tasks import PretrainerBEHRT


@SequenceRegistry.tasks.register("model_from_checkpoint")
def load_model_from_checkpoint(checkpoint_path: Path) -> PretrainerBEHRT:
    return PretrainerBEHRT.load_from_checkpoint(checkpoint_path)


@SequenceRegistry.tasks.register("embedder_from_checkpoint")
def load_embedder_from_checkpoint(checkpoint_path: Path) -> BEHRTEmbedder:
    model = PretrainerBEHRT.load_from_checkpoint(checkpoint_path)
    return model.embedder


@SequenceRegistry.tasks.register("encoder_from_checkpoint")
def load_encoder_from_checkpoint(checkpoint_path: Path) -> nn.Module:
    model = PretrainerBEHRT.load_from_checkpoint(checkpoint_path)
    return model.encoder
