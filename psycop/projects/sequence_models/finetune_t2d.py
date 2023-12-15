import logging
from pathlib import Path

from torch import nn

from psycop.common.sequence_models.embedders.BEHRT_embedders import BEHRTEmbedder
from psycop.common.sequence_models.registry import Registry
from psycop.common.sequence_models.tasks import BEHRTForMaskedLM
from psycop.common.sequence_models.train import train


@Registry.tasks.register("model_from_checkpoint")
def load_model_from_checkpoint(
    checkpoint_path: Path,
) -> BEHRTForMaskedLM:
    return BEHRTForMaskedLM.load_from_checkpoint(checkpoint_path)


@Registry.tasks.register("embedder_from_checkpoint")
def load_embedder_from_checkpoint(
    checkpoint_path: Path,
) -> BEHRTEmbedder:
    model = BEHRTForMaskedLM.load_from_checkpoint(checkpoint_path)
    return model.embedder


@Registry.tasks.register("encoder_from_checkpoint")
def load_encoder_from_checkpoint(
    checkpoint_path: Path,
) -> nn.Module:
    model = BEHRTForMaskedLM.load_from_checkpoint(checkpoint_path)
    return model.encoder


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S")
    config_path = Path(__file__).parent / "finetune_behrt_t2d_with_pretrain.cfg"
    train(config_path)
