import logging
from pathlib import Path

from psycop.common.sequence_models.train import train

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S")
    config_path = Path(__file__).parent / "t2d_behrt_finetune_no_pretrain.cfg"
    train(config_path)
