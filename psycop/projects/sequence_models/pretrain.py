import logging
import sys
from pathlib import Path

project_path = Path(__file__).parents[3]
print(project_path)
sys.path.append(str(project_path))

from psycop.common.sequence_models.train import train

if __name__ == "__main__":
    config_path = Path(__file__).parent / "pretrain_behrt.cfg"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )
    logging.info("Starting Training")
    train(config_path)
