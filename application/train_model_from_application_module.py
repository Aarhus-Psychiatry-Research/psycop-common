"""Script using the train_model module to train a model.

Required to allow the trainer_spawner to point towards a python script
file, rather than an installed module.
"""
import hydra

from psycop_model_training.application_modules.train_model.main import train_model
from psycop_model_training.training.train_and_predict import CONFIG_PATH


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main():
    """Main."""
    train_model()


if __name__ == "__main__":
    main()
