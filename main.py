# from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from training_utils.trainer import Trainer

def cli_main():
    LightningCLI(
        trainer_class=Trainer,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main()
