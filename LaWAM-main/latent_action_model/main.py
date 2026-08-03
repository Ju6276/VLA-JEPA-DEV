from lightning.pytorch.cli import LightningCLI

from latent_action_model.core.lam_lightinng import VJEPA_LAM
from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule


class _LAMLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.image_hw", "data.image_hw", apply_on="parse")


cli = _LAMLightningCLI(
    VJEPA_LAM,
    LeRobotDataModule,
    seed_everything_default=42,
    save_config_kwargs={"overwrite": True},
)
