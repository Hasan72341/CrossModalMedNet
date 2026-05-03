"""
pix2pix_standard/config.py
-------------------------
Project-wide settings for Brain, Pelvis, and Joint Pix2Pix models.
Hierarchical organization of outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_MODULE_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _MODULE_ROOT / ".env"
_DATA_ROOT = _MODULE_ROOT.parents[3] / "SynthRAD2023_Dataset"


class DataConfig(BaseSettings):
    dataset_dir: str = Field(default=str(_DATA_ROOT), validation_alias="DATASET_DIR")
    manifest_path: str = Field(default=str(_DATA_ROOT / "manifest.csv"), validation_alias="MANIFEST_PATH")
    
    # Regional scenarios
    train_regions: tuple[str, ...] = Field(default=("brain",), validation_alias="TRAIN_REGIONS")
    eval_regions: tuple[str, ...] = Field(default=("brain",), validation_alias="EVAL_REGIONS")
    
    image_size: int = Field(default=256, validation_alias="IMAGE_SIZE")
    num_workers: int = Field(default=4, validation_alias="NUM_WORKERS")
    
    slice_strategy: Literal["random", "center", "fixed"] = Field(
        default="random", validation_alias="SLICE_STRATEGY"
    )
    slice_axis: Literal["axial", "coronal", "sagittal"] = Field(
        default="axial", validation_alias="SLICE_AXIS"
    )

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        extra="ignore",
    )


class TrainingConfig(BaseSettings):
    batch_size: int = Field(default=1, validation_alias="BATCH_SIZE")
    test_batch_size: int = Field(default=1, validation_alias="TEST_BATCH_SIZE")
    num_epochs: int = Field(default=500, validation_alias="NUM_EPOCHS")
    n_epochs: int = Field(default=250, validation_alias="N_EPOCHS")
    n_epochs_decay: int = Field(default=250, validation_alias="N_EPOCHS_DECAY")
    
    lr_g: float = Field(default=2e-4, validation_alias="LR_G")
    lr_d: float = Field(default=2e-4, validation_alias="LR_D")
    lr_d_scale: float = Field(default=1.0, validation_alias="LR_D_SCALE")
    beta1: float = Field(default=0.5, validation_alias="BETA1")
    beta2: float = Field(default=0.999, validation_alias="BETA2")
    
    init_type: Literal["normal", "xavier", "kaiming", "orthogonal"] = Field(
        default="normal", validation_alias="INIT_TYPE"
    )
    init_gain: float = Field(default=0.02, validation_alias="INIT_GAIN")
    
    lambda_l1: float = Field(default=100.0, validation_alias="LAMBDA_L1")
    lambda_identity: float = Field(default=0.0, validation_alias="LAMBDA_ID")
    
    gan_mode: Literal["lsgan", "vanilla", "wgangp"] = Field(default="lsgan", validation_alias="GAN_MODE")
    
    use_wandb: bool = Field(default=True, validation_alias="USE_WANDB")
    wandb_entity: str | None = Field(default=None, validation_alias="WANDB_ENTITY")
    wandb_mode: Literal["online", "offline", "disabled"] = Field(
        default="online", validation_alias="WANDB_MODE"
    )
    
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        extra="ignore",
    )


class Settings(DataConfig, TrainingConfig):
    # Model Identification
    project_name: str = Field(default="pix2pix_brain", validation_alias="PROJECT_NAME")
    seed: int = Field(default=42, validation_alias="SEED")
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", validation_alias="DEVICE"
    )
    
    max_steps_per_epoch: int = Field(default=-1, validation_alias="MAX_STEPS_PER_EPOCH")
    eval_metrics_every_epochs: int = Field(default=5, validation_alias="EVAL_METRICS_EVERY")
    eval_visuals_every_epochs: int = Field(default=1, validation_alias="EVAL_VISUALS_EVERY")
    
    input_nc: int = Field(default=1, validation_alias="INPUT_NC")
    output_nc: int = Field(default=1, validation_alias="OUTPUT_NC")
    ngf: int = Field(default=64, validation_alias="NGF")
    ndf: int = Field(default=64, validation_alias="NDF")
    n_layers_d: int = Field(default=3, validation_alias="N_LAYERS_D")
    norm: Literal["instance", "batch", "none", "syncbatch"] = Field(default="batch", validation_alias="NORM")
    use_dropout: bool = Field(default=True, validation_alias="USE_DROPOUT")

    # Logging & Saving Settings
    use_tensorboard: bool = Field(default=True, validation_alias="USE_TENSORBOARD")
    use_torchmetrics: bool = Field(default=True, validation_alias="USE_TORCHMETRICS")
    
    log_every: int = Field(default=100, validation_alias="LOG_EVERY")
    image_log_every: int = Field(default=500, validation_alias="IMAGE_LOG_EVERY")
    save_every_epochs: int = Field(default=5, validation_alias="SAVE_EVERY_EPOCHS")
    
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        extra="ignore",
    )

    # Hierarchical Structure Properties
    @property
    def project_root(self) -> Path:
        return _MODULE_ROOT / self.project_name

    @property
    def log_dir(self) -> str:
        return str(self.project_root / "runs")

    @property
    def checkpoint_dir(self) -> str:
        return str(self.project_root / "checkpoints")
        
    @property
    def eval_dir(self) -> str:
        return str(self.project_root / "eval")
        
    @property
    def visuals_dir(self) -> str:
        return str(self.project_root / "visuals")

    @property
    def wandb_project_name(self) -> str:
        return self.project_name


def get_settings() -> Settings:
    return Settings()
