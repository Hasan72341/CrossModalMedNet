"""
cyclegan_pelvis/config.py
-------------------------
Project-wide settings loaded from environment variables / .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_MODULE_ROOT = Path(__file__).resolve().parent
_ENV_FILE = Path("/usershome/cs671_user4/cyclegan_models/src/attention_off/.env")


class DataConfig(BaseSettings):
    ct_train_dir: str = Field(default="pelvis", validation_alias="CT_TRAIN_DIR")
    mri_train_dir: str = Field(default="pelvis", validation_alias="MRI_TRAIN_DIR")
    ct_val_dir: str = Field(default="pelvis", validation_alias="CT_VAL_DIR")
    mri_val_dir: str = Field(default="pelvis", validation_alias="MRI_VAL_DIR")
    
    image_size: int = Field(default=256, validation_alias="IMAGE_SIZE")
    num_workers: int = Field(default=4, validation_alias="NUM_WORKERS")
    
    data_mode: Literal["2d", "2.5d", "3d"] = Field(
        default="2d", validation_alias="DATA_MODE"
    )
    @property
    def use_25d(self) -> bool:
        return self.data_mode == "2.5d"
    
    @property
    def use_3d(self) -> bool:
        return self.data_mode == "3d"

    num_adjacent_slices: int = Field(default=3, validation_alias="NUM_SLICES")
    volume_depth: int = Field(default=64, validation_alias="VOLUME_DEPTH")
    slice_axis: Literal["axial", "coronal", "sagittal"] = Field(
        default="axial", validation_alias="SLICE_AXIS"
    )

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        extra="ignore",
    )


class TrainingConfig(BaseSettings):
    batch_size: int = Field(default=1, validation_alias="BATCH_SIZE")
    num_epochs: int = Field(default=200, validation_alias="NUM_EPOCHS")
    lr_g: float = Field(default=2e-4, validation_alias="LR_G")
    lr_d: float = Field(default=2e-4, validation_alias="LR_D")
    init_type: Literal["normal", "xavier", "kaiming", "orthogonal"] = Field(
        default="normal", validation_alias="INIT_TYPE"
    )
    init_gain: float = Field(default=0.02, validation_alias="INIT_GAIN")
    lambda_lpips: float = Field(default=1.0, validation_alias="LAMBDA_LPIPS")
    use_wandb: bool = Field(default=True, validation_alias="USE_WANDB")
    wandb_project_name: str = Field(default="cyclegan", validation_alias="WANDB_PROJECT")
    wandb_entity: str | None = Field(default=None, validation_alias="WANDB_ENTITY")
    wandb_mode: Literal["online", "offline", "disabled"] = Field(
        default="online", validation_alias="WANDB_MODE"
    )
    
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        extra="ignore",
    )


class Settings(DataConfig, TrainingConfig):
    # ── Data ──────────────────────────────────────────────────────────────────
    dataset_dir: str = Field(default="/usershome/cs671_user4/SynthRAD2023_Dataset", validation_alias="DATASET_DIR")
    ct_train_dir: str = Field(default="pelvis", validation_alias="CT_TRAIN_DIR")
    mri_train_dir: str = Field(default="pelvis", validation_alias="MRI_TRAIN_DIR")
    ct_val_dir: str = Field(default="pelvis", validation_alias="CT_VAL_DIR")
    mri_val_dir: str = Field(default="pelvis", validation_alias="MRI_VAL_DIR")

    image_size: int = Field(default=256, validation_alias="IMAGE_SIZE")
    batch_size: int = Field(default=1, validation_alias="BATCH_SIZE")
    test_batch_size: int = Field(default=1, validation_alias="TEST_BATCH_SIZE")
    shuffle: bool = Field(default=True, validation_alias="SHUFFLE")
    num_workers: int = Field(default=0, validation_alias="NUM_WORKERS")
    pin_memory: bool = Field(default=True, validation_alias="PIN_MEMORY")
    ct_shift: float = Field(default=0.0, validation_alias="CT_SHIFT")
    ct_scale: float = Field(default=1.0, validation_alias="CT_SCALE")
    mri_scale: float = Field(default=1.0, validation_alias="MRI_SCALE")
    data_mode: Literal["2d", "2.5d", "3d"] = Field(
        default="2d", validation_alias="DATA_MODE"
    )
    @property
    def use_25d(self) -> bool:
        return self.data_mode == "2.5d"
    
    @property
    def use_3d(self) -> bool:
        return self.data_mode == "3d"

    num_adjacent_slices: int = Field(default=3, validation_alias="NUM_SLICES")
    volume_depth: int = Field(default=64, validation_alias="VOLUME_DEPTH")
    slice_axis: Literal["axial", "coronal", "sagittal"] = Field(
        default="axial", validation_alias="SLICE_AXIS"
    )
    num_slices: int = Field(default=3, validation_alias="NUM_SLICES")
    patch_size: str | None = Field(default=None, validation_alias="PATCH_SIZE")
    slice_strategy: Literal["random", "center", "fixed"] = Field(
        default="random", validation_alias="SLICE_STRATEGY"
    )
    fixed_slice_index: int | None = Field(
        default=None, validation_alias="FIXED_SLICE_INDEX"
    )

    # ── Region Settings ───────────────────────────────────────────────────────
    early_training_regions: tuple[str, ...] = Field(default=("brain",), validation_alias="EARLY_REGIONS")
    late_training_regions: tuple[str, ...] = Field(default=("brain",), validation_alias="LATE_REGIONS")
    region_switch_epoch: int = Field(default=999999, validation_alias="REGION_SWITCH_EPOCH")

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs: int = Field(default=500, validation_alias="NUM_EPOCHS")
    n_epochs: int = Field(default=250, validation_alias="N_EPOCHS")
    n_epochs_decay: int = Field(default=250, validation_alias="N_EPOCHS_DECAY")
    # Use -1 to disable the per-epoch cap and iterate the full DataLoader.
    max_steps_per_epoch: int = Field(default=-1, validation_alias="MAX_STEPS_PER_EPOCH")
    eval_metrics_every_epochs: int = Field(default=5, validation_alias="EVAL_METRICS_EVERY")
    eval_visuals_every_epochs: int = Field(default=1, validation_alias="EVAL_VISUALS_EVERY")

    gan_mode: Literal["lsgan", "vanilla", "wgangp"] = Field(default="lsgan", validation_alias="GAN_MODE")
    pool_size: int = Field(default=50, validation_alias="POOL_SIZE")

    lr_g: float = Field(default=2e-4, validation_alias="LR_G")
    lr_d: float = Field(default=2e-4, validation_alias="LR_D")
    lr_d_scale: float = Field(default=1.0, validation_alias="LR_D_SCALE")
    d_update_every: int = Field(default=1, validation_alias="D_UPDATE_EVERY")
    beta1: float = Field(default=0.5, validation_alias="BETA1")
    beta2: float = Field(default=0.999, validation_alias="BETA2")
    init_type: Literal["normal", "xavier", "kaiming", "orthogonal"] = Field(
        default="normal", validation_alias="INIT_TYPE"
    )
    init_gain: float = Field(default=0.02, validation_alias="INIT_GAIN")

    lambda_cycle: float = Field(default=10.0, validation_alias="LAMBDA_CYCLE")
    lambda_identity: float = Field(default=0.5, validation_alias="LAMBDA_ID")
    lambda_feature: float = Field(default=0.0, validation_alias="LAMBDA_FM")
    lambda_volume: float = Field(default=0.0, validation_alias="LAMBDA_VOL")
    lambda_edge: float = Field(default=0.0, validation_alias="LAMBDA_EDGE")
    lambda_lpips: float = Field(default=1.0, validation_alias="LAMBDA_LPIPS")
    model_variant: str = Field(default="standard", validation_alias="MODEL_VARIANT")
    use_attention: bool = Field(default=False, validation_alias="USE_ATTENTION")
    use_transformer_attention: bool = Field(default=False, validation_alias="USE_TRANSFORMER_ATTENTION")
    use_dropout: bool = Field(default=False, validation_alias="USE_DROPOUT")
    use_multiscale: bool = Field(default=False, validation_alias="USE_MULTISCALE")
    num_discriminators: int = Field(default=1, validation_alias="NUM_DISCRIMINATORS")
    input_nc: int = Field(default=1, validation_alias="INPUT_NC")
    output_nc: int = Field(default=1, validation_alias="OUTPUT_NC")
    ngf: int = Field(default=64, validation_alias="NGF")
    ndf: int = Field(default=64, validation_alias="NDF")
    n_layers_d: int = Field(default=3, validation_alias="N_LAYERS_D")
    norm: Literal["instance", "batch", "none", "syncbatch"] = Field(default="instance", validation_alias="NORM")

    seed: int = Field(default=42, validation_alias="SEED")
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", validation_alias="DEVICE"
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    use_tensorboard: bool = Field(default=True, validation_alias="USE_TENSORBOARD")
    use_torchmetrics: bool = Field(default=True, validation_alias="USE_TORCHMETRICS")
    use_wandb: bool = Field(default=True, validation_alias="USE_WANDB")
    wandb_project_name: str = Field(default="cyclegan", validation_alias="WANDB_PROJECT")
    wandb_entity: str | None = Field(default=None, validation_alias="WANDB_ENTITY")
    wandb_mode: Literal["online", "offline", "disabled"] = Field(
        default="online", validation_alias="WANDB_MODE"
    )
    log_dir: str = Field(
        default=str(_MODULE_ROOT / "runs"),
        validation_alias="LOG_DIR",
    )
    log_every: int = Field(default=100, validation_alias="LOG_EVERY")
    image_log_every: int = Field(default=500, validation_alias="IMAGE_LOG_EVERY")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str = Field(
        default=str(_MODULE_ROOT / "checkpoints"),
        validation_alias="CHECKPOINT_DIR",
    )
    save_every_epochs: int = Field(default=1, validation_alias="SAVE_EVERY_EPOCHS")
    save_every_steps: int = Field(default=0, validation_alias="SAVE_EVERY_STEPS")
    save_full_model: bool = Field(default=False, validation_alias="SAVE_FULL_MODEL")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
