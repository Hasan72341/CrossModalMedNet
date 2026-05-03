from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_MODULE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_ROOT.parent
_REPO_ROOT = _PROJECT_ROOT.parent
_DEFAULT_ENV_FILE = _MODULE_ROOT / ".env"


class Settings(BaseSettings):
    # Data
    manifest_path: str = Field(
        default="/usershome/cs671_user4/SynthRAD2023_Dataset/manifest.csv",
        validation_alias="MANIFEST_PATH",
    )
    split: Literal["train", "val", "test"] = Field(default="train", validation_alias="SPLIT")
    eval_split: Literal["train", "val", "test"] = Field(default="val", validation_alias="EVAL_SPLIT")
    region: str | None = Field(default="brain", validation_alias="REGION")
    data_mode: Literal["2d", "2.5d"] = Field(default="2.5d", validation_alias="DATA_MODE")
    slice_axis: Literal["axial", "coronal", "sagittal"] = Field(
        default="axial", validation_alias="SLICE_AXIS"
    )
    num_slices: int = Field(default=3, validation_alias="NUM_SLICES")
    image_size: int = Field(default=256, validation_alias="IMAGE_SIZE")
    resize_mode: Literal["resize", "center_crop", "none"] = Field(
        default="resize", validation_alias="RESIZE_MODE"
    )
    num_workers: int = Field(default=4, validation_alias="NUM_WORKERS")
    pin_memory: bool = Field(default=True, validation_alias="PIN_MEMORY")
    augment: bool = Field(default=False, validation_alias="AUGMENT")

    source_modality: Literal["ct", "mr"] = Field(default="ct", validation_alias="SOURCE_MODALITY")
    target_modality: Literal["ct", "mr"] = Field(default="mr", validation_alias="TARGET_MODALITY")
    prompt_source: str = Field(default="CT scan", validation_alias="PROMPT_SOURCE")
    prompt_target: str = Field(default="MRI scan", validation_alias="PROMPT_TARGET")
    normalize_method: Literal["minmax", "shift_scale"] = Field(
        default="minmax", validation_alias="NORMALIZE_METHOD"
    )
    ct_shift: float = Field(default=0.0, validation_alias="CT_SHIFT")
    ct_scale: float = Field(default=1.0, validation_alias="CT_SCALE")
    mri_shift: float = Field(default=0.0, validation_alias="MRI_SHIFT")
    mri_scale: float = Field(default=1.0, validation_alias="MRI_SCALE")

    # Model
    base_model: str = Field(default="stabilityai/sd-turbo", validation_alias="BASE_MODEL")
    lora_rank_unet: int = Field(default=8, validation_alias="LORA_RANK_UNET")
    lora_rank_vae: int = Field(default=4, validation_alias="LORA_RANK_VAE")
    pretrained_lora_path: str | None = Field(default=None, validation_alias="PRETRAINED_LORA_PATH")
    enable_xformers: bool = Field(default=False, validation_alias="ENABLE_XFORMERS")
    gradient_checkpointing: bool = Field(default=False, validation_alias="GRADIENT_CHECKPOINTING")
    allow_tf32: bool = Field(default=False, validation_alias="ALLOW_TF32")

    # Training
    batch_size: int = Field(default=1, validation_alias="BATCH_SIZE")
    max_train_epochs: int = Field(default=1000, validation_alias="MAX_TRAIN_EPOCHS")
    max_train_steps: int | None = Field(default=None, validation_alias="MAX_TRAIN_STEPS")
    gradient_accumulation_steps: int = Field(default=8, validation_alias="GRADIENT_ACCUMULATION_STEPS")
    learning_rate: float = Field(default=5e-6, validation_alias="LEARNING_RATE")
    lr_scheduler: str = Field(default="constant", validation_alias="LR_SCHEDULER")
    lr_warmup_steps: int = Field(default=500, validation_alias="LR_WARMUP_STEPS")
    lr_num_cycles: int = Field(default=1, validation_alias="LR_NUM_CYCLES")
    lr_power: float = Field(default=1.0, validation_alias="LR_POWER")
    adam_beta1: float = Field(default=0.9, validation_alias="ADAM_BETA1")
    adam_beta2: float = Field(default=0.999, validation_alias="ADAM_BETA2")
    adam_weight_decay: float = Field(default=1e-2, validation_alias="ADAM_WEIGHT_DECAY")
    adam_epsilon: float = Field(default=1e-8, validation_alias="ADAM_EPSILON")
    max_grad_norm: float = Field(default=10.0, validation_alias="MAX_GRAD_NORM")

    # Loss weights
    gan_disc_type: str = Field(default="vagan_clip", validation_alias="GAN_DISC_TYPE")
    gan_loss_type: str = Field(default="multilevel_sigmoid", validation_alias="GAN_LOSS_TYPE")
    lambda_gan: float = Field(default=0.5, validation_alias="LAMBDA_GAN")
    lambda_denoise: float = Field(default=1.0, validation_alias="LAMBDA_DENOISE")
    lambda_lpips: float = Field(default=5.0, validation_alias="LAMBDA_LPIPS")
    lambda_l1: float = Field(default=10.0, validation_alias="LAMBDA_L1")

    # Logging and outputs
    output_dir: str = Field(
        default="outputs/pix2pix_diffusion_standard",
        validation_alias="OUTPUT_DIR",
    )
    checkpoint_dir: str = Field(
        default="checkpoints/pix2pix_diffusion_standard",
        validation_alias="CHECKPOINT_DIR",
    )
    log_dir: str = Field(
        default="runs/pix2pix_diffusion_standard",
        validation_alias="LOG_DIR",
    )
    report_to: str = Field(default="wandb", validation_alias="REPORT_TO")
    tracker_project_name: str = Field(
        default="paired_ct2mri_brain",
        validation_alias="TRACKER_PROJECT_NAME",
    )
    viz_freq: int = Field(default=99999999, validation_alias="VIZ_FREQ")
    max_visuals: int = Field(default=4, validation_alias="MAX_VISUALS")
    wandb_max_visuals: int = Field(default=4, validation_alias="WANDB_MAX_VISUALS")
    log_every: int = Field(default=1, validation_alias="LOG_EVERY")
    save_25d_slices: bool = Field(default=True, validation_alias="SAVE_25D_SLICES")
    checkpoint_steps: int = Field(default=999999, validation_alias="CHECKPOINT_STEPS")

    # Validation
    validation_steps: int = Field(default=999999, validation_alias="VALIDATION_STEPS")
    eval_every_n_epochs: int = Field(default=5, validation_alias="EVAL_EVERY_N_EPOCHS")
    validation_num_images: int = Field(default=16, validation_alias="VALIDATION_NUM_IMAGES")
    log_checkpoints_to_wandb: bool = Field(default=False, validation_alias="LOG_CHECKPOINTS_TO_WANDB")

    # Runtime
    seed: int = Field(default=42, validation_alias="SEED")
    mixed_precision: str | None = Field(default=None, validation_alias="MIXED_PRECISION")
    cuda_memory_fraction: float | None = Field(
        default=None,
        validation_alias="CUDA_MEMORY_FRACTION",
    )

    model_config = SettingsConfigDict(env_file=str(_DEFAULT_ENV_FILE), extra="ignore")


def get_settings(env_file: Path | None = None) -> Settings:
    if env_file is not None:
        return Settings(_env_file=str(env_file))
    return Settings()
