from dataclasses import dataclass
from simple_parsing import Serializable, list_field

@dataclass
class BaseSaeConfig(Serializable):
    """
    Base configuration for training a sparse autoencoder.
    """
    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""


@dataclass
class SaeConfig(BaseSaeConfig):
    """
    Configuration for a dense sparse autoencoder.
    """
    pass


@dataclass
class ConvSaeConfig(BaseSaeConfig):
    """
    Configuration for a convolutional sparse autoencoder.
    """
    kernel_size: int = 3
    """Size of the convolutional kernel."""

    dilation: int = 1
    """Dilation of the convolutional kernel."""

    groups: int = 1
    """Number of groups for convolution."""

    def __post_init__(self):
        assert self.expansion_factor % self.groups == 0, "expansion_factor must be divisible by groups."


@dataclass
class TrainConfig(Serializable):
    sae: BaseSaeConfig
    """Sparse Autoencoder configuration (can be SaeConfig or ConvSaeConfig)."""

    batch_size: int = 8
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train SAEs on."""

    layers: list[int] = list_field()
    """List of layer indices to train SAEs on."""

    layer_stride: int = 1
    """Stride between layers to train SAEs on."""

    transcode: bool = False
    """Predict the output of a module given its input."""

    distribute_modules: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1

    arch_type: str = "sae"

    def __post_init__(self):
        assert not (
            self.layers and self.layer_stride != 1
        ), "Cannot specify both `layers` and `layer_stride`."
