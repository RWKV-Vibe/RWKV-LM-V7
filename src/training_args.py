from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """
    Configuration for training RWKV models.

    This class encapsulates all command-line arguments as class members,
    providing better type hinting and default value management.
    """

    load_model: str = field(
        default="", metadata={"help": "Full path to the model checkpoint (.pth)."}
    )
    wandb: str = field(
        default="", metadata={"help": "Weights & Biases project name. Empty string to disable wandb."}
    )
    proj_dir: str = field(
        default="out", metadata={"help": "Project directory for outputs and checkpoints."}
    )
    random_seed: int = field(
        default=-1, metadata={"help": "Random seed for reproducibility. -1 for no specific seed."}
    )

    data_file: str = field(
        default="", metadata={"help": "Path to the training data file."}
    )
    data_type: str = field(
        default="utf-8", metadata={"help": "Type of data encoding (e.g., 'utf-8')."}
    )
    vocab_size: int = field(
        default=0, metadata={"help": "Vocabulary size. 0 means auto-detect (for char-level LM and .txt data)."}
    )

    ctx_len: int = field(
        default=1024, metadata={"help": "Context length for the model."}
    )
    epoch_steps: int = field(
        default=1000, metadata={"help": "Number of steps per mini 'epoch'."}
    )
    epoch_count: int = field(
        default=500, metadata={"help": "Number of 'epochs' to train. Will continue afterwards with lr = lr_final."}
    )
    epoch_begin: int = field(
        default=0, metadata={"help": "Starting epoch index (useful when resuming training)."}
    )
    epoch_save: int = field(
        default=5, metadata={"help": "Save the model every N 'epochs'."}
    )

    micro_bsz: int = field(
        default=12, metadata={"help": "Micro batch size (batch size per GPU)."}
    )
    n_layer: int = field(
        default=6, metadata={"help": "Number of transformer layers."}
    )
    n_embd: int = field(
        default=512, metadata={"help": "Embedding dimension (model dimension)."}
    )
    dim_att: int = field(
        default=0, metadata={"help": "Attention dimension. 0 means calculated from n_embd."}
    )
    dim_ffn: int = field(
        default=0, metadata={"help": "Feed-forward network dimension. 0 means calculated from n_embd."}
    )

    lr_init: float = field(
        default=6e-4, metadata={"help": "Initial learning rate (e.g., 6e-4 for L12-D768)."}
    )
    lr_final: float = field(
        default=1e-5, metadata={"help": "Final learning rate after decay."}
    )
    warmup_steps: int = field(
        default=-1, metadata={"help": "Number of warmup steps for learning rate. -1 for no warmup."}
    )
    beta1: float = field(
        default=0.9, metadata={"help": "Adam optimizer beta1 parameter."}
    )
    beta2: float = field(
        default=0.99, metadata={"help": "Adam optimizer beta2 parameter."}
    )
    adam_eps: float = field(
        default=1e-18, metadata={"help": "Adam optimizer epsilon parameter."}
    )
    grad_cp: int = field(
        default=0, metadata={"help": "Gradient checkpointing (0: off, 1: on). Saves VRAM, but slower."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for optimizer (e.g., 0.1)."}
    )
    grad_clip: float = field(
        default=1.0, metadata={"help": "Gradient clipping value. Reduce for problematic samples."}
    )

    train_stage: int = field(
        default=0, metadata={"help": "Special training stage mode (e.g., 0 for default)."}
    )
    ds_bucket_mb: int = field(
        default=200, metadata={"help": "DeepSpeed bucket size in MB. 200 is usually enough."}
    )

    head_size: int = field(
        default=64, metadata={"help": "Head size for attention."}
    )
    load_partial: int = field(
        default=0, metadata={"help": "Whether to load partial model weights (0: off, 1: on)."}
    )
    magic_prime: int = field(
        default=0, metadata={"help": "Magic prime for specific data handling (if applicable)."}
    )
    my_testing: str = field(
        default="x070", metadata={"help": "Custom testing parameter."}
    )
    my_exit_tokens: int = field(
        default=0, metadata={"help": "Number of tokens after which to exit during generation/evaluation."}
    )
    compile: int = field(
        default=1, metadata={"help": "Whether to compile the model for faster execution (0: off, 1: on)."}
    )

    # Note: Trainer.add_argparse_args(parser) implies that Trainer adds its own arguments.
    # You would typically handle those by either:
    # 1. Inheriting from a base TrainerArgs class if available.
    # 2. Having a separate dataclass for Trainer-specific args and composing them.
    # 3. Manually adding them here if they are always part of the main config.
    # For now, I'll assume they would be handled by the Trainer itself or composed.

    def to_argparse(self, parser):
        """
        Populates an ArgumentParser with arguments from this dataclass.
        This allows you to parse command-line arguments into an instance of this class.
        """
        for field_name, field_obj in self.__dataclass_fields__.items():
            field_type = field_obj.type
            default_value = field_obj.default

            # Handle Optional types to get the underlying type
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                # Get the type inside Optional[T]
                field_type = field_type.__args__[0]

            # Special handling for Tuple default values (betas)
            if isinstance(default_value, tuple):
                # We assume tuples are for things like (beta1, beta2) which are typically parsed as float lists
                # If they are fixed like (0.9, 0.99), then argparse needs to handle it as individual args or a string
                # For simplicity, we'll keep them as simple types for now.
                # If `betas` was a single argument, it would be parser.add_argument("--betas",
                #  nargs=2, type=float, default=[0.9, 0.99])
                pass

            parser.add_argument(
                f"--{field_name}",
                default=default_value,
                type=field_type,
                help=field_obj.metadata.get("help", "")
            )
        return parser

    @classmethod
    def from_argparse(cls, parsed_args):
        """
        Creates an instance of TrainingArguments from parsed argparse arguments.
        """
        # Create an empty instance
        instance = cls()
        # Populate it with values from parsed_args
        for field_name in cls.__dataclass_fields__:
            if hasattr(parsed_args, field_name):
                setattr(instance, field_name, getattr(parsed_args, field_name))
        return instance
