########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import datetime
import logging
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader

from src.training_args import TrainingArguments

deepspeed_version = None
try:
    import deepspeed

    deepspeed_version = deepspeed.__version__
except ImportError:
    pass  # deepspeed not installed


logging.basicConfig(level=logging.INFO)


class Trainer:  # Your custom Trainer helper class for adding args
    @staticmethod
    def add_argparse_args(parser):
        # This is where your Trainer might add its specific arguments
        parser.add_argument(
            "--num_nodes", default=1, type=int, help="Number of training nodes."
        )
        parser.add_argument(
            "--devices", default=1, type=int, help="Number of devices per node."
        )
        parser.add_argument(
            "--accelerator",
            default="gpu",
            type=str,
            help="Accelerator type (e.g., 'gpu', 'cpu').",
        )
        parser.add_argument(
            "--strategy", default="ddp", type=str, help="Distributed training strategy."
        )
        parser.add_argument(
            "--precision",
            default="fp16",
            type=str,
            help="Training precision (e.g., 'fp16', 'bf16').",
        )
        return parser


# --- Main execution block ---
if __name__ == "__main__":

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()
    default_args = TrainingArguments()
    # Add args from TrainingArguments
    parser = default_args.to_argparse(parser)
    parser = Trainer.add_argparse_args(
        parser
    )  # Add args from your custom Trainer helper

    # Parse arguments ONLY ONCE
    args = parser.parse_args()

    # Convert a subset of args into the TrainingArguments dataclass
    training_args = TrainingArguments.from_argparse(args)

    ########################################################################################################

    # Use training_args for arguments defined in TrainingArguments
    if training_args.random_seed >= 0:
        print(
            f"########## WARNING: GLOBAL SEED {training_args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n"
            * 3
        )
        seed_everything(training_args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings(
        "ignore", ".*Consider increasing the value of the `num_workers` argument*"
    )
    warnings.filterwarnings(
        "ignore", ".*The progress bar already tracks a metric with the*"
    )
    # os.environ["WDS_SHOW_SEED"] = "1"

    # Use training_args where appropriate
    args.my_timestamp = datetime.datetime.today().strftime(
        "%Y-%m-%d-%H-%M-%S"
    )  # This is fine if you modify 'args' directly
    args.enable_checkpointing = False  # Fine to modify 'args' directly
    args.replace_sampler_ddp = False  # Fine to modify 'args' directly
    args.logger = False  # Fine to modify 'args' directly
    args.gradient_clip_val = training_args.grad_clip  # Use training_args here
    args.num_sanity_val_steps = 0  # Fine to modify 'args' directly
    args.check_val_every_n_epoch = int(1e20)  # Fine to modify 'args' directly
    args.log_every_n_steps = int(1e20)  # Fine to modify 'args' directly
    args.max_epochs = -1  # continue forever # Fine to modify 'args' directly

    # Crucially, `args.betas` should be set using `training_args.beta1` and `training_args.beta2`
    args.betas = (training_args.beta1, training_args.beta2)

    # `args.real_bsz` uses parameters from both `args` (num_nodes, devices) and `training_args` (micro_bsz)
    args.real_bsz = int(args.num_nodes) * \
        int(args.devices) * training_args.micro_bsz

    os.environ["RWKV_MY_TESTING"] = training_args.my_testing
    os.environ["RWKV_CTXLEN"] = str(training_args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(training_args.head_size)

    if training_args.dim_att <= 0:
        training_args.dim_att = (
            training_args.n_embd
        )  # Modify training_args directly if it's the source of truth
    if training_args.dim_ffn <= 0:
        multiplier = 4 if training_args.my_testing == "x070" else 3.5
        training_args.dim_ffn = int(
            (training_args.n_embd * multiplier) // 32 * 32
        )  # multiple of 32

    # run_name construction: use training_args
    # Still setting on 'args' for consistency if 'run_name' is used broadly
    args.run_name = f"{training_args.vocab_size} ctx{training_args.ctx_len} L{training_args.n_layer} D{training_args.n_embd}"
    if not os.path.exists(training_args.proj_dir):
        os.makedirs(training_args.proj_dir)

    # Use training_args for calculations
    training_args.epoch_count = training_args.magic_prime // 40320
    training_args.epoch_steps = (
        40320 // args.real_bsz
    )  # Note: args.real_bsz is calculated earlier

    assert training_args.epoch_steps * args.real_bsz == 40320

    if training_args.train_stage >= 2:  # find latest saved model
        list_p = []
        for p in os.listdir(training_args.proj_dir):
            if p.startswith("rwkv") and p.endswith(".pth"):
                p = ((p.split("-"))[1].split("."))[0]
                if p != "final":
                    if p == "init":
                        p = -1
                    else:
                        p = int(p)
                    list_p += [p]
        list_p.sort()
        max_p = list_p[-1]
        if len(list_p) > 1:
            # args.my_pile_prev_p is not in TrainingArguments, so set it on args
            args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
        if max_p == -1:
            training_args.load_model = f"{training_args.proj_dir}/rwkv-init.pth"
        else:
            training_args.load_model = f"{training_args.proj_dir}/rwkv-{max_p}.pth"
            if training_args.warmup_steps < 0:
                training_args.warmup_steps = 10
        training_args.epoch_begin = max_p + 1

    # Use training_args for calculations
    samples_per_epoch = training_args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * training_args.ctx_len

    # The header printout: mix `args` (for Trainer-specific params) and `training_args`
    rank_zero_info(
        (
            "############################################################################\n"
            f"#\n"
            f"# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, "
            f"bsz {args.num_nodes}x{args.devices}x{training_args.micro_bsz}={args.real_bsz}, "
            # Use training_args.grad_cp
            f"{args.strategy} {'with grad_cp' if training_args.grad_cp > 0 else ''}\n"
            f"#\n"
            f"# Data = {training_args.data_file} ({training_args.data_type}), ProjDir = {training_args.proj_dir}\n"
            f"#\n"
            f"# Epoch = {training_args.epoch_begin} to {training_args.epoch_begin + training_args.epoch_count - 1} "
            f"(will continue afterwards), save every {training_args.epoch_save} epoch\n"
            f"#\n"
            f'# Each "epoch" = {training_args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens\n'
            f"#\n"
            f"# Model = {training_args.n_layer} n_layer, {training_args.n_embd} n_embd, {training_args.ctx_len} ctx_len\n"
            f"#\n"
            f"# Adam = lr {training_args.lr_init} to {training_args.lr_final}, warmup {training_args.warmup_steps} steps, "
            # Use args.betas which was set earlier
            f"beta {args.betas}, eps {training_args.adam_eps}\n"
            f"#\n"
            f"# Found torch {torch.__version__}, recommend latest torch\n"
            f"# Found deepspeed {deepspeed_version}, recommend latest deepspeed\n"
            f"# Found pytorch_lightning {pl.__version__}, recommend 1.9.5\n"
            f"#\n"
            "############################################################################"
        )
    )

    # Printing `vars(args)` is useful for debugging, as it shows all parsed and modified arguments
    rank_zero_info(str(vars(args)) + "\n")

    assert training_args.data_type in ["binidx"]

    # Use training_args
    if training_args.lr_final == 0 or training_args.lr_init == 0:
        rank_zero_info(
            "\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n"
        )

    assert args.precision in [
        "fp32",
        "tf32",
        "fp16",
        "bf16",
    ]  # 'precision' is on `args`
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info(
                "\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n"
            )
    if args.precision == "fp16":
        rank_zero_info(
            "\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n"
        )

    # Use training_args
    if training_args.compile == 1:
        os.environ["RWKV_COMPILE_ON"] = "1"
    else:
        os.environ["RWKV_COMPILE_ON"] = "0"

    # Use args
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"  # somehow incompatible
        os.environ["RWKV_COMPILE_ON"] = "0"  # somehow incompatible
    else:
        os.environ["RWKV_JIT_ON"] = "1"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":  # Use args
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Convert precision string to int/bf16, store on `args` for consistency with how PL might use it
    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from src.dataset import MyDataset
    from src.trainer import (  # Assuming these are actual imports
        generate_init_weight, train_callback)

    # Pass training_args
    train_data = MyDataset(training_args)
    training_args.vocab_size = (
        train_data.vocab_size
    )  # Update vocab_size in training_args

    from src.model import RWKV

    # Pass training_args
    model = RWKV(training_args)

    # Use training_args
    if (
        len(training_args.load_model) == 0 or training_args.train_stage == 1
    ):  # shall we build the initial weights?
        init_weight_name = f"{training_args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        training_args.load_model = init_weight_name

    rank_zero_info(
        f"########## Loading {training_args.load_model}... ##########")
    try:
        load_dict = torch.load(training_args.load_model, map_location="cpu")
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith("_forward_module."):
                load_dict[k.replace("_forward_module.", "")] = load_dict[k]
                del load_dict[k]
    except BaseException:
        rank_zero_info(f"Bad checkpoint {training_args.load_model}")
        if training_args.train_stage >= 2:  # try again using another checkpoint
            max_p = args.my_pile_prev_p  # This variable might only be on `args`
            if max_p == -1:
                training_args.load_model = f"{training_args.proj_dir}/rwkv-init.pth"
            else:
                training_args.load_model = f"{training_args.proj_dir}/rwkv-{max_p}.pth"
            training_args.epoch_begin = max_p + 1
            rank_zero_info(f"Trying {training_args.load_model}")
            load_dict = torch.load(
                training_args.load_model, map_location="cpu")

    if training_args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]
    model.load_state_dict(load_dict)

    # Use PLTrainer (the PyTorch Lightning Trainer) and pass the combined `args`
    # because it expects an object that contains all its relevant parameters.
    trainer = PLTrainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)],  # train_callback might expect `args`
    )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")

    # Use args for deepspeed strategy
    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = (
            training_args.ds_bucket_mb * 1000 * 1000  # Use training_args.ds_bucket_mb
        )
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = (
            training_args.ds_bucket_mb * 1000 * 1000  # Use training_args.ds_bucket_mb
        )

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(
        train_data,
        shuffle=False,
        pin_memory=True,
        batch_size=training_args.micro_bsz,  # Use training_args.micro_bsz
        num_workers=1,
        persistent_workers=False,
        drop_last=True,
    )

    trainer.fit(model, data_loader)
