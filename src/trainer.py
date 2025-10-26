import datetime
import math
import time

import pytorch_lightning as pl
import torch
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only


def my_save(args, trainer, dd, ff):
    """
    Save model or data to disk, using the trainer's DeepSpeed-aware checkpoint method when appropriate.
    
    Parameters:
        args: Configuration object with a `strategy` attribute (iterable or string) used to detect DeepSpeed stage 3.
        trainer: PyTorch Lightning Trainer used to save checkpoints when DeepSpeed stage 3 is active.
        dd: The object to save (e.g., state dict or full object) when not using trainer checkpointing.
        ff (str): Destination file path for the saved checkpoint or object.
    """
    if "deepspeed_stage_3" in args.strategy:
        trainer.save_checkpoint(ff, weights_only=True)
    else:
        torch.save(dd, ff)


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps

        if args.my_exit_tokens != 0:  # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (
                abs(args.my_exit_tokens) - warmup_tokens
            )
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init
            lr_mult = (0.5 + lr_final_factor / 2) + (
                0.5 - lr_final_factor / 2
            ) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ("deepspeed_stage_3" in args.strategy):
                    final_path = f"{args.proj_dir}/rwkv-final.pth"
                    my_save(
                        args,
                        trainer,
                        pl_module.state_dict(),
                        final_path,
                    )
                    rank_zero_info(
                        f"\n✅ End of training. Model saved to: {final_path}\n"
                    )
                    import sys

                    sys.exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.01 + 0.99 * trainer.global_step / w_step)

        wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            param_group["lr"] = lr * param_group["my_lr_scale"]

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(
                    f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n"
                )
                try:
                    rank_zero_info(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except BaseException:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    rank_zero_info("Login to wandb...")
                    import wandb

                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Update logging, running loss statistics, optional Weights & Biases metrics, and save a final checkpoint at a configured milestone after each training batch.
        
        Parameters:
            trainer: Lightning trainer instance used to read and update global step, timing, and per-run state (e.g., my_time_ns, my_loss_sum, my_loss_count, my_lr, my_wd, my_wandb).
            pl_module: The LightningModule currently being trained; used to obtain state_dict when saving the final checkpoint.
            outputs: Model outputs for the batch; may be a dict with a 'loss' tensor or a tensor-like loss value.
            batch: The current training batch (unused by this callback aside from signature compatibility).
            batch_idx: Index of the current batch within the epoch (unused by this callback aside from signature compatibility).
        
        Behavior:
            - When running on the primary process (trainer.is_global_zero), updates timing and computes iteration/throughput metrics, updates running loss statistics (trainer.my_loss, trainer.my_loss_sum, trainer.my_loss_count, trainer.my_epoch_loss), logs lr and loss to the progress bar, and, if configured, logs a metrics dict to Weights & Biases including loss, lr, wd, Gtokens, and kt/s when available.
            - When running on the primary process or when using DeepSpeed stage 3, checks whether the current real step matches the configured magic_prime milestone and, if so, saves the model state_dict to rwkv-final.pth and logs a completion message.
        
        Notes:
            - This callback mutates trainer state fields described above.
            - No value is returned.
        """
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except BaseException:
                pass
            trainer.my_time_ns = t_now
            if isinstance(outputs, dict):
                current_loss = outputs['loss'].item()
            else:
                current_loss = outputs.item()
            trainer.my_loss = current_loss
            trainer.my_loss_sum += current_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {
                    "loss": trainer.my_loss,
                    "lr": trainer.my_lr,
                    "wd": trainer.my_wd,
                    "Gtokens": real_step * token_per_step / 1e9,
                }
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))

        if (trainer.is_global_zero) or (
            "deepspeed_stage_3" in args.strategy
        ):  # save pth
            if args.magic_prime > 0:
                if int(real_step) == int(args.magic_prime // args.real_bsz) - 1:
                    to_save_dict = pl_module.state_dict()
                    final_path = f"{args.proj_dir}/rwkv-final.pth"
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        final_path,
                    )
                    rank_zero_info(
                        f"\n✅ End of training. Model saved to: {final_path}\n"
                    )

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Prepare the training dataset for the upcoming epoch by setting distributed and epoch metadata.
        
        This sets dataset.global_rank, dataset.real_epoch (computed as args.epoch_begin + trainer.current_epoch), and dataset.world_size on the dataset obtained from trainer.train_dataloader. It also asserts that the dataset's representation contains "MyDataset".
        
        Parameters:
        	trainer (pl.Trainer): The trainer providing global_rank, current_epoch, and train_dataloader.
        	pl_module (pl.LightningModule): The LightningModule for the current training run (unused).
        """
        args = self.args
        dataset = trainer.train_dataloader.dataset
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or (
            "deepspeed_stage_3" in args.strategy
        ):  # save pth
            if (
                args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0
            ) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == "wds_img":
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith("encoder.") or k.startswith("decoder."):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    rank_zero_info("Error\n\n", e, "\n\n")

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(
                (
                    f"{args.epoch_begin + trainer.current_epoch} "
                    f"{trainer.my_epoch_loss:.6f} "
                    f"{math.exp(trainer.my_epoch_loss):.4f} "
                    f"{trainer.my_lr:.8f} "
                    f"{datetime.datetime.now()} "
                    f"{trainer.current_epoch}\n"
                )
            )
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.train_stage == 1:
        if len(model.args.load_model) > 0:
            rank_zero_info(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except BaseException:
                    rank_zero_info("missing", k)
                    import sys
                    sys.exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except BaseException:
                    tmp = mm[k].squeeze().clone()
                    rank_zero_info(k, src.shape, "-->", mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss - 1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1 - ii) + src[p0 + 1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    rank_zero_info(sss[:10], "...", sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    rank_zero_info(mmm[:10], "...", mmm[-10:])

    rank_zero_info(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.train_stage == 1:
        rank_zero_info("Done. Now go for stage 2.")
        import sys
        sys.exit(0)