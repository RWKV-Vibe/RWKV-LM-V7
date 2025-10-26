#################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#################################################################

import math

import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import Dataset

from .binidx import MMapIndexedDataset


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class MyDataset(Dataset):
    def __init__(self, args):
        """
        Initialize the dataset object, load the memory-mapped token dataset, compute dataset sizes and slots, initialize epoch/rank placeholders, and validate configuration invariants.
        
        Parameters:
            args: Configuration object containing required fields:
                - vocab_size: vocabulary size to store in self.vocab_size.
                - data_file: path used to create the MMapIndexedDataset.
                - ctx_len: context length used to compute dataset slots.
                - epoch_steps, real_bsz: used to compute samples_per_epoch (must equal 40320).
                - train_stage: (used for informational reporting).
                - magic_prime: integer that must be prime, satisfy magic_prime % 3 == 2, and have magic_prime / dataset_slot in (0.9, 1].
        
        Behavior:
            Sets the following attributes on self:
              - args, vocab_size, data (MMapIndexedDataset), data_size (number of tokens),
                samples_per_epoch, global_rank (0), real_epoch (0), world_size (1).
            Enforces these assertions:
              - samples_per_epoch == 40320
              - is_prime(magic_prime) is True
              - magic_prime % 3 == 2
              - 0.9 < magic_prime / dataset_slot <= 1
        """
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(
            self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        dataset_slot = self.data_size // args.ctx_len
        # add default rank parameter
        self.global_rank = 0
        self.real_epoch = 0
        self.world_size = 1

        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")

        dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y