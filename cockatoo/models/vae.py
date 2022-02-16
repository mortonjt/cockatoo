import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from cockatoo.dataset.biom import BiomDataset
from cockatoo.dataset.biom import collate_single_f
from cockatoo.modules._vae import LinearVAE

import pytorch_lightning as pl

from biom import load_table
import pandas as pd
import os


class BiomDataModule(pl.LightningDataModule):
    def __init__(self, train_biom, test_biom, valid_biom,
                 metadata=None, batch_size=10, num_workers=1):
        super().__init__()
        self.train_biom = train_biom
        self.test_biom = test_biom
        self.val_biom = valid_biom
        self.batch_size = batch_size
        self.num_workers = num_workers
        if metadata is not None:
            self.metadata = pd.read_table(
                metadata, dtype=str)
            index_name = self.metadata.columns[0]
            self.metadata = self.metadata.set_index(index_name)
        else:
            self.metadata = None
        self.collate_f = collate_single_f

    def train_dataloader(self):
        train_dataset = BiomDataset(
            load_table(self.train_biom),
            metadata=self.metadata)
        batch_size = min(len(train_dataset) - 1, self.batch_size)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size,
            collate_fn=self.collate_f, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = BiomDataset(
            load_table(self.val_biom),
            metadata=self.metadata)
        batch_size = min(len(val_dataset) - 1, self.batch_size)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size,
            collate_fn=self.collate_f, shuffle=False,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True)
        return val_dataloader

    def test_dataloader(self):
        test_dataset = BiomDataset(
            load_table(self.test_biom),
            metadata=self.metadata)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            collate_fn=self.collate_f,
            shuffle=False, num_workers=self.num_workers,
            drop_last=True, pin_memory=True)
        return test_dataloader


class MultVAE(pl.LightningModule):
    def __init__(self, input_dim, latent_dim=32,
                 learning_rate=0.001):
        super().__init__()
        self.vae = LinearVAE(input_dim, latent_dim)

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        self.vae.train()   # may not need this
        counts = batch.to(self.device)
        neg_elbo = self.vae(counts)
        # TODO : add tensorboard metrics
        return {'loss': neg_elbo}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            counts = batch.to(self.device)
            neg_elbo = self.vae(counts)
            # TODO: add tensorboard metrics
            return {'loss': neg_elbo}

    # TODO: implement validation_epoch_end
    def configure_optimizers(self):
        # TODO: pass learning_rate from __init__ to here
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=0.001)
        return optimizer
