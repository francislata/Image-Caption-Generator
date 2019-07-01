"""This contains the base definition of a model."""

from typing import Optional, Any, Mapping
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim.optimizer import Optimizer #pylint: disable=no-name-in-module
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from ..datasets.dataset import Dataset

class Model:
    """This subclass contains the basic definition of a model."""
    def __init__(self,
                 network_cls: type,
                 train_ds: Dataset,
                 val_ds: Optional[Dataset] = None,
                 test_ds: Optional[Dataset] = None,
                 network_kwargs: Optional[Mapping] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network_cls(**(network_kwargs if network_kwargs else {})).to(self.device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def optimizer(self) -> Optimizer:
        """Returns the optimizer to use."""
        return SGD(self.network.parameters(),
                   lr=self.learning_rate(),
                   momentum=self.momentum(),
                   weight_decay=self.weight_decay())

    def loss_fn(self) -> Module: #pylint: disable=no-self-use
        """Returns the loss function to optimize."""
        return CrossEntropyLoss()

    def learning_rate(self) -> float: #pylint: disable=no-self-use
        """Returns the learning to use for the optimizer."""
        return 1e-3

    def momentum(self) -> float: #pylint: disable=no-self-use
        """Returns the amount of momentum to use."""
        return 0.0

    def weight_decay(self) -> float: #pylint: disable=no-self-use
        """Returns the amount of weight decay to use."""
        return 0.0

    def train(self,
              num_epochs: int,
              train_dl_kwargs: Any,
              val_dl_kwargs: Any) -> None:
        """Trains the network."""
        optimizer, loss_fn = self.optimizer(), self.loss_fn()
        train_dl = self.train_ds.create_dataloader(**train_dl_kwargs)

        if self.val_ds:
            val_dl_kwargs = val_dl_kwargs if val_dl_kwargs else train_dl_kwargs
            val_dl = self.val_ds.create_dataloader(**val_dl_kwargs)

        wandb.watch(self.network)

        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(epoch, train_dl, loss_fn, optimizer)
            print('[Epoch {}] Training loss: {:.3f}'.format(epoch, train_loss))

            if val_dl:
                val_loss = self._run_epoch(epoch, val_dl, loss_fn, optimizer, is_training=False)
                print('[Epoch {}] Validation loss: {:.3f}'.format(epoch, val_loss))

    def _run_epoch(self,
                   epoch: int,
                   dataloader: DataLoader,
                   loss_fn: Module,
                   optimizer: Optimizer,
                   is_training=True) -> float:
        """Runs an epoch through the dataset."""
        if is_training:
            self.network.train()
        else:
            self.network.eval()

        losses = []

        for inps, lbls, lbl_lengths in tqdm(dataloader, desc='Epoch {}'.format(epoch)):
            inps, lbls = inps.to(self.device), lbls.to(self.device)

            if is_training:
                optimizer.zero_grad()

            preds = self.network(inps, lbls, lbl_lengths)
            preds = preds.view(preds.size(1), preds.size(2), preds.size(0))
            lbls = lbls.t()

            loss = loss_fn(preds, lbls)
            losses.append(loss.item())

            if is_training:
                loss.backward()
                optimizer.step()

            if is_training:
                wandb.log({'train_loss': loss})
            else:
                wandb.log({'val_loss': loss})

        return sum(losses) / len(losses)
