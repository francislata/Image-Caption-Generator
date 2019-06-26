"""This contains the base definition of a model."""

from typing import Optional, Any
from torch.nn import Module, CrossEntropyLoss
from torch.optim.optimizer import Optimizer #pylint: disable=no-name-in-module
from torch.optim import SGD
from ..datasets.dataset import Dataset

class Model:
    """This subclass contains the basic definition of a model."""
    def __init__(self,
                 network: Module,
                 train_ds: Dataset,
                 val_ds: Optional[Dataset] = None,
                 test_ds: Optional[Dataset] = None) -> None:
        self.network = network
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

    def num_epochs(self) -> int: #pylint: disable=no-self-use
        """Returns the number of epochs to train the model for."""
        return 10

    def train(self,
              train_dl_kwargs: Any,
              val_dl_kwargs: Optional[Any] = None) -> None:
        """Trains the network."""
        optimizer, loss_fn = self.optimizer(), self.loss_fn()
        train_dl = self.train_ds.create_dataloader(**train_dl_kwargs)

        if self.val_ds and val_dl_kwargs:
            val_dl = self.val_ds.create_dataloader(**val_dl_kwargs)

        for _ in range(self.num_epochs()):
            for inps, lbls, lbl_lengths in train_dl:
                optimizer.zero_grad()

                preds = self.network(inps, lbls, lbl_lengths)
                loss = loss_fn(preds, lbls)

                loss.backward()
                optimizer.step()

            if val_dl:
                ...
