from typing import Any

from lightning import LightningModule
import torch.nn.functional as F
from module.classifier import MLP
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC, MetricCollection

class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.metrics = MetricCollection({
            "auroc": AUROC(task="binary"),
        })

    def criterion(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y, reduction='mean')

    def get_input_from_batch(self, batch):
        x = torch.concat(batch[:-1], dim=-1)
        y = batch[-1]
        if y.dim() == 1:
            y = y.unsqueeze(1)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss}
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.log('train_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {
            "loss": loss,
            "y_hat": y_hat.detach(),
            "y": y.detach(),
        }
    
    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # update metrics
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        self.log('val_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.update(y_hat, y)
    
    def on_validation_epoch_end(self) -> None:
        self.log("val_aucroc", self.metrics["auroc"].compute(), on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        y_hat = self(x)
        return y_hat

class MLPClassifier(BaseModel):
    def __init__(
        self,
        hidden_dim: int=256,
        num_layers: int=4,
        batch_norm: bool=False,
        layer_norm: bool=True,
        output_dim: int=1,
        dropout: float=0.,
        activation: str="relu",
    ):
        super().__init__()

        self.net = MLP(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

class MLPLowLevelClassifier(MLPClassifier):

    def get_input_from_batch(self, batch):
        X_proc, cond_proc, X_hlf, y = batch
        X = torch.concat([X_proc, cond_proc], dim=-1)
        y = y.unsqueeze(1)
        return X, y
    
class MLPHighLevelClassifier(MLPClassifier):

    def get_input_from_batch(self, batch):
        X_proc, cond_proc, X_hlf, y = batch
        X = torch.concat([X_hlf, cond_proc], dim=-1)
        y = y.unsqueeze(1)
        return X, y

class MLPLatentClassifier(MLPClassifier):

    def get_input_from_batch(self, batch):
        X_proc, cond_proc, y = batch
        X = torch.concat([X_proc, cond_proc], dim=-1)
        y = y.reshape(-1, 1)
        return X, y