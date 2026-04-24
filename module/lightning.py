from lightning import LightningModule
import torch.nn.functional as F
from module.classifier import MLP
import torch

class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()

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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

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