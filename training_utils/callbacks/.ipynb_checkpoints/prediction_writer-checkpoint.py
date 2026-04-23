import os
import shutil

import lightning as L
import torch
from genericpath import isfile


class PredictionWriter(L.pytorch.callbacks.BasePredictionWriter):

    def __init__(self, save_dir):

        super().__init__()

        self.save_dir = save_dir

    
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):

        super().setup(trainer, pl_module, stage)

        os.makedirs(self.save_dir, exist_ok=True)


    def write_on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, prediction, batch_indices, batch, batch_idx, dataloader_idx):

        epoch = trainer.current_epoch

        file_path = os.path.join(self.save_dir, f"prediction_epoch_{epoch}_batch_{batch_idx}.pt")

        torch.save(
            {
                "batch": batch,
                "prediction": prediction,
                "batch_indices": batch_indices,
                "dataloader_idx": dataloader_idx
            },
            file_path
        )
    
    def on_predict_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        
        super().on_predict_epoch_start(trainer, pl_module)
        
        for name in os.listdir(self.save_dir):
            path = os.path.join(self.save_dir, name)

            if os.path.isfile(path):
                os.remove(path)
            if os.path.isdir(path):
                shutil.rmtree(path)
        return
    
    






