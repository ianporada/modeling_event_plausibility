import logging
from datetime import datetime
from typing import Optional

import click
import datasets
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoConfig, AutoModelForSequenceClassification

from data import DataModule


@click.command()
@click.option('--model_name_or_path', default='roberta-base', help='Transformers model name.')
@click.option('--cache_dir', default='./cache', help='The person to greet.')
@click.option('--data_dir', default='./data', help='The person to greet.')


class PlausibilityModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 10000,
        weight_decay: float = 0.0,
        train_batch_size: int = 128,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                        config=self.config)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        scores = torch.nn.functional.log_softmax(logits, dim=1)

        preds = torch.argmax(logits, axis=1)

        labels = batch["labels"]
        labels = torch.argmax(labels, axis=1)

        return {"loss": val_loss, "preds": preds, "labels": labels, "scores": scores}

    def validation_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
            labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
            scores = torch.cat([x["scores"] for x in output]).detach().cpu().numpy()
            loss = torch.stack([x["loss"] for x in output]).mean()

            if i == 0:
                self.log(f"val_loss", loss, prog_bar=True)
                self.log(f"accuracy", sklearn.metrics.accuracy_score(labels, preds), prog_bar=True)
            else:
                dataset_name = {1:'pep', 2:'20q'}[i]

                self.log(f"{dataset_name}_val_loss", loss, prog_bar=True)
                self.log(f"{dataset_name}_accuracy",
                         sklearn.metrics.accuracy_score(labels, preds),
                         prog_bar=True)
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores[:,1])
                self.log(f"{dataset_name}_auc", sklearn.metrics.auc(fpr, tpr), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def main(model_name_or_path, cache_dir, data_dir):
    pl.seed_everything(42)

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    dm = DataModule(model_name_or_path=model_name_or_path,
                    cache_dir=cache_dir,
                    data_dir=data_dir)
    dm.prepare_data()
    dm.setup("fit")

    model = PlausibilityModule(
        model_name_or_path=model_name_or_path,
        num_labels=2
    )

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        val_check_interval=250,
        logger=pl.loggers.CSVLogger("logs")
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
