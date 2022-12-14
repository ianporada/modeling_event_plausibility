import logging
import os

import click
import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
import transformers
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AdamW, AutoConfig

from data import DataModule
from model.base import PlausibilityModel
from model.conceptinject import ConceptInject
from model.conceptmax import ConceptMax


class PlausibilityModule(pl.LightningModule):
    def __init__(
        self,
        model_type: str,
        model_name_or_path: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 10000,
        weight_decay: float = 0.0,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        **kwargs,
    ):
        """Initialize model and save hyperparameters."""
        super().__init__()

        self.save_hyperparameters()
        
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=1)
        if model_type == 'roberta':
            model = PlausibilityModel.from_pretrained(model_name_or_path, config=self.config)
        elif model_type == 'conceptmax':
            model = ConceptMax.from_pretrained(model_name_or_path, config=self.config)
        elif model_type == 'conceptinject':
            model = ConceptInject.from_pretrained(model_name_or_path, config=self.config)
            raise NotImplementedError('ConceptInject has not been added to this library.')
        else:
            raise ValueError(f'Invalid model type')
        self.model = model

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        # round logits before sigmoid for calculating model accuracy
        preds = (logits > 0).float()
        labels = batch['labels']
        return {'loss': val_loss, 'preds': preds, 'labels': labels, 'scores': logits}

    def validation_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            preds = torch.cat([x['preds'] for x in output]).detach().cpu().numpy()
            labels = torch.cat([x['labels'] for x in output]).detach().cpu().numpy()
            scores = torch.cat([x['scores'] for x in output]).detach().cpu().numpy()
            loss = torch.stack([x['loss'] for x in output]).mean()
            # each dataloader id corresponds to a different validation set
            if i == 0:
                # wikipedia validation
                self.log(f'val_loss', loss, prog_bar=True)
                self.log(f'val_acc', sklearn.metrics.accuracy_score(labels, preds), prog_bar=True)
            else:
                # plausibility validation
                dataset_name = {1:'pep', 2:'20q'}[i]
                self.log(f'{dataset_name}_val_loss', loss, prog_bar=True)
                self.log(f'{dataset_name}_val_acc',
                         sklearn.metrics.accuracy_score(labels, preds),
                         prog_bar=True)
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)
                self.log(f'{dataset_name}_val_auc', sklearn.metrics.auc(fpr, tpr), prog_bar=True)
                
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def test_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            preds = torch.cat([x['preds'] for x in output]).detach().cpu().numpy()
            labels = torch.cat([x['labels'] for x in output]).detach().cpu().numpy()
            scores = torch.cat([x['scores'] for x in output]).detach().cpu().numpy()
            dataset_name = {0:'pep', 1:'20q'}[i]
            self.log(f'{dataset_name}_test_acc',
                        sklearn.metrics.accuracy_score(labels, preds),
                        prog_bar=True)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)
            self.log(f'{dataset_name}_test_auc', sklearn.metrics.auc(fpr, tpr), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule"""
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]


@click.command()
@click.option('--model_name_or_path', default='roberta-base', help='Huggingface model name or path.')
@click.option('--data_dir',           default='./data',       help='Loads datasets from here.')
@click.option('--output_dir',         default='./output',     help='Saves models and logs here.')
@click.option('--cache_dir',          default='./cache',      help='Caches preprocessed datasets here.')
@click.option('--ckpt_path',          default=None,           help='Resume from a checkpoint.')
@click.option('--model_type',         default='roberta',      help='The model type to train or evaluate.',
              type=click.Choice(['roberta', 'conceptmax', 'conceptinject'], case_sensitive=False))
@click.option('--stage',              default='train',        help='Stage of lightning module.',
              type=click.Choice(['train', 'test'], case_sensitive=False))
def main(model_name_or_path, output_dir, data_dir, cache_dir, ckpt_path,
         model_type, stage):
    """Train a given model type using pytorch lightning."""

    pl.seed_everything(42)

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    data_module = DataModule(
                    model_type=model_type,
                    model_name_or_path=model_name_or_path,
                    cache_dir=cache_dir,
                    data_dir=data_dir)
    data_module.prepare_data()
    data_module.setup(stage)

    model_module = PlausibilityModule(
        model_type=model_type,
        model_name_or_path=model_name_or_path
    )
    
    checkpoint_callback = ModelCheckpoint(save_top_k=-1,
                                          every_n_epochs=1,
                                          save_last=True,
                                          save_on_train_epoch_end=True)

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator='auto',
        devices=1,
        precision=16,
        enable_checkpointing=True,
        val_check_interval=0.5,
        default_root_dir=output_dir,
        logger=pl.loggers.CSVLogger(output_dir),
        callbacks=[checkpoint_callback]
    )
    
    if stage == 'train':
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=ckpt_path
        )
        
    trainer.validate(
        model_module,
        datamodule=data_module,
        ckpt_path=ckpt_path
    )
    
    if stage == 'test':
        trainer.test(
            model_module,
            datamodule=data_module,
            ckpt_path=ckpt_path
        )


if __name__ == '__main__':
    main()
