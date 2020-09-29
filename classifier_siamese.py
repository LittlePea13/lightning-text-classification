# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel
from torch.nn.functional import kl_div, softmax, log_softmax
import pytorch_lightning as pl
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from utils import mask_fill
from transformers import AutoTokenizer

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Classifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """
    
    class DataModule(pl.LightningDataModule):
        def __init__(self, classifier_instance):
            super().__init__()
            self.hparams = classifier_instance.hparams
            self.classifier = classifier_instance
            # Label Encoder
            self.label_encoder = LabelEncoder(
                pd.read_csv(self.hparams.train_csv).label.unique().tolist(), 
                reserved_labels=[]
            )
            self.label_encoder.unknown_index = None

        def read_csv(self, path: str) -> list:
            """ Reads a comma separated value file.

            :param path: path to a csv file.
            
            :return: List of records as dictionaries
            """
            df = pd.read_csv(path)
            df = df[["text_en", "text_de", "label"]]
            df["text_en"] = df["text_en"].astype(str)
            df["text_de"] = df["text_de"].astype(str)
            df["label"] = df["label"].astype(str)
            return df.to_dict("records")

        def train_dataloader(self) -> DataLoader:
            """ Function that loads the train set. """
            self._train_dataset = self.read_csv(self.hparams.train_csv)
            self._train_dataset_de = self.read_csv(self.hparams.train_de_csv)
            self.concat_dataset = ConcatDataset(
                self._train_dataset,
                self._train_dataset_de
            )
            return DataLoader(
                dataset=self.concat_dataset,
                sampler=RandomSampler(self.concat_dataset),
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        def val_dataloader(self) -> DataLoader:
            """ Function that loads the validation set. """
            self._dev_dataset = self.read_csv(self.hparams.dev_csv)
            return DataLoader(
                dataset=self._dev_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample_val,
                num_workers=self.hparams.loader_workers,
            )

        def test_dataloader(self) -> DataLoader:
            """ Function that loads the validation set. """
            self._test_dataset = self.read_csv(self.hparams.test_csv)
            return DataLoader(
                dataset=self._test_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample_test,
                num_workers=self.hparams.loader_workers,
            )

    def __init__(self, hparams: Namespace) -> None:
        super(Classifier, self).__init__()
        if type(hparams) == dict:
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        print(hparams)
        # Build Data module
        self.data = self.DataModule(self)
        
        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.bert_en = AutoModel.from_pretrained(
            self.hparams.encoder_model, output_hidden_states=True
        )
        self.bert_de = AutoModel.from_pretrained(
            self.hparams.encoder_model_de, output_hidden_states=True
        )
        # set the number of features our encoder model will return...
        if self.hparams.encoder_model == "google/bert_uncased_L-2_H-128_A-2":
            self.encoder_features = 128
        else:
            self.encoder_features = 768

        # Tokenizer
        self.tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")#Tokenizer("bert-base-uncased")
        self.tokenizer_de = AutoTokenizer.from_pretrained("bert-base-uncased")#Tokenizer("bert-base-uncased")

        # Classification head
        self.classification_head_en = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),
        )
        # Classification head
        self.classification_head_de = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),
        )
    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.CrossEntropyLoss()
        self._loss_uns = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert_en.parameters():
                param.requires_grad = True
            for param in self.bert_de.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.bert_en.parameters():
            param.requires_grad = False
        for param in self.bert_de.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample_test([sample], prepare_target=False)
            model_out = self.forward(model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    def forward(self, inputs_en, inputs_de):
        """ Usual pytorch forward function.
        :param inputs_en: text sequences [batch_size x src_seq_len]
        :param inputs_de: text sequences [batch_size x src_seq_len]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        #tokens = tokens[:, : lengths.max()]
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        #mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        if inputs_en is not None:
            inputs_en_sup, inputs_en_unsup = {}, {}
            for key in inputs_en:
                inputs_en_sup[key], inputs_en_unsup[key] = torch.chunk(inputs_en[key], 2)
            outputs_en = self.bert_en(**inputs_en_sup)
            logits_en = self.classification_head_en(outputs_en[0][:, 0, :])
            with torch.no_grad():
                outputs_en_unsup = self.bert_en(**inputs_en_unsup)
                logits_en_unsup = self.classification_head_en(outputs_en_unsup[0][:, 0, :])
            #logits_en = torch.cat([logits_en, logits_en_unsup])  
        else:
            logits_en = None
            logits_en_unsup = None
        if inputs_de is not None:
            outputs_de = self.bert_de(**inputs_de)
            logits_de = self.classification_head_de(outputs_de[0][:, 0, :])
        else:
            logits_de = None
        # Average Pooling
        # word_embeddings = mask_fill(
        #     0.0, tokens, word_embeddings, self.tokenizer.pad_token_id
        # )
        # sentemb = torch.sum(word_embeddings, 1)
        # sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        # sentemb = sentemb / sum_mask

        return {"logits_en": logits_en, "logits_en_unsup": logits_en_unsup, "logits_de": logits_de}

    def loss_task(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        predictions_sup_de, predictions_unsup_de = torch.chunk(predictions["logits_de"], 2)

        loss = self._loss(predictions["logits_en"], targets["labels_en"]) + self._loss(predictions_sup_de, targets["labels_en"])

        predictions_unsup_en = softmax(predictions["logits_en_unsup"], dim=1).detach()
        predictions_unsup_de = log_softmax(predictions_unsup_de, dim=1)
        assert len(predictions_unsup_en) == len(predictions_unsup_de)# == C.get()['batch_unsup']

        loss_kldiv = kl_div(predictions_unsup_de, predictions_unsup_en, reduction='none')    # loss for unsupervised
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        #assert len(loss_kldiv) == len(unlabel1)
        # loss += (epoch / 200. * C.get()['ratio_unsup']) * torch.mean(loss_kldiv)
        if self.hparams.ratio_mode == 'constant':
            loss += self.hparams['ratio_unsup'] * torch.mean(loss_kldiv)
        elif self.hparams['ratio_mode'] == 'gradual':
            loss += (self.current_epoch / float(self.hparams['max_epoch'])) * self.hparams['ratio_unsup'] * torch.mean(loss_kldiv)
        else:
            raise ValueError
        return loss
    
    def loss_unsup(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        _, predictions_unsup_de = torch.chunk(predictions["logits_de"], 2)
        predictions_unsup_en = softmax(predictions["logits_en_unsup"], dim=1).detach()
        predictions_unsup_de = log_softmax(predictions_unsup_de, dim=1)
        assert len(predictions_unsup_en) == len(predictions_unsup_de)# == C.get()['batch_unsup']

        loss_kldiv = kl_div(predictions_unsup_de, predictions_unsup_en, reduction='none')    # loss for unsupervised
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        #assert len(loss_kldiv) == len(unlabel1)
        # loss += (epoch / 200. * C.get()['ratio_unsup']) * torch.mean(loss_kldiv)
        if self.hparams.ratio_mode == 'constant':
            loss = self.hparams['ratio_unsup'] * torch.mean(loss_kldiv)
        elif self.hparams['ratio_mode'] == 'gradual':
            loss = (self.current_epoch / float(self.hparams['max_epoch'])) * self.hparams['ratio_unsup'] * torch.mean(loss_kldiv)
        else:
            raise ValueError
        return loss

    def loss_val(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """

        return self._loss(torch.cat([predictions["logits_en"], predictions["logits_en_unsup"]]), targets["labels"])

    def loss_test(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits_de"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample_en = [element[0] for element in sample]
        sample_de = [element[1] for element in sample]
        sample_en = collate_tensors(sample_en)
        sample_de = collate_tensors(sample_de)

        inputs_en = self.tokenizer_en(sample_en["text_en"] + sample_de["text_en"],
            return_tensors="pt", 
            padding=True, 
            #return_length=True, 
            return_token_type_ids=False, 
            return_attention_mask=True,
            truncation="only_first",
            max_length=512
        )
        inputs_en = dict(inputs_en)
        inputs_de = self.tokenizer_de(sample_en["text_de"] + sample_de["text_de"],
            return_tensors="pt", 
            padding=True, 
            #return_length=True, 
            return_token_type_ids=False, 
            return_attention_mask=True,
            truncation="only_first",
            max_length=512
        )
        inputs_de = dict(inputs_de)
        if not prepare_target:
            return inputs_en, inputs_de, {}

        # Prepare target:
        try:
            targets = {"labels_en": self.data.label_encoder.batch_encode(sample_en["label"])}
            return inputs_en, inputs_de, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")


    def prepare_sample_val(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)

        inputs_en = self.tokenizer_en(sample["text_en"],
            return_tensors="pt", 
            padding=True, 
            #return_length=True, 
            return_token_type_ids=False, 
            return_attention_mask=True,
            truncation="only_first",
            max_length=512
        )
        inputs_en = dict(inputs_en)

        if not prepare_target:
            return inputs_en, {}

        # Prepare target:
        try:
            targets = {"labels": self.data.label_encoder.batch_encode(sample["label"])}
            return inputs_en, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def prepare_sample_test(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)

        inputs_en = self.tokenizer_de(sample["text_de"],
            return_tensors="pt", 
            padding=True, 
            #return_length=True, 
            return_token_type_ids=False, 
            return_attention_mask=True,
            truncation="only_first",
            max_length=512
        )
        inputs_en = dict(inputs_en)

        if not prepare_target:
            return inputs_en, {}

        # Prepare target:
        try:
            targets = {"labels": self.data.label_encoder.batch_encode(sample["label"])}
            return inputs_en, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs_en, inputs_de, targets = batch
        model_out = self.forward(inputs_en, inputs_de)
        loss_task = self.loss_task(model_out, targets)
        loss_unsup = self.loss_unsup(model_out, targets)
        loss_val = loss_task + loss_unsup
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val, "loss_task": loss_task, "loss_unsup": loss_unsup}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs_en, targets = batch
        model_out = self.forward(inputs_en, None)
        loss_val = self.loss_val(model_out, targets)

        y = targets["labels"]
        y_hat = torch.cat([model_out["logits_en"], model_out["logits_en_unsup"]])

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc,})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(None, inputs)
        loss_val = self.loss_test(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits_de"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc,})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def test_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head_en.parameters()},
            {"params": self.classification_head_de.parameters()},
            {
                "params": self.bert_en.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
            {
                "params": self.bert_de.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()
    
    @classmethod
    def add_model_specific_args(
        cls, parser: ArgumentParser
    ) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: argparse.ArgumentParser

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_model",
            default="distilbert-base-uncased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--encoder_model_de",
            default="distilbert-base-german-cased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        parser.add_argument(
            "--train_csv",
            default="data/imdb_reviews_train_en_de.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--train_de_csv",
            default="data/filmstars_train_en_de_balanced.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/imdb_reviews_test_en_de.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/filmstars_test_en_de.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        parser.add_argument(
            "--ratio_mode",
            default="constant",
            type=str,
            help="Schedule for the unsupervised loss.",
        )
        parser.add_argument(
            "--ratio_unsup",
            default=5.0,
            type=float,
            help="Ratio for the unsupervised loss.",
        )
        return parser
