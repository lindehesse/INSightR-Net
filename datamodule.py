""" 
11-05-2022 Linde S. Hesse

File containing the pytorch lightning datamodule
"""
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch
import logging
from pytorch_lightning.utilities import rank_zero_info
import numpy as np
from helpers import load_json

from dataset_DiabeticRet import DiabeticRet

txt_logger = logging.getLogger("pytorch_lightning")


class MyDataModuleDiabRet(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        files = list(self.params.train_dir.glob('*.jpeg'))
        files = sorted(files)

        # Load datadict
        datadict_train = load_json(params.datasplittrain_path)

        # extract corret train files -------------------------------------------------
        train_names = datadict_train[f'Fold {params.cv_fold}']['train']['files']

        if self.params.labels == 'classes':
            train_labels = datadict_train[f'Fold {params.cv_fold}']['train']['labels']
        elif self.params.labels == 'fussy':
            train_labels = datadict_train[f'Fold {params.cv_fold}']['train']['fussed_labels']

        # only keep files that exists
        self.train_files = []
        self.train_labels = []
        for name, lab in zip(train_names, train_labels):
            if (params.train_dir / (name + '.jpeg')).is_file():
                self.train_files.append(params.train_dir / (name + '.jpeg'))
                self.train_labels.append(lab)

        if len(self.train_files) != len(train_names):
            print(
                f'There are {len(train_names) - len(self.train_files)} files not found for training')

        # extract correct val files -------------------------------------------------------
        val_names = datadict_train[f'Fold {params.cv_fold}']['val']['files']

        if self.params.labels == 'classes':
            val_labels = datadict_train[f'Fold {params.cv_fold}']['val']['labels']
        elif self.params.labels == 'fussy':
            val_labels = datadict_train[f'Fold {params.cv_fold}']['val']['fussed_labels']

        # only keep files that exists
        self.val_files = []
        self.val_labels = []
        for name, lab in zip(val_names, val_labels):
            if (params.train_dir / (name + '.jpeg')).is_file():
                self.val_files.append(params.train_dir / (name + '.jpeg'))
                self.val_labels.append(lab)

        if len(self.val_files) != len(val_names):
            print(
                f'There are {len(val_names) - len(self.val_files)} files not found for training')

        # get test files ------------------------------------------------------
        datadict_test = load_json(params.datasplittest_path)
        test_names = datadict_test['files']
        test_labels = datadict_test['labels']

        # only keep files that exists
        self.test_files = []
        self.test_labels = []
        for name, lab in zip(test_names, test_labels):
            if (params.test_dir / (name + '.jpeg')).is_file():
                self.test_files.append(params.test_dir / (name + '.jpeg'))
                self.test_labels.append(lab)

        if len(self.test_files) != len(test_names):
            print(
                f'There are {len(test_names) - len(self.test_files)} files not found for testing')

    def train_dataloader(self):
        """ Dataset required for training

        Returns:
            pytorch dataloader
        """
        augm_transform = transforms.RandomAffine(
            degrees=(0, 360), scale=(0.9, 1.1), fillcolor=None)
        train_dataset = DiabeticRet(self.train_files, self.train_labels,
                                    preload=self.params.preload, transform=augm_transform)

        rank_zero_info(f'Training dataset size: {len(train_dataset)}')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.params.train_batch_size,
                                                   shuffle=True, num_workers=12,
                                                   pin_memory=True, drop_last=True)

        return train_loader

    def push_dataloader(self):
        """ Dataset required for prototype pushing

        Returns:
            pytorch dataloader
        """
        push_dataset = DiabeticRet(self.train_files, self.train_labels, preload=self.params.preload,)
        rank_zero_info(f'Pushing dataset size: {len(push_dataset)}')
        push_loader = torch.utils.data.DataLoader(push_dataset,
                                                  batch_size=self.params.train_batch_size,
                                                  shuffle=True, num_workers=12,
                                                  pin_memory=True, drop_last=True)
        return push_loader

    def val_dataloader(self):
        """ Dataset required for validation

        Returns:
            pytorch dataloader
        """
        val_dataset = DiabeticRet(self.val_files, self.val_labels, preload=self.params.preload,)
        rank_zero_info(f'Validation dataset size: {len(val_dataset)}')
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.params.val_batch_size,
                                                 shuffle=False, num_workers=12,
                                                 pin_memory=True, drop_last=True)

        return val_loader

    def test_dataloader(self):
        """ Dataset required for testing

        Returns:
            pytorch dataloader
        """
        test_dataset = DiabeticRet(self.test_files, self.test_labels, preload=self.params.preload,)
        rank_zero_info(f'Validation dataset size: {len(test_dataset)}')
        val_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=self.params.val_batch_size,
                                                 shuffle=False, num_workers=12,
                                                 pin_memory=True, drop_last=False)

        return val_loader
