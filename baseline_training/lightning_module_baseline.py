
""" 
11-05-2022 Linde S. Hesse

Main pytorch lightning module to train baseline
"""
import os
import pytorch_lightning as pl
import torch
from dataclasses import asdict
from pathlib import Path
from helpers import log_batch_ims
import logging
import time
import torchmetrics
from torch import nn
import shutil
from torch.optim.lr_scheduler import StepLR
from helpers import plot_confmatrix, summary_string
from insight_training.loss_function import MSE_loss
import baseline_training.model_baseline as model


txt_logger = logging.getLogger("pytorch_lightning")


class LitModel_baseline(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Initialize network
        self.model = model.construct_basemodel(network_params=params.baseline_params,
                                               pretrain_path=params.pretrained_path)
        #
        #  Set losses
        if self.params.baseline_params.loss == 'CE':
            self.loss = nn.CrossEntropyLoss()
        elif self.params.baseline_params.loss == 'MSE':
            self.loss = MSE_loss()
        else:
            raise ValueError(f'The defined loss is not implemented')

        self.automatic_optimization = True

        # Set accuracy metric
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy(compute_on_step=False)
        self.test_accuracy = torchmetrics.Accuracy(compute_on_step=False)

        # Set kapp metrics
        self.train_kappa = torchmetrics.CohenKappa(5, weights='quadratic')
        self.val_kappa = torchmetrics.CohenKappa(
            5, weights='quadratic', compute_on_step=False)
        self.test_kappa = torchmetrics.CohenKappa(
            5, weights='quadratic', compute_on_step=False)

        # Set matrix for confusion matrix
        self.val_confmatrix = torchmetrics.ConfusionMatrix(5)
        self.test_confmatrix = torchmetrics.ConfusionMatrix(5)

        # Keep track of best loss
        self.best_loss = float("inf")

        # remove prevous cf matrices
        if (self.params.save_path_ims / 'conf_matrices').is_dir():
            shutil.rmtree(self.params.save_path_ims / 'conf_matrices')

    def on_train_start(self):
        """ Log all hyperparameters and make snapshot of code
        """

        self.logger.log_hyperparams(asdict(self.params))

        # Save all current code
        code_files = list(Path.cwd().glob('*.py'))
        for file in code_files:
            self.logger.experiment.log_artifact(
                self.logger.run_id, local_path=file, artifact_path='code_snapshot')

        # saves the network architecture used to a .txt file
        input_size = (3, self.params.network_params.img_size,
                      self.params.network_params.img_size)
        string = summary_string(self.model, input_size)
        with open(os.path.join(self.params.save_path, 'network_summary.txt'), 'w') as f:
            f.write(string)

        self.logger.experiment.log_artifact(
            self.logger.run_id, local_path=os.path.join(self.params.save_path,
                                                        'network_summary.txt'),
            artifact_path='network_summary')

    def forward(self, im):
        output = self.model(im)
        return output

    def forward_step(self, batch, step='train'):
        """ Defines forward step that can be used for validation and training

        Args:
            batch (tuple): tensor, labels

        Returns:
            dict: dictionary containing all losses
        """
        # Extract image and label from batch
        input = batch[0]
        label = batch[1]

        # Forward batch through network
        output = self.model(input)
        # squeeze necessary for MSE (goes from size [x,1] to [x])
        output = torch.squeeze(output)

        # regression style
        if output.shape == label.shape:
            loss = self.loss(output, label)
            pred = torch.clamp(torch.round(output).type(
                torch.long), min=1, max=self.params.baseline_params.max_pred) - 1
            label_class = torch.round(label).type(torch.long) - 1
        # Classification style
        else:
            label_class = torch.round(label).long() - 1
            loss = self.loss(output, label_class)
            pred = torch.argmax(output, dim=1)

        # Compute Confusion matrices only for val and test set
        if step == 'val' or step == 'test':
            getattr(self, f'{step}_confmatrix')(pred, label_class)

        return {'loss': loss, 'pred': pred.detach(), 'target': label_class.detach()}

    def training_step(self, batch, batch_idx):
        """ Defines the training step for a single batch
        Args:
            batch (tensor): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """
        losses_dict = self.forward_step(batch, step='train')

        # Log the input images for epoch 0 to ensure inputs are correct
        if self.current_epoch == 0 and batch_idx < 3:
            self.savedir_input = self.params.save_path_ims / 'Input_Epoch0'
            log_batch_ims(batch[0], batch[1], batch_idx,
                          self.savedir_input, max_save=10)

        return losses_dict

    def validation_step(self, batch, batch_idx):
        """ Defines the validation step for a single batch
        Args:
            batch (tensor): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """
        losses_dict = self.forward_step(batch, step='val')
        return losses_dict

    def test_step(self, batch, batch_idx):
        """ Defines the testing step for a single batch
        Args:
            batch (tensor): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """
        losses_dict = self.forward_step(batch, step='test')
        return losses_dict

    def predict_step(self, batch, batch_idx):
        return self(batch[0]).squeeze().detach().cpu(), batch[1].long().cpu()

    def training_step_end(self, outputs):
        """ Logging for trainer step

        Args:
            outputs (dict): dict with keys 'pred' and 'target'
        """
        self.train_accuracy(outputs['pred'], outputs['target'])

        # substract 1 due to predictions from 1 - 5 (and not 0 - 4)
        self.train_kappa(outputs['pred'], outputs['target'])

    def validation_step_end(self, outputs):
        """ Logging for validation step

        Args:
            outputs (dict): dict with keys 'pred' and 'target'
        """
        self.val_accuracy(outputs['pred'], outputs['target'])

        # substract 1 due to predictions from 1 - 5 (and not 0 - 4)
        self.val_kappa(outputs['pred'], outputs['target'])

    def test_step_end(self, outputs):
        """ Logging for testing step

        Args:
            outputs (dict): dict with keys 'pred' and 'target'
        """
        self.test_accuracy(outputs['pred'], outputs['target'])

        # substract 1 due to predictions from 1 - 5 (and not 0 - 4)
        self.test_kappa(outputs['pred'], outputs['target'])

    def log_epoch_end(self, output_losses, tag='train'):
        """ Logs all the losses at end of epoch

        Args:
            steps_output_dict (dict): losses returned by train step
            tag (str, optional): Whether we are logging from a train or validation epoch. Defaults to 'train'.
        """

        # Log the metrics at end of epoch
        for metric in ['accuracy', 'kappa']:
            metric_function = getattr(self, f'{tag}_{metric}')
            metric_value = metric_function.compute()
            metric_function.reset()

            self.logger.experiment.log_metric(self.logger.run_id,
                                              key=f'{tag}_epoch_{metric}',
                                              value=metric_value.item(),
                                              step=self.current_epoch)

        # Sum dicts over all steps
        total_loss = 0
        for x in output_losses:
            total_loss += x['loss'].item()

        # Log to txt logger
        average_loss = total_loss / len(output_losses)
        self.log(f'{tag}_loss_monitor', average_loss, sync_dist=True)

        # Log to txt logger
        local_time = time.ctime(time.time())
        if self.global_rank == 0:
            txt_logger.info(
                local_time + f' Total {tag} loss: {average_loss:.4f}')

    def training_epoch_end(self, training_steps_output):
        """ Function to collect losses from epoch and save this per epoch

        Args:
            training_steps_output (dict): dict containing all losses
        """
        self.log_epoch_end(training_steps_output, tag='train')

    def validation_epoch_end(self, validation_steps_output):
        """ Function to collect losses from epoch and save this per epoch

        Args:
            validation_steps_output (dict): dict containing all losses
        """
        self.log_epoch_end(validation_steps_output, tag='val')

        # Compute confusion matrix
        conf_matrix = getattr(
            self, f'val_confmatrix').compute().detach().cpu().numpy()
        getattr(self, f'val_confmatrix').reset()

        # Plot confusion matrix and asave as image
        savepath = self.params.save_path_ims / \
            'conf_matrices' / f'map_epoch_{self.current_epoch}'
        plot_confmatrix(conf_matrix, classes=[
                        '0', '1', '2', '3', '4'], savepath=savepath)

        # Log the resulting image as artifact
        self.logger.experiment.log_artifacts(
            self.logger.run_id, local_dir=savepath.parents[0], artifact_path='conf_matrices')

    def test_epoch_end(self, steps_output):
        """ Function to collect losses from testing epoch and save this per epoch

        Args:
            steps_output (dict): dict containing all losses
        """
        self.log_epoch_end(steps_output, tag='test')

        # Compute confusion matrix
        conf_matrix = getattr(
            self, f'test_confmatrix').compute().detach().cpu().numpy()
        getattr(self, f'test_confmatrix').reset()

        # Plot confusion matrix and asave as image
        savepath = self.params.save_path_ims / 'testing_confmatrix.png'
        plot_confmatrix(conf_matrix, classes=[
                        '0', '1', '2', '3', '4'], savepath=savepath)

        # Log the resulting image as artifact
        self.logger.experiment.log_artifact(
            self.logger.run_id, local_path=savepath)

    def configure_optimizers(self):
        """ Set up all optimizers for training

        Returns:
            dict: dict containing keys 'optimizer' and optionally 'lr_scheduler'
        """
        if self.params.baseline_params.optim == 'Adam':
            optim = torch.optim.Adam(
                self.model.parameters(), lr=self.params.baseline_params.single_lr)
        elif self.params.baseline_params.optim == 'AdamW':
            optim = torch.optim.AdamW(
                self.model.parameters(), lr=self.params.baseline_params.single_lr)

        if self.params.optim_params.joint_lr_decay:
            scheduler = StepLR(
                optim, step_size=self.params.optim_params.joint_lr_stepsize, gamma=0.5)
            return {'optimizer': optim, 'lr_scheduler': scheduler}
        else:
            return {'optimizer': optim}

    def on_train_epoch_end(self):
        """ Step performed after training for one epoch """

        # Log images as artifacts after first training epoch
        if self.current_epoch == 0:
            self.logger.experiment.log_artifacts(
                self.logger.run_id, local_dir=self.savedir_input, artifact_path='input_epoch0')
