""" 
11-05-2022 Linde S. Hesse

Pytorch Lightning model to train InsightR-Net model
"""
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
from dataclasses import asdict
from pathlib import Path
from skimage.transform import resize
from pytorch_lightning.utilities import rank_zero_info
import math
import logging
import time
import numpy as np
import torchmetrics
from collections import Counter
import shutil
import matplotlib.pyplot as plt

from helpers import log_batch_ims
from helpers import make_prediction_grid
from helpers import plot_confmatrix
from helpers import plot_prediction
from helpers import summary_string
from helpers import MySparsity

from insight_training.push_prototypes import plot_embeddings
from insight_training.loss_function import CombinedProtoLoss_Regr
from insight_training.push_prototypes import PushPrototypes
from insight_training import model

txt_logger = logging.getLogger("pytorch_lightning")

class LitModelProto(pl.LightningModule):
    def __init__(self, params, dataloader_push):
        super().__init__()
        self.params = params

        # Initialize network
        self.ppnet = model.construct_PPNet(
            network_params=params.network_params, pretrain_path=params.pretrained_path)

        # Set loss
        self.protoloss = CombinedProtoLoss_Regr(
            self.ppnet, self.params.loss_coefs)


        self.dataloader_push = dataloader_push

        # Set to false in order to work with multiple optimizers
        self.automatic_optimization = False

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

        # Initialize metrics to quantify sparsity as percentage weights needed for explanation
        self.val_sparsity_80 = MySparsity(level=0.8)
        self.test_sparsity_80 = MySparsity(level=0.8)
        self.train_sparsity_80 = MySparsity(level=0.8)

        # Init training stage parameter
        self.training_stage = 0

        self.save_hyperparameters("params")

    def forward(self, batch, num_proto=5, save_predictions=False, savenames=None):
        """ Forward a batch through the network and visualizes the predictions
        Args:
            batch (tuple): batch of data containing ims and labels
            num_proto (int, optional): Number of prototypes to be visualized. Defaults to 5.
            save_predictions (bool, optional): Whether to save predictions. Defaults to False.
            savenames ([str], optional): List of savenames . Defaults to None.

        Returns:
            tensor: predicted labels
        """

        # Forward batch through network
        batch_ims = batch[0].to(self.device)
        labels = batch[1].to(self.device)
        logits, distances, _, protoact, activation = self.ppnet.forward_testtime(
            batch_ims)

        preds = torch.clamp(logits.squeeze().detach(),
                            min=self.params.min_label, max=self.params.max_label)

        # Compute distances 
        distances = distances.detach().cpu().numpy()
        reshaped = distances.reshape(
            distances.shape[0], distances.shape[1], -1)

  
        min_indices_flat = np.argmin(reshaped, axis=-1)  # 30 x 50

        if save_predictions:
            # Generate savenames if not given
            if savenames == None:
                savenames = [f'im {i}' for i in range(len(batch_ims))]

            # Get last layer weihgts
            last_layer_conns = self.ppnet.last_layer.weight.square()

            # Iterate over images in batch with multiprocessing
            figsize = (20, 15)
            ax_info = make_prediction_grid(n_protos=5, figsize=figsize)

            for image_n in range(batch_ims.shape[0]):

                # Get true label and prediction
                true_label = labels[image_n].item()

                class_n = 0
                pred_label = preds[image_n]

                # Take last_layer weights for that class = last_layer[n]
                class_IDx = torch.unsqueeze(
                    self.ppnet.proto_classes, dim=0)
                ll_noclass = last_layer_conns[class_n] / class_IDx

                meanaverage_weights = torch.squeeze(
                    ll_noclass * protoact[image_n])
                _, indices = torch.topk(
                    torch.abs(meanaverage_weights), k=num_proto)

                weights_selected = meanaverage_weights[indices]
                ll_noclass_selected = torch.squeeze(ll_noclass)[
                    indices]
                simil_selected = protoact[image_n][indices]
                class_IDx_selected = torch.squeeze(class_IDx)[indices]

                # code to go from indices to prototypes
                proto_ims = self.ppnet.prototype_images[indices].cpu(
                ).numpy()[:, :, :, ::-1]/255  # 5 x 540 x 540 x 3#
                proto_activ = self.ppnet.prototype_actmaps[indices].cpu(
                ).numpy() / 255  # 5x17x17
                proto_activ_up = resize(proto_activ, (proto_activ.shape[0], proto_ims.shape[1
                                                                                            ], proto_ims.shape[2]))  # 5 x 540 x 540

                # Prepare for plotting
                im_npy = batch_ims[image_n].permute(2, 1, 0).cpu().numpy()[
                    :, :, ::-1]  # BGR to RGB
                single_activation = activation[image_n, indices].detach(
                ).cpu().numpy()
                im_activ = resize(single_activation, (single_activation.shape[0], im_npy.shape[0
                                                                                                ], im_npy.shape[1]))

                # Plot activations and prototypes
                percentage_weights = torch.sum(
                    ll_noclass_selected * simil_selected) / torch.sum(meanaverage_weights) * 100
                dict_titleinfo = {'true_label': true_label, 'pred_label': pred_label.item(), 'weights': weights_selected, 'll_noclass': ll_noclass_selected, 'similarities': simil_selected,
                                    'proto_labels': class_IDx_selected, 'studied_class': class_n, 'logits': logits[image_n], 'percentage': percentage_weights, 'style': 'regr'}
            
                fig_, _, _ = plot_prediction(ax_info, im_npy, im_activ, proto_ims, proto_activ_up, figsize=(
                    20, 15), title_info=dict_titleinfo)
                plt.savefig(
                    f'{savenames[image_n]}_class{class_n}' + '.png', bbox_inches=0)

            plt.close(ax_info[0])
        return preds

    def on_train_start(self):
        """ Log all hyperparameters and make snapshot of code
        """

        self.logger.log_hyperparams(asdict(self.params))
 

        # Log numeric correspondence for training stages
        self.logger.log_hyperparams({
            'training_stage_warm': 0,
            'training_stage_joint': 1,
            'training_stage_convexlast': 2,
            'training_stage_proto_push': 3})

        # Save all current code
        code_files = list(Path.cwd().glob('*.py'))
        for file in code_files:
            self.logger.experiment.log_artifact(
                self.logger.run_id, local_path=file, artifact_path='code_snapshot')

        # saves the network architecture used to a .txt file
        input_size = (3, self.params.network_params.img_size, self.params.network_params.img_size)
        string = summary_string(self.ppnet, input_size)
        with open(os.path.join(self.params.save_path, 'network_summary.txt'), 'w') as f:
            f.write(string)

        self.logger.experiment.log_artifact(
            self.logger.run_id, local_path=os.path.join(self.params.save_path,
                                                        'network_summary.txt'),
            artifact_path='network_summary')

        # remove prevous cf matrices
        if self.trainer.is_global_zero:
            if (self.params.save_path_ims / 'conf_matrices').is_dir():
                shutil.rmtree(self.params.save_path_ims / 'conf_matrices')

    def on_train_epoch_start(self):
        """ Dependent on epoch number, select training stage and initialization accordingly
        """

        # Use epoch number to derive training stage
        # In first cycle the number of epochs before push is determined by push_start,
        # not by push_frequency
        if self.current_epoch < (self.params.push_start + self.params.ll_iteration):
            self.epoch_state = self.current_epoch - \
                (self.params.push_start - self.params.push_frequency)
        else:
            self.epoch_state = self.current_epoch % (
                self.params.push_frequency + self.params.ll_iteration)

        # Log everytime with time
        local_time = time.ctime(time.time())

        # Warming up stage
        if self.current_epoch < self.params.num_warm_epochs:
            self.training_stage = 0
            rank_zero_info(
                local_time + f' Start training epoch {self.current_epoch}: warming up epoch')
            self.init_warmonly()

        #  Joint training stage
        elif self.epoch_state <= self.params.push_frequency:
            self.training_stage = 1
            rank_zero_info(
                local_time + f' Start training epoch {self.current_epoch}: joint learning epoch')
            self.init_joint()

        #  Last layer convex optimization only
        else:
            self.training_stage = 2
            rank_zero_info(
                local_time + f' Start training epoch {self.current_epoch}: convex optimization epoch')
            self.init_lastonly()

        # save model at start
        if self.epoch_state == 0 and self.current_epoch > self.params.push_start:
            self.save_model('StartStage0')

        # Log training stage for easy overview
        self.logger.experiment.log_metric(self.logger.run_id,
                                          key='training_stage',
                                          value=self.training_stage,
                                          step=self.current_epoch)
    
    def predict_step(self, batch, batch_idx):
        """ Defines step to predict for new images

        Args:
            batch (tuple): batch of data 

        Returns:
            predictions
        """
        input = batch[0]
        label = batch[1]
    
        output, _, prototype_activations, convs = self.ppnet(input, return_convs=True, k = 5)

        return label, output.squeeze(), prototype_activations, convs
    
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
        if step == 'val':
            output, min_distances, prototype_activations, convs_features = self.ppnet(input, return_convs=True)
            convs_features = convs_features.detach()
        else:
            output, min_distances, prototype_activations = self.ppnet(input, return_convs=False)
            convs_features = torch.zeros((1, 1))  # dummy

        losses_dict = self.protoloss(
            output, min_distances,label, self.ppnet)

        # Convert regression pred into classes for accuracy computation
        output_classes = torch.clamp(torch.round(output.squeeze()).type(
            torch.long), min=self.params.min_label, max=self.params.max_label) - 1
        label_class = label.round().squeeze().type(torch.long) - 1

        # Compute Confusion matrices only for val and test set
        if step == 'val' or step == 'test':
            getattr(self, f'{step}_confmatrix')(output_classes, label_class)

        # compute sparsity of last layer
        getattr(self, f'{step}_sparsity_80')(prototype_activations)

        return {**losses_dict, 'pred': output.detach(), 'pred_classes': output_classes.detach(), 'target': label_class.detach(), 'convs_features': convs_features}

    def training_step(self, batch, batch_idx):
        """ Defines the training step for a single batch
        Args:
            batch (tensor): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """

        # extract optimizers and lr schedulers
        warm, joint, last = self.optimizers()
        lr_sched = self.lr_schedulers()

        # Do forward step
        losses_dict = self.forward_step(batch, step='train')
        loss = losses_dict['total_loss']

        # Log here per step (do not log pred and target)
        temp_dict = dict(losses_dict)
        temp_dict.pop('pred')
        temp_dict.pop('target')
        temp_dict.pop('pred_classes')
        temp_dict.pop('convs_features')
        self.log_dict({f'train_step_{k}': v for k, v in temp_dict.items()})

        # Warming up stage
        if self.training_stage == 0:
            warm.zero_grad()
            self.manual_backward(loss)
            warm.step()

        #  Joint training stage
        elif self.training_stage == 1:
            joint.zero_grad()
            self.manual_backward(loss)
            joint.step()

            # Do lr step once per epoch
            if self.trainer.is_last_batch :
                if self.params.optim_params.joint_lr_decay:
                    lr_sched.step()
            # Hardcode so that actual lr steps are performed
            # if batch_idx == 8:

        # Do convex optimization of last layer
        elif self.training_stage == 2:
            last.zero_grad()
            self.manual_backward(loss)
            last.step()

        self.trainer.fit_loop.running_loss.append(loss)

        # Log the input images for epoch 0 to ensure inputs are correct
        if self.current_epoch == 0 and batch_idx < 3:
            self.savedir_input = self.params.save_path_ims / 'Input_Epoch0'
            log_batch_ims(batch[0], batch[1], batch_idx,
                          self.savedir_input, max_save=10)

        return {f'{k}': v.detach() for k, v in losses_dict.items()}

    def validation_step(self, batch, batch_idx):
        """ Defines the validation step for a single batch
        Args:
            batch (tensor): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """
        return self.forward_step(batch, step='val')

    def test_step(self, batch, batch_idx):
        """ Defines the test step for a single batch
        Args:
            batch (tensor): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """
        return self.forward_step(batch, step='test')

    def training_step_end(self, outputs):
        """ Logging for trainer step

        Args:
            outputs (dict): dict with keys 'pred' and 'target'
        """
        self.train_accuracy(outputs['pred_classes'], outputs['target'])
        self.train_kappa(outputs['pred_classes'], outputs['target'])

    def validation_step_end(self, outputs):
        """ Logging for validation step

        Args:
            outputs (dict): dict with keys 'pred' and 'target'
        """
        self.val_accuracy(outputs['pred_classes'], outputs['target'])
        self.val_kappa(outputs['pred_classes'], outputs['target'])

    def test_step_end(self, outputs):
        """ Logging for testing step

        Args:
            outputs (dict): dict with keys 'pred' and 'target'
        """
        self.test_accuracy(outputs['pred_classes'], outputs['target'])
        self.test_kappa(outputs['pred_classes'], outputs['target'])

    def log_epoch_end(self, output_losses, tag='train'):
        """ Logs all the losses at end of epoch

        Args:
            output_losses (dict): losses returned by train step
            tag (str, optional): Whether we are logging from a train or validation epoch. Defaults to 'train'.
        """
        # Log lr per epoch
        if tag == 'train':
            lr_sched = self.lr_schedulers()
            lrates = lr_sched.get_last_lr()
            for i, lr in enumerate(lrates):
                self.logger.experiment.log_metric(
                    self.logger.run_id, key=f'lr_epoch_{i}', value=lr, step=self.current_epoch)

        # Log the metrics at end of epoch
        for metric in ['accuracy', 'kappa', 'sparsity_80']:
            metric_function = getattr(self, f'{tag}_{metric}')
            metric_value = metric_function.compute()
            metric_function.reset()
            self.logger.experiment.log_metric(self.logger.run_id,
                                              key=f'{tag}_epoch_{metric}',
                                              value=metric_value.item(),
                                              step=self.current_epoch)

            print(f'{metric}: {metric_value}')

        # Sum total loss over all steps
        total_counters = Counter()
        for x in output_losses:
            x.pop('pred')
            x.pop('target')
            x.pop('pred_classes')
            x.pop('convs_features')
            total_counters.update(Counter(x))

        for key in total_counters:
            self.logger.experiment.log_metric(self.logger.run_id,
                                              key=f'{tag}_epoch_{key}',
                                              value=total_counters[key].item(
                                              ) / len(output_losses),
                                              step=self.current_epoch)

        # Log to monitoring loss
        self.average_loss = total_counters['total_loss'] / len(output_losses)
        self.log(f'{tag}_loss_monitor', self.average_loss, sync_dist=True)

        # Loss best loss in stage 2 (log as inf if not in stage 2)
        if self.training_stage == 2:
            self.log(f'{tag}_loss_stage2', self.average_loss, sync_dist=True)
        else:
            self.log(f'{tag}_loss_stage2', math.inf, sync_dist=True)

        # Log to txt logger
        local_time = time.ctime(time.time())
        rank_zero_info(
            local_time + f' Total {tag} loss: {self.average_loss:.4f}')

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
        # Needs to be done before log_epoch end to prevent deleting
        conv_list = []
        label_list = []
        for dictstep in validation_steps_output:
            conv_list.append(dictstep['convs_features'])
            label_list.append(dictstep['target'])
        embedding_numpy = torch.vstack(conv_list).cpu().numpy()
        embedded_labels = torch.hstack(label_list).cpu().numpy()

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

        # Log txt logger to mlflow
        self.logger.experiment.log_artifact(
            self.logger.run_id, local_path=self.params.save_path / 'logfile.log')

        # Plot the vector space embeddings
        savepath_emb = self.params.save_path_ims / 'embeddings'
        savepath_emb.mkdir(parents=True, exist_ok=True)

        for embed_type in ['pca', 'tsne']:
            for dim in ['2D', '3D']:
                embed = self.ppnet.prototype_vectors.detach()
                plot_embeddings(embed,
                                self.ppnet.proto_classes,
                                savepath=savepath_emb /
                                f'emb_{embed_type}{dim}_epoch_{self.current_epoch}.png',
                                embed_type=embed_type,
                                dim=dim,
                                sample_points=embedding_numpy,
                                sample_labels=embedded_labels)

        # Log the resulting embeddings as artifact
        self.logger.experiment.log_artifacts(
            self.logger.run_id, local_dir=savepath_emb, artifact_path='embeddings')

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
            list, list: optimizers, lr_schedulers
        """

        optim_params = self.params.optim_params

        # Joint training optimizers and lr schedulers
        joint_optim_specs = \
            [{'params': self.ppnet.features.parameters(), 'lr': optim_params.joint_optim_lrs.features},
             {'params': self.ppnet.add_on_layers.parameters(
             ), 'lr': optim_params.joint_optim_lrs.add_on_layers, 'weight_decay': 1e-3},
                {'params': self.ppnet.prototype_vectors,
                 'lr': optim_params.joint_optim_lrs.prototype_vectors}
             ]

        joint_optim = torch.optim.Adam(joint_optim_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            joint_optim, step_size=optim_params.joint_lr_stepsize, gamma=0.5)

        # Warm training optimizers
        warm_optim_specs = \
            [{'params': self.ppnet.add_on_layers.parameters(), 'lr': optim_params.warm_optim_lrs.add_on_layers},
             {'params': self.ppnet.prototype_vectors,
                'lr': optim_params.warm_optim_lrs.prototype_vectors}
             ]
        warm_optim = torch.optim.Adam(warm_optim_specs)

        # Last layer training optimizers
        ll_optim_specs = [
            {'params': self.ppnet.last_layer.parameters(), 'lr': optim_params.ll_optim_lr}]
        ll_optim = torch.optim.Adam(ll_optim_specs)

        # Gather all optimizers together and return
        optimizers = [warm_optim, joint_optim, ll_optim]
        lr_schedulers = {"scheduler": joint_lr_scheduler,
                         'name': 'joint_scheduler'}

        return optimizers, lr_schedulers

    def on_train_epoch_end(self):
        """ Step performed after training for one epoch, this is push of prototypes"""

        # Log images after first training epoch
        if self.current_epoch == 0:
            self.logger.experiment.log_artifacts(
                self.logger.run_id, local_dir=self.savedir_input, artifact_path='input_epoch0')

        # At the end of the joint training stage, push prototype vectors to closest image patch
        if self.epoch_state == self.params.push_frequency:
            self.save_model('before_protopushing')
            self.logger.experiment.log_metric(self.logger.run_id,
                                              key='training_stage',
                                              value=3,
                                              step=self.current_epoch)
            # Push prototypes
            if self.global_rank == 0:
                txt_logger.info('Prototype_Pushing')
                pushprototypes = PushPrototypes(self)
                pushprototypes.push_prototypes(self.dataloader_push)

                # Log new prototypes
                self.logger.experiment.log_artifacts(self.logger.run_id,
                                                     local_dir=pushprototypes.proto_epoch_dir,
                                                     artifact_path=f'prototypes/epoch{self.current_epoch}')

                self.save_model('after_protopushing')

            # Syncronizes parameters between models  (not sure this is necessary)
            if self.trainer.gpus > 1:
                torch.cuda.synchronize()
                self.trainer.training_type_plugin.model._sync_params_and_buffers(
                    authoritative_rank=0)

    def init_warmonly(self):
        """ Set required gradients for feature layers to false
        """
        for p in self.ppnet.features.parameters():
            p.requires_grad = False
        for p in self.ppnet.add_on_layers.parameters():
            p.requires_grad = True
        for p in self.ppnet.last_layer.parameters():
            p.requires_grad = True

        self.ppnet.prototype_vectors.requires_grad = True

    def init_joint(self):
        """ Set required gradients for all layers
        """
        for p in self.ppnet.features.parameters():
            p.requires_grad = True
        for p in self.ppnet.add_on_layers.parameters():
            p.requires_grad = True
        for p in self.ppnet.last_layer.parameters():
            p.requires_grad = True

        self.ppnet.prototype_vectors.requires_grad = True

    def init_lastonly(self):
        """ Only require last layer gradients
        """
        for p in self.ppnet.features.parameters():
            p.requires_grad = False
        for p in self.ppnet.add_on_layers.parameters():
            p.requires_grad = False
        for p in self.ppnet.last_layer.parameters():
            p.requires_grad = True

        self.ppnet.prototype_vectors.requires_grad = False

    @rank_zero_only
    def save_model(self, savename):
        path = Path(self.params.save_path) / 'saved_models'
        path.mkdir(exist_ok=True, parents=True)
        torch.save(self.ppnet.state_dict(), path /
                   f'Epoch_{self.current_epoch}_{savename}.pth')
