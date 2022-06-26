""" 
11-05-2022 Linde S. Hesse

File doing the prototype pushing

Prototype pushing is based on: https://github.com/cfchen-duke/ProtoPNet
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from helpers import plot_embeddings, plot_prototypes
from tqdm import tqdm
import matplotlib

from helpers import plot_rectangle
from helpers import find_high_activation_crop
from helpers import plot_heatmap

matplotlib.use('Agg')


class PushPrototypes:
    def __init__(self, pl_model):
        """ Initializes pushproto class to push the prototypes in latent space to closest image patch

        Args:
            pl_model (Lightning Model): Pytorch lightning model containing class attributes 'ppnet', 'params', 'current_epoch', and 'device'
        """

        # The network
        self.ppnet = pl_model.ppnet
        self.ppnet.eval()

        # parameters settings
        self.params = pl_model.params
        self.current_epoch = pl_model.current_epoch
        self.proto_shape = self.ppnet.proto_shape
        self.num_proto = self.ppnet.num_prototypes

        # device to push protos on
        self.device = pl_model.device

        # Make save folder
        self.proto_epoch_dir = self.params.save_path_ims / 'prototypes' / \
            f'epoch_{self.current_epoch}'
        self.proto_epoch_dir.mkdir(parents=True, exist_ok=True)

    def push_prototypes(self, dataloader):
        """ Pushes the model prototypes to closest image from dataloader

        Args:
            dataloader (iterator): iterator over data
        """
        # Save prototype visuzliation before protopushing
        for embed_type in ['pca', 'tsne']:
            for dim in ['2D', '3D']:
                embed = self.ppnet.prototype_vectors.detach()
                plot_embeddings(embed,
                                self.ppnet.proto_classes,
                                savepath=self.proto_epoch_dir /
                                f'emb_{embed_type}{dim}_beforepush.png',
                                embed_type=embed_type)

        # Saves the closest distance seen so far
        self.global_mindist = np.full(self.num_proto, np.inf)

        # saves the latent representation of prototype that gives the current smallest distance
        self.proto_latent_repr = np.zeros(self.proto_shape)

        # proto_information
        self.proto_bound_boxes = np.full(shape=[self.num_proto, 6], fill_value=-1)
        self.prototype_images = np.zeros(
            (self.num_proto, self.ppnet.img_size, self.ppnet.img_size, 3), dtype=np.uint8)
        self.prototype_actmaps = np.zeros(
            (self.num_proto, self.ppnet.output_size_conv, self.ppnet.output_size_conv))

        # Keep track of prototype classes
        self.proto_class_ID = self.ppnet.proto_classes.detach().clone()
        self.protonames = list(range(self.num_proto))

        # Iterate over data, and each time push protoypes to closest im
        for push_iter, (batch_im, batch_label, batch_names) in enumerate(tqdm(dataloader)):
            batch_im = batch_im.to(self.device)
            batch_label = batch_label.to(self.device)

            start_index = push_iter * dataloader.batch_size

            # Perform update step to update prototypes
            self.update_protos_batch(
                batch_im, batch_label, batch_names,start_index=start_index)

        # Set everything in model
        self.ppnet.prototype_vectors.data.copy_(
            torch.tensor(self.proto_latent_repr, dtype=torch.float32))        
        self.ppnet.prototype_images.data.copy_(
            torch.tensor(self.prototype_images))
        self.ppnet.prototype_actmaps.data.copy_(
            torch.tensor(self.prototype_actmaps))
      
        # Save the proto_class_ID and global min distances as txt to get an idea of prototype pushing
        data = np.column_stack([list(range(len(self.proto_class_ID))), self.ppnet.proto_classes.cpu().numpy(), self.global_mindist])
        datafile_path = Path(self.proto_epoch_dir) / 'Prototype_Information.txt'
        np.savetxt(datafile_path, data, fmt=['%10.4f','%10.4f', '%10.4f' ], header = 'ProtoNum    class   min-distance')

        # save filenames for prototypes for later visualization
        datafile_path = Path(self.proto_epoch_dir) / 'Prototype_Information_files.txt'
        with open(datafile_path, 'w') as filehandle:
            for listitem in self.protonames:
                filehandle.write('%s\n' % listitem)

        # Save the protype images as png
        if self.params.save_prototypes:
            for j in range(self.ppnet.prototype_images.shape[0]):
                # Bounding box information
                bb_info = self.proto_bound_boxes[j]
                target_class = bb_info[-1]

                # Class connections
                fc_weights = self.ppnet.last_layer.weight[:,j].detach().cpu().numpy()

                # Define savepath
                savepath = Path(self.proto_epoch_dir) / \
                    f'prototype_{j}_class_{target_class}.png'
                savepath.parents[0].mkdir(parents=True, exist_ok=True)

                # Upsampled activation map to image size
                upsampled_actmap = cv2.resize(self.prototype_actmaps[j],
                                              dsize=(self.prototype_images[j].shape[1],
                                                     self.prototype_images[j].shape[0]),
                                              interpolation=cv2.INTER_CUBIC)

                # Crop prototype
                proto_crop = self.prototype_images[j][int(bb_info[1]):int(
                    bb_info[2]), int(bb_info[3]):int(bb_info[4])]
                
                # Plot prototypes and swap BGR to RGB
                plot_prototypes(self.prototype_images[j][:, :, ::-1]/255,
                                upsampled_actmap, proto_crop[:, :, ::-1]/255,  bb_info, fc_weights = fc_weights, savepath=savepath)

        # Save prototype visuzliation after protopushing
        for embed_type in ['pca', 'tsne']:
            embed = self.ppnet.prototype_vectors.detach()
            plot_embeddings(embed,
                        self.ppnet.proto_classes,
                        savepath=self.proto_epoch_dir /
                        f'emb_{embed_type}_afterpush.png',
                        embed_type=embed_type)

    def update_protos_batch(self, batch_im, batch_label, batch_names,
                            start_index=0 ):
        """ Update all current prototypes with batch of data

        Args:
            batch_im (array): images
            batch_label (array): corresponding image labels
            start_index (int, optional): start index of the batch in the whole dataset. Defaults to 0.
          
        """

        # Put batch through trained network
        with torch.no_grad():
            batch_protodist, batch_convfeatures = self.ppnet.prototype_distances(batch_im)

        # Get them to cpu
        batch_convfeatures = batch_convfeatures.detach().cpu().numpy()
        batch_protodist = batch_protodist.detach().cpu().numpy()

        # Iterate over all current prototypes
        for proto_j in range(self.num_proto):

            # obtain distances to current prototype
            proto_dist_j = batch_protodist[:, proto_j] 

            # Get minimum distance to prototype j in current batch
            batch_mindist_j = np.amin(proto_dist_j)

            # If current batch has ims with smaller distance
            if batch_mindist_j < self.global_mindist[proto_j]:
        
                # get index of min distance (unravel to get it in array format)
                flat_index = np.argmin(proto_dist_j, axis=None)
                index_mindist = list(np.unravel_index(
                    flat_index, proto_dist_j.shape))
                assert proto_dist_j[index_mindist[0], index_mindist[1], index_mindist[2]] == batch_mindist_j

                # get class of imag eprototype is pushed to
                proto_class = batch_label[index_mindist[0]].item()

                # Extract corresponding latent patch
                latent_patch_mindist = batch_convfeatures[index_mindist[0],
                                                          :,
                                                          index_mindist[1],
                                                          index_mindist[2]]

                # Add retrieved proto distance to global variables
                self.global_mindist[proto_j] = batch_mindist_j
                self.proto_latent_repr[proto_j] = np.expand_dims(latent_patch_mindist,(1,2))

                # Get the whole corresponding input image of prototype
                orig_img_j = batch_im[index_mindist[0]].cpu().numpy()
                orig_img_j = np.transpose(orig_img_j, (2, 1, 0)) 

                # Find the highly activated path in original image 
                proto_dist_img_j = batch_protodist[index_mindist[0], proto_j]
                proto_act_img_j = self.ppnet.distance_2_similarity(proto_dist_img_j)

                # Upsample activation region to original image input size
                upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(orig_img_j.shape[1], orig_img_j.shape[0]),
                                                 interpolation=cv2.INTER_CUBIC)
                proto_bound_j = find_high_activation_crop(upsampled_act_img_j)

                # save the prototype boundary (rectangular boundary of highly act region)
                global_bb_boxes = [index_mindist[0]+start_index,*proto_bound_j, proto_class]
                self.proto_bound_boxes[proto_j] = np.array(global_bb_boxes, dtype=np.uint)

                # Save pushed prototypes info also to ppnet model so that these can be used for visualization
                self.prototype_images[proto_j] = (255 * orig_img_j).astype(np.uint8)
                self.prototype_actmaps[proto_j] = proto_act_img_j
               
                self.protonames[proto_j] = batch_names[index_mindist[0]]
       