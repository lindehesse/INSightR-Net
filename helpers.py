""" 
11-05-2022 Linde S. Hesse

File defining all helper functions
"""
import os
import torch
import numpy as np
import imageio
import logging.config
import matplotlib.patches as patches
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import cv2
import pickle 
from functools import wraps
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics import Metric
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


"""
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

"""
def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1



def log_batch_ims(input, label, batch_idx, savedir, max_save = 10):
    for num in range(max(max_save, input.shape[0])):
        savedir.mkdir(parents=True, exist_ok = True)
        filename = savedir / f'batch{batch_idx}_im_{num}_label_{label[num]}.png'

    #    im_int = img_as_float(input[num,:].cpu())
        norm_im = input[num].cpu().numpy() 
        im = ((norm_im - np.amin(norm_im)) / (np.amax(norm_im)-np.amin(norm_im)) * 255)

        im_RGB = np.transpose(im.astype(np.uint8), [1,2,0])
        imageio.imwrite(filename, im_RGB)

    
def plot_confmatrix(conf_matrix, classes, savepath):
    """ Plot confusion matrices and save the resulting plot as an image

    Args:
        conf_matrix (numpy array): confusion matrix 
        classes (list[str]): class names
        savepath (Path): path where the resulting matrix is saved as image
    """
     
    df_cfm = pd.DataFrame(conf_matrix, index = classes, columns=classes)

    plt.figure(figsize = (5,5))
    cfm_plot = sns.heatmap(df_cfm, annot = True, fmt = ".3f", robust = True)
    plt.xlabel("prediction")
    plt.ylabel("target")

    (savepath.parents[0]).mkdir(parents=True, exist_ok=True)
    cfm_plot.figure.savefig(savepath)
 
def format_axes(fig, axs):
    """ Function to remover certain axis from plot (need for make predition grid)

    Args:
        fig ([type]): [description]
        axs ([type]): [description]

    Returns:
        [type]: [description]
    """
    for i, ax in enumerate(fig.axes):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    rows, cols = axs.shape
    for col in [cols-1]:
        for row in range(rows):
            axs[row,col].axis('off')
    for row in [rows-1]:
        for col in range(cols):
            axs[row,col].axis('off')



class Cached(object):
    def __init__(self, filename):

        self.filename = filename

    def __call__(self, func):
        @wraps(func)
        def new_func(*args, **kwargs):
            if not os.path.exists(self.filename):
                results = func(*args, **kwargs)
                with open(self.filename, 'wb') as outfile:
                    pickle.dump(results, outfile, pickle.HIGHEST_PROTOCOL)
            else:
                with open(self.filename, 'rb') as infile:
                    results = pickle.load(infile)
            return results
        return new_func

@Cached('temp.pkl')
def make_prediction_grid(n_protos, figsize =(15,20) ):
    """ Makes grid to plot the prototypes and predictions on

    Args:
        n_protos (int): Number of prototypes to be plofted

    Returns:
        figure: figure grid
        axs: axs from original figure
        ax1: bigger ax for image
    """

    # Make the grid
    fig, axs = plt.subplots(ncols=7, nrows=n_protos+1, figsize =figsize)
    gs = axs[1,4].get_gridspec()

    # remove the underlying axes
    for col in [0,1]:
        for row in range(n_protos+1):
            axs[row, col].remove()
  
    # Add single axs at start   
    gs = GridSpec(7, 7, figure=fig)
    ax1 = fig.add_subplot(gs[2:4, :2])
    format_axes(fig, axs)
   
    # Remove first two columns in axs indexing
    axs_updated = axs[:,2:]   

    for bullet in [0.75, 0.80, 0.85]:
        axs_updated[5,4].text(0.84,bullet, r'$\bullet$')
    
    for num in range(4):
        for bullet in [0.75, 0.80, 0.85]:
            axs_updated[5,num].text(0.5,bullet, r'$\bullet$')

    return fig, axs_updated, ax1

def plot_rectangle(box_coords, edgecolor = 'r'):
    """ Makes a rectangular patch

    Args:
        box_coords ([type]): coordinates (equiv to cropping x[box_coords[0]:box_coords[1], box_coords[2]:box_coords[3]])
        edgecolor (str, optional): Color of line. Defaults to 'r'.

    Returns:
        [type]: patch
    """
    rect = patches.Rectangle((box_coords[2], box_coords[1]), box_coords[3]-box_coords[2],
                        -box_coords[1] + box_coords[0], linewidth=1, facecolor='none', edgecolor=edgecolor)
    return rect

def plot_prediction(ax_info, im_npy, im_activ, proto_ims, proto_activ,  figsize = (20,20), title_info = None):
    """ Plots the predictions with matchine prototypes

    Args:
        im_npy (array): Numpy array of size [xdim, ydim, 3]
        im_activ (array): Numpy array of image activations of size [n_proto, xdim, ydim]
        proto_ims (array): Prototype images of size [n_proto, xdim, ydim, 3]
        proto_activ (array): Prototype activations of size [n_proto, xdim, ydim]

    Returns:
        figure: figure with plotted images
    """

    fig, ax, main_ax = ax_info # make_prediction_grid(n_protos=5, figsize = figsize)
    
    # Remove any previous patches
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].patches = []
            ax[i,j].texts = []
            ax[i,j].images = []
    main_ax.texts=[]
    main_ax.image = []


    # Plot image itself
    main_ax.imshow(im_npy)

    for proto_n in range(proto_ims.shape[0]):
        # Plot image itself with bounding boxes
        heatmap = plot_heatmap(im_activ[proto_n])
        overlayed_heatmap = 0.5 * im_npy + 0.3 * heatmap
        ax[proto_n, 0].imshow(overlayed_heatmap)

        # Find highly activated crop in image
        crop = find_high_activation_crop(im_activ[proto_n])

        # Plot bounding box around highest activated region in test image
        rect = plot_rectangle(crop, edgecolor = 'b')
        ax[proto_n, 0].add_patch(rect)

        ax[proto_n,1].imshow(im_npy)
        rect = plot_rectangle(crop, edgecolor = 'b')
        ax[proto_n, 1].add_patch(rect)

        # Plot same for protytpe it is similar to
        ax[proto_n, 2].imshow(proto_ims[proto_n])

        heatmap = plot_heatmap(proto_activ[proto_n])
        heatmap_proto = 0.5 * proto_ims[proto_n] + 0.3 * heatmap
        ax[proto_n,3].imshow(heatmap_proto)

        # Plot bounding box
        proto_crop = find_high_activation_crop(
            proto_activ[proto_n])
        rect_proto = patches.Rectangle((proto_crop[2], proto_crop[1]), proto_crop[3]  -proto_crop[2],
                                -proto_crop[1] + proto_crop[0], linewidth=1, facecolor='none', edgecolor='b')
        rect2 = copy.copy(rect_proto)
        ax[proto_n,2].add_patch(rect_proto)
        ax[proto_n,3].add_patch(rect2)
    
    if title_info is not None:
        main_ax.set_title(f'True Label:{title_info["true_label"]}, Pred Label:{title_info["pred_label"]:.2f}')
        
        if title_info['style'] == 'regr':
            ax[0,4].set_title(f'  Simil x Proto_weight x class_ID = weight x Class_ID')
            for num in range(5):
                proto_label = title_info["proto_labels"][num].item()
                ax[num,2].set_title(f'Prototype class: {proto_label}')

                sim = title_info["similarities"][num]
                fc_weight = title_info['ll_noclass'][num]
        
                ax[num,4].text(0.5, 0.5, f'    {sim:.2f} x {fc_weight:.2f} x {proto_label:.1f} = {sim * fc_weight:.2f} x {proto_label:.1f}', ha="center", va="center", fontsize = 14)
              
            #textx = r'$\frac{\sum_{i = 0}^{i=nprotos} w_i * ProtoClass_i}{\sum_{i=0}^{i=nprotos} w_i}$'
            textx = f'Weighted Average Mean = Sum(weights * Proto_Class) / Sum(weights) = {title_info["pred_label"]:.2f}'
            ax[5,3].text(0.5,0.5 , textx, ha="center", va="center", fontsize = 14)

            texty = f'Plotted weights are {title_info["percentage"].item():.2f}% of total'
            ax[5,3].text(0.5,0.7, texty, ha="center", va="center", fontsize = 14)

  
    return fig, ax, main_ax

def load_json(json_path):
    """ Loads a json file
    Args:
        json_path (string): path of json
    Returns:
        [dict]: dictionary with content of json
    """
    with open(json_path) as f:
        dict = json.load(f)
    return dict

def plot_heatmap(activ_img):
    """ Makes a heatmap from an activation image

    Args:
        activ_img (array): Array of shape [x,y]

    Returns:
        [type]: [description]
    """

    # Rescale
    activ_img = activ_img - np.amin(activ_img)
    activ_img = activ_img / np.amax(activ_img)

    # Apply colormap
    heatmap = cv2.applyColorMap(
        np.uint8(255*activ_img), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    return heatmap


def preprocess_labels(csv_path, traintype = 'regr'):
    # Get df with all labels
    df = pd.read_csv(csv_path)
    df = df.fillna(-1)

    # use grade 1 - 5 instead of 0 - 4 to avoid problems with regression  
    df.level = df.level + 1

    # Make labels into dict
    label_df = pd.Series(df.level.values,
                            index=df.image)
    label_dict = label_df.to_dict()

    dict_bylabel = {}
    for lab in np.unique(label_df.values):
        dict_bylabel[lab] = label_df[label_df == lab].index.to_list()

    return label_dict, dict_bylabel

from torchmetrics import Metric

class MySparsity(Metric):
    def __init__(self, dist_sync_on_step=False, level = 0.9):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.level = level
        self.add_state('percentage_expl', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('total',default = torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prototype_activations: torch.Tensor):
        # Normalize by dividing by sum
        proto_norm = prototype_activations/ torch.sum(prototype_activations, dim=1).unsqueeze(-1)

        # sort and compute computitative sum
        sorted, indices = torch.sort(proto_norm, descending=True, dim = 1)
        cumsum = torch.cumsum(sorted, dim =1)
        num_weights = torch.ge(cumsum, self.level).type(torch.uint8).argmax(dim=1)

        # Gather results
        self.percentage_expl += torch.sum(num_weights)
        self.total += num_weights.numel()

    def compute(self):
        return self.percentage_expl.float() / self.total






def summary(model, input_size, batch_size=-1, device="cuda"):
    print(summary_string(model, input_size, batch_size, device))


def summary_string(model, input_size, batch_size=-1, device="cuda"):
    """Function to print the network architecture to a string

    Args:
        model (torch model): the model
        input_size (tuple): size of input images
        batch_size (int, optional): batch size. Defaults to -1.
        device (str, optional):  Defaults to "cuda".

    Returns:
        _type_: _description_
    """
    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) *
                           batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(
        total_params - trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"

    return summary_str



def unravel_index(
        indices,
        shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


def plot_prototypes(orig_img_j, upsampled_act_img_j, proto_img_j, proto_bound_j, fc_weights, savepath):
    """ Plot the updated protytpes and saved

    Args:
        orig_img_j (array): original input image
        upsampled_act_img_j (array): activation on original image
        proto_img_j (array): prototype patch from image
        proto_bound_j ([type]): prototype batch information (bounding box coords)
        savepath (str): path to save resulting plots to 
    """

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # plot original image
    ax[0].imshow(orig_img_j, vmin=0, vmax=1)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    # plot overlay of activation and original imge
    heatmap = plot_heatmap(upsampled_act_img_j)
    overlayed_original_img_j = 0.5 * orig_img_j + 0.3 * heatmap
    ax[1].imshow(overlayed_original_img_j,  vmin=0, vmax=1)
    ax[1].axis('off')
    ax[1].set_title('Activation Overlay')


    # show final prototype visualization as blue box
    rect = plot_rectangle(proto_bound_j[1:], edgecolor='b')
    ax[0].add_patch(rect)
    rect = plot_rectangle(proto_bound_j[1:], edgecolor='b')
    ax[1].add_patch(rect)

    legend_elms = [Patch(facecolor='none', edgecolor='b', label='Prototype Patch', linewidth=2)]

    ax[0].legend(legend_elms, [ 'Prototype Patch'],
                 bbox_to_anchor=(-0.05, 1), loc='upper right')

    #  plot proto image patch
    ax[2].imshow(proto_img_j, vmin=0, vmax=1)
    ax[2].axis('off')
    ax[2].set_title('Prototype Patch')

    # plot titles
    fc_string = '\n'.join([f'Class {i}: {fc:.3f} ' for i, fc in enumerate(fc_weights)])
    target_txt = f'Prototype Class: {proto_bound_j[-1]}'
    txt_string = '\n'.join([target_txt, fc_string])

    ax[0].text(-0.5, .5, txt_string, horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)

    fig.savefig(savepath)
    plt.close('all')


def plot_embeddings(prototypes, labels, savepath, embed_type='tsne', dim = '2D', sample_points=None, sample_labels=None):
    """ 
    Args:
        prototype_vectors ([type]): [description]
        labels ([type]): [description]
        savepath ([type]): [description]
        embed_type (str, optional): [description]. Defaults to 'tsne'.
        dim (str, optional): Whether to plot the embeddings in 2D ('2D') or 3D ('3D')
    """

    assert embed_type in ['tsne', 'pca']
    
    if sample_points is not None:
        if len(sample_points.shape) > 2:
            sample_labels = np.repeat(sample_labels, sample_points.shape[2])
            
            swap = np.swapaxes(sample_points, 1, 2)
            sample_points = swap.reshape(-1, swap.shape[2])
            
        all_points = np.vstack([torch.squeeze(prototypes).cpu().numpy(), sample_points])

    else:
        all_points = torch.squeeze(prototypes).cpu().numpy()

    if dim == '2D':
        n_components = 2
    elif dim == '3D':
        n_components = 3
    else:
        raise ValueError(f'Dim is not implemented for: {dim}')

    if embed_type == 'tsne':
        embed = TSNE(n_components=n_components, init='pca', learning_rate = 'auto').fit_transform(all_points)
    elif embed_type == 'pca':
        embed = PCA(n_components=n_components).fit_transform(all_points)

    # if labels are in one hot format convert them
    labels = labels.cpu().numpy()
    if len(labels.shape) != 1:
        labels = np.argmax(labels, axis=1)

    # make actual figure
    fig = plt.figure()
    num_protos = prototypes.shape[0]

    if dim == '3D': 
        ax = fig.add_subplot(projection='3d')
        embeds_protos = [embed[:num_protos, 0], embed[:num_protos, 1], embed[:num_protos,2]]
    else:
        ax = fig.add_subplot()
        embeds_protos = [embed[:num_protos, 0], embed[:num_protos, 1]]

    if sample_points is not None:
        if dim == '3D':
            embeds_points = [embed[num_protos:, 0], embed[num_protos:, 1], embed[num_protos:,2]]
        else:
            embeds_points = [embed[num_protos:, 0], embed[num_protos:, 1]]
        
        scatt = ax.scatter(*embeds_points, c=sample_labels, marker='*', alpha = 0.3)

    scatt = ax.scatter(*embeds_protos, c=labels, edgecolors = 'k', alpha =0.5)
    legend1 = ax.legend(*scatt.legend_elements(), title="Classes", loc = 'upper left')
    ax.add_artist(legend1)

    plt.savefig(savepath)
    plt.close('all')
