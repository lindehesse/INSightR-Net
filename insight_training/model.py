""" 
11-05-2022 Linde S. Hesse

File defining the INSightR-Net model

Prototypical layer implementation is based on: https://github.com/cfchen-duke/ProtoPNet
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import unravel_index
from insight_training.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from define_parameters import NetworkParams

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features}


class PPNet(nn.Module):

    def __init__(self, features,
                 network_params=NetworkParams(),
                 init_weights=True):

        super(PPNet, self).__init__()
        self.img_size = network_params.img_size
        self.proto_shape = network_params.proto_shape
        self.num_prototypes = network_params.proto_shape[0]
        self.epsilon = network_params.epsilon
        self.output_size_conv = network_params.output_size_conv
        self.bias_ll = network_params.bias_ll
        self.init_ll = network_params.init_ll
        self.features = features
        self.max_dist = self.proto_shape[0] * self.proto_shape[1] * self.proto_shape[2]

        # number of output nodes
        self.output_num = 1

        # prototype_activation_function could be 'log' or 'exp'
        self.proto_activation = network_params.proto_activation

        # set up prototypes for regression like prototypes
        proto_classes = torch.linspace(network_params.proto_minrange, network_params.proto_maxrange, self.num_prototypes)
        self.register_buffer("proto_classes", proto_classes)

        # Determine number of channels of last feature layer ( = num input channels add layers)
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(
                i, nn.Conv2d)][-1].out_channels

        # make add_on_layers
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels,
                        out_channels=self.proto_shape[1], kernel_size=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.proto_shape[1], out_channels=self.proto_shape[1], kernel_size=1),
            nn.Sigmoid()
        )

        # prototype information
        self.prototype_vectors = nn.Parameter(torch.rand(self.proto_shape),
                                              requires_grad=True)

        # Save all proto info
        # This takes a bit of memory but makes final predictions easier to visualize
        self.register_buffer("prototype_images", torch.zeros(
            self.num_prototypes, self.img_size, self.img_size, 3, dtype=torch.uint8))
        self.register_buffer("prototype_actmaps", torch.zeros(
            self.num_prototypes, self.output_size_conv, self.output_size_conv))
        self.register_buffer("proto_bounding_boxes",
                             torch.zeros(self.num_prototypes, 6))
        
        # Required for l2 convolution
        self.ones = nn.Parameter(torch.ones(self.proto_shape),
                                 requires_grad=False)

        # Last layer applies squared weights so that they are always positive
        self.last_layer = LinearSquared(self.num_prototypes, self.output_num,
                                        bias=self.bias_ll)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x_sigm = self.add_on_layers(x)

        return x_sigm

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        implemented from: https://ieeexplore.ieee.org/abstract/document/8167877
        (eq. 17 with S =weights, M = prototypes, X = input)
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
               apply self.prototype_vectors as l2-convolution filters on input x
        implemented from: https://ieeexplore.ieee.org/abstract/document/8167877
        (eq. 17 with S =self.ones, M = prototypes, X = input)
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        activated_protos = self.prototype_vectors

        p2 = activated_protos ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=activated_protos)

        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        Computes l2 distance between output features and prototypes
        '''
        conv_features = self.conv_features(x) 
        distances = self._l2_convolution(conv_features)
        return distances, conv_features

    def distance_2_similarity(self, distances):
        """ converts distances to similarities (neg)

        Args:
            distances (tensor): L2 distances

        Returns:
            tensor: similarities
        """
        if self.proto_activation == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.proto_activation == 'linear':
            return -distances
        elif self.proto_activation == 'exp_norm':
            dist_max = self.proto_shape[1] * \
                self.proto_shape[2] * self.proto_shape[3]
            return 1 / ((distances / dist_max) + self.epsilon)
        else:
            return self.proto_activation(distances)

    def forward(self, x, return_convs=False, k = 1):
        distances, conf_features = self.prototype_distances(x)

        class_IDx = torch.unsqueeze(self.proto_classes, dim=0)

        # assert class IDx does not contain 0's
        assert class_IDx.nonzero(as_tuple = False).size()[0] == class_IDx.shape[1], "Network class IDx contains zero's which causes division by 0"

        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                        kernel_size=(distances.size()[2],
                                                    distances.size()[3]))
        if return_convs:              
            # get min k  distance to each prototype
            distances_res = distances.view(distances.size()[0],distances.size()[1], -1)
            mink_distances, indices = torch.topk(-distances_res, k = 1)
            mink , indices2 = torch.topk(mink_distances.squeeze(), k = k)

            closest_convfeatures = torch.zeros((conf_features.shape[0], conf_features.shape[1], k))

            for j in range(distances.shape[0]):
                im_index = indices2[j]
                for enum, index in enumerate(im_index):
                    ind = indices[j, index]
                    ind_unravel = unravel_index(ind, (9,9))

                    conv = conf_features[j, :, ind_unravel[0][0], ind_unravel[0][1]]
                    closest_convfeatures[j, :, enum ] = conv

            closest_convfeatures = closest_convfeatures.squeeze()

        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)

        # multiply by class labels (to get final weighted mean prediction) 
        ll_noclass = self.last_layer.weight.square() / class_IDx

        # Weights consist of everything except class_IDx
        sum_of_weights = torch.sum(
            prototype_activations * ll_noclass, dim=1)

        # Apply squared weights (implemented as squared layer)
        logits = self.last_layer(prototype_activations)

        # Final prediction is divided by sum of weights
        logits = logits / torch.unsqueeze(sum_of_weights, 1)

        # Compute prototype activation:
        if return_convs:
            return logits, min_distances, prototype_activations, closest_convfeatures
        else:
            return logits, min_distances, prototype_activations

    def forward_testtime(self, x):
        """ Forward function to be used during test time, also returns the activation maps of the matching 
            prototype and the image

        Args:
            x (tensor): batch of images (size of [batch_size, 3, im_size, im_size])
        """
        # get distances to each of the prototypes for each image in batch
        distances, _ = self.prototype_distances(
            x)  # [batch_size, num_proto, 17 , 17]
        img_activations = self.distance_2_similarity(distances)

        class_IDx = torch.unsqueeze(self.proto_classes, dim=0)

        # global min pooling
        # Get min distance of each image to each prototype ([batch_size, num_proto, 1, 1])
        min_distances = -F.max_pool2d(-distances,
                                        kernel_size=(distances.size()[2],
                                                    distances.size()[3]))
        # [batch_size, num_proto, 1,1]
        min_distances = min_distances.squeeze()

        # Convert to similirities and put through last FC layer
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(
            min_distances)

        # Last layer is a squared weight layer
        ll_noclass = self.last_layer.weight.square() / class_IDx

        # Weights consist of everything except class_IDx
        sum_of_weights = torch.sum(
            prototype_activations * ll_noclass, dim=1)

        # Apply squared weights (implemented as squared layer)
        logits = self.last_layer(prototype_activations)

        # Final prediction is divided by sum of weights
        logits = logits / torch.unsqueeze(sum_of_weights, 1)
     
        return logits, distances, min_distances, prototype_activations, img_activations

    def set_last_layer_connection(self):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''

        class_IDx = torch.unsqueeze(self.proto_classes, dim=0)

        # set weights to prototype class label
        if self.init_ll == 'class_idx':
            self.last_layer.weight = nn.Parameter(torch.sqrt(class_IDx))
        
        elif self.init_ll == 'ones':
            self.last_layer.weight = nn.Parameter(torch.ones_like(class_IDx))

    def _initialize_weights(self):
        """ Initializes all weights in the network
        """
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_connection()


class LinearSquared(nn.Linear):
    """ Class that applies squared weights to the input so that these are always positive
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return F.linear(input, self.weight.square(), self.bias)


def construct_PPNet(network_params=NetworkParams(), pretrain_path=None):

    features = base_architecture_to_features[network_params.base_architecture](
        pre_path = pretrain_path)
    
    return PPNet(features=features,
                 network_params=network_params)


