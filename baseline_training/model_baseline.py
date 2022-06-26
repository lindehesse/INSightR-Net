""" 
11-05-2022 Linde S. Hesse

Baseline ResNet model
"""
import torch.nn as nn
from insight_training.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from define_parameters import ParamsBaseline


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features}

class BaseModel(nn.Module):

    def __init__(self, features,
                 network_params=ParamsBaseline(),
                 init_weights=True):

        super(BaseModel, self).__init__()
        self.params = network_params
        self.features = features

        if self.params.add_onlayers:
            self.fc = nn.Linear(128, network_params.num_classes)
        else:
            self.fc = nn.Linear(512, network_params.num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(
                i, nn.Conv2d)][-1].out_channels

        if self.params.add_onlayers:
            self.proto_shape = [50, 128, 1 , 1]
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels,
                        out_channels=self.proto_shape[1], kernel_size=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.proto_shape[1], out_channels=self.proto_shape[1], kernel_size=1),
                nn.Sigmoid()
            ) 

    def forward(self, x):
              
        x = self.features(x)
        if self.params.add_onlayers:
            x = self.add_on_layers(x) 

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

 

def construct_basemodel(network_params=ParamsBaseline(), pretrain_path = None):

    features = base_architecture_to_features[network_params.base_architecture](pre_path = pretrain_path)
 
    return BaseModel(features=features,
                 network_params=network_params)

