""" 
11-05-2022 Linde S. Hesse

File containing all loss functions used for training
"""
import torch
from torch import nn


class ClusterLoss_Regr(nn.Module):
    """ Class to compute the regression cluster loss 
    """
    def __init__(self, model, class_specific=True, delta=0.5, kmin=1):
        super(ClusterLoss_Regr, self).__init__()

        self.max_dist = model.proto_shape[1] * \
            model.proto_shape[2] * model.proto_shape[3]
        self.class_specific = class_specific
        self.delta = delta
        self.kmin = kmin

    def forward(self, min_distances, labels, model):
        # proto_classID can change after protopush
        proto_classID = model.proto_classes

        # pull closest prototype
        if not self.class_specific:
            inverted_distances, _ = torch.max(
                self.max_dist - min_distances, dim=1)
            cluster_cost = torch.mean(self.max_dist - inverted_distances)

        # pull closest prototype within distance
        else:
            # get prototypes within delta of image (based on label differenc)
            proto_correct_class = torch.stack(
                [torch.le(torch.abs(proto_classID.cuda() - x), self.delta) for x in labels])

            # Convert boolean to 0 and 1
            proto_correct_class = proto_correct_class.type_as(
                labels.floor().long())

            # Convert to distances
            inverted_distances, _ = torch.topk((self.max_dist - min_distances)
                                               * proto_correct_class, k=self.kmin, dim=1)

            cluster_cost = torch.mean(self.max_dist - inverted_distances)

        return cluster_cost


class ProtoSampleDist(nn.Module):
    """ Class to compute the prototype sample distance loss
    """
    def __init__(self,   model):
        super(ProtoSampleDist, self).__init__()

        self.max_dist = model.proto_shape[1] * \
            model.proto_shape[2] * model.proto_shape[3]

    def forward(self, min_distances, labels):

        # compute min distance from each prototype to a sample (to keep them close around)
        assert not torch.max(min_distances) > self.max_dist

        # dists is min_distance from each prototype to sample (irrespective of class)
        dists, _ = torch.min(min_distances, dim=0)
        dists_norm = dists / self.max_dist
        reg_cost = -torch.mean(torch.log(1-dists_norm))

        return reg_cost


class MSE_loss(nn.Module):
    """ MSE loss including type conversion to float

    """
    def __init__(self):
        super(MSE_loss, self).__init__()
        self.MSE = nn.MSELoss()

    def forward(self, output, labels):
        return self.MSE(output.squeeze(), labels.float())


class CombinedProtoLoss_Regr(nn.Module):
    def __init__(self, model, coefs):
        super(CombinedProtoLoss_Regr, self).__init__()

        self.coefs = coefs

        self.MSE_loss = MSE_loss()
        self.Clst_Regr = ClusterLoss_Regr(
            model, self.coefs.clst_class_specific,
            self.coefs.clst_delta, self.coefs.clst_kmin)

        self.PSD_loss = ProtoSampleDist(model)
        
  
    def forward(self, output, min_distances, labels, model):
        self.total_loss = 0
        self.losses_dict = {}

        # Only compute losses that are required (have non-zeros coefs)
        if self.coefs.mse != 0:
            self.update_MSE(output, labels)

        if self.coefs.PSD != 0:
            self.update_PSD(min_distances, labels)

        if self.coefs.clst_regr != 0:
            self.update_Clst_regr(min_distances, labels, model)

        # Add total loss
        self.losses_dict['total_loss'] = self.total_loss

        return self.losses_dict

    def update_MSE(self, output, labels):
        MSE = self.MSE_loss(output, labels)
        self.total_loss = self.total_loss + MSE * self.coefs.mse
        self.losses_dict['MSE'] = MSE

    def update_Clst_regr(self, output, labels, model):
        Clst_regr = self.Clst_Regr(output, labels, model)
        self.total_loss = self.total_loss + Clst_regr * self.coefs.clst_regr
        self.losses_dict['Clst_regr'] = Clst_regr
    
    def update_PSD(self, min_distances, labels):
        PSD = self.PSD_loss(min_distances, labels)
        self.total_loss = self.total_loss + PSD * self.coefs.PSD
        self.losses_dict['PSD'] = PSD
