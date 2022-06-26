""" 
11-05-2022 Linde S. Hesse

File defining all parameters for training in dataclasses
"""
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, Undefined
from enforce_typing import enforce_types
from pathlib import Path
from typing import List
import os


@dataclass_json
@dataclass
class Joint_Lrs:
    features: float = 1e-5
    add_on_layers: float = 1e-3
    prototype_vectors: float = 1e-3

@dataclass_json
@dataclass
class Warm_Lrs:
    add_on_layers: float = 1e-3
    prototype_vectors: float =  1e-3

@enforce_types
@dataclass_json
@dataclass
class OptimParams:
    # All parameters regarding optimization and lr decay
    # When using pretrained deepDR model, features 1e-5 and other 3e-3 work`s best
    joint_optim_lrs: Joint_Lrs = Joint_Lrs()
    warm_optim_lrs: Warm_Lrs = Warm_Lrs()
    
    joint_lr_stepsize: int = 5
    joint_lr_decay: bool = True
    ll_optim_lr: float = 1e-3

@dataclass_json
@dataclass
class ParamsBaseline:
    # All parameters regarding optimization of baseline model
    single_lr: float = 5e-4
    lr_decay: bool = True
    lr_step: int = 5 # step size used for lr decay (only used if lr_decay == True)
    loss: str = 'MSE'
    num_classes: int = 1 # can be either 1 for MSE (regression-like) or higher than 1 for CE (classification-like)
    max_pred: int = 5 # will be used to clamp predictions for regression so only if MSE
    network: str = 'resnet18'
    assert loss in ['MSE', 'CE']
   
    optim: str = 'Adam' # Either Adam or AdamW
    add_onlayers: bool = False
    base_architecture: str = 'resnet18'
    pretrained: bool = True


@dataclass_json
@dataclass
class LossCoefs:
    # Coefficients for the loss calculation
    mse: float = 1
    PSD: float = 10
    clst_regr: float = 1

    clst_class_specific: bool = True
    clst_delta: float = 0.5
    clst_kmin: int = 4


@dataclass_json
@dataclass
class NetworkParams:
    base_architecture: str = 'resnet18'
    img_size: int = 540 # input image size
    output_size_conv: int = 9 # latent space dimension
    proto_shape: List[int] = field(default_factory=lambda: [50, 128, 1, 1]) # latent size of prototypes
    bias_ll: bool = False # whether the last layer has a bias therm
    epsilon:float = 1e-4 # size of epsilon use in simmiliarty function
    pretrained: bool = True 

    proto_activation: str = 'exp_norm'           
    proto_minrange: float = 0.1 # minimum prototype label
    proto_maxrange: float = 5.9 # maximum prototype label
    init_ll = 'class_idx' # last layer initialization with class_idx or with ones
    assert init_ll in ['class_idx', 'ones']


@enforce_types
@dataclass_json(undefined = Undefined.RAISE)
@dataclass
class Parameters:
    # Initialize these with command line bash
    param_jsonpath: str 
    datapath: str
    savepath: str
    
    pretrained_path: str = None 
    
    # Name for experiment
    experiment_run: str = 'Experiments'
    run_name: str = 'RandomRun'
    
    cv_fold: int = 0
    
    # Define training parameters
    num_train_epochs: int = 60
    num_warm_epochs: int = 5
    push_start: int = 20 
    push_frequency: int =  20  # frequency of protype pushing (= number of joint epochs in between)
    ll_iteration: int = 10  # number of convex iterations for last layer
    
    # Define network parameters
    network_params: NetworkParams = NetworkParams()

    # Define optimizers and loss coefs
    optim_params: OptimParams = OptimParams()
    loss_coefs: LossCoefs = LossCoefs()
    baseline_params: ParamsBaseline = ParamsBaseline()

    # Define batchsizes
    train_batch_size: int = 30
    val_batch_size: int = 30
    push_batch_size: int = 30
    preload: bool = False # preloading data
    labels: str = 'classes' # either 'fussy' for real-valued or 'classes' for ordinal
    min_label: int = 1 # min predicted label 
    max_label: int = 5 # max predicted label

    # Define datapaths
    datasplittrain_path: Path = Path('config/datasplit/new_data_cv.json')
    datasplittest_path : Path = Path('config/datasplit/new_data_test.json')
    
    # Define folders names for saving
    save_prototypes: bool = True

    # Initialize variables defined in post_init and make savepaths
    data_path: Path = field(init=False)
    save_basepath: Path = field(init=False)
    train_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    val_dir: Path = field(init=False)

    save_path_experiment: Path = field(default_factory=Path)
    save_path: Path = field(default_factory=Path)
    save_path_ims: Path = field(default_factory=Path)
    

    def __post_init__(self):
        # Determine datapath based on location
        self.data_path = Path(self.datapath)
        self.save_basepath = Path(self.savepath)
        self.pretrained_path = Path(self.pretrained_path)

        # Make full folder names for data
        self.train_dir = self.data_path / 'train'
        self.val_dir  = self.data_path / 'train'
        self.test_dir = self.data_path / 'test'

    def set_savepaths(self):
        # Make full folder names for saving
        self.save_path_experiment = self.save_basepath / self.experiment_run
        self.save_path =  self.save_path_experiment / self.run_name
        self.save_path_ims =  self.save_path / 'img'
       


