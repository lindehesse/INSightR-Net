""" 
11-05-2022 Linde S. Hesse

Main runner code to run baseline training  
"""
import torch
import torch.utils.data
from datamodule import  MyDataModuleDiabRet
from define_parameters import Parameters
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import logging
import argparse
from callbacks import MyPrintingCallback
from pytorch_lightning.profiler import SimpleProfiler
from baseline_training.lightning_module_baseline import LitModel_baseline
from helpers import load_json
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning import seed_everything


def train_runner_baseline(params):
    """ Function to run the network training for the baseline model    
    Args:
        params (Parameter): dataclass containing all parameters
    """

    # Set up ML Flow logger to monitor experimnets
    mlf_logger = MLFlowLogger(
        experiment_name=params.experiment_run, run_name=params.run_name)

    # logging for outside trainer object
    txt_logger = logging.getLogger("pytorch_lightning")

    # remove file of old logger
    txt_logger.addHandler(logging.FileHandler(
        params.save_path / "logfile.log"))

    try:
        dataset = MyDataModuleDiabRet(params)

        # Set up required callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss_monitor',  dirpath=params.save_path,
            filename='{epoch}-{step}-{val_kappa:.3f}')
        lr_callback = LearningRateMonitor(logging_interval='epoch')
        print_callback = MyPrintingCallback()
        all_callbacks = [checkpoint_callback,
                         lr_callback, print_callback]


        # Set up memory / time profiler
        profiler = SimpleProfiler(
            dirpath=params.save_path, filename="profiler.txt")

        # Set up model and trainer and train with PL
        model = LitModel_baseline(params)

        # Set up trainer
        trainer = pl.Trainer(logger=mlf_logger,
                             gpus=torch.cuda.device_count(),
                             max_epochs=params.num_train_epochs,
                             callbacks=all_callbacks,
                             fast_dev_run=False,
                             limit_train_batches=1.0,
                             limit_val_batches=1.0,
                             log_every_n_steps=10,
                             flush_logs_every_n_steps=50, profiler=profiler, 
                             deterministic=True)
        trainer.fit(model, dataset)
        txt_logger.info('Finished Training Succesfully')

        # Test with model from best validation
        trainer.test(ckpt_path=checkpoint_callback.best_model_path)
        rank_zero_info('Completed PL testing')

    except Exception:
        txt_logger.error('Failed to run trainer', exc_info=True)

    # Log txt logger to mlflow
    mlf_logger.experiment.log_artifact(
        run_id=mlf_logger.run_id,
        local_path=params.save_path / 'logfile.log')

    # Log time/memory profile to mlflow
    profile_files = params.save_path.glob('*profiler*.txt')
    for file in profile_files:
        mlf_logger.experiment.log_artifact(
            run_id=mlf_logger.run_id,
            local_path=file)
    

def run_crossvalidation(params, num_folds = 1):
    """ Runs crossvalidation with given parameters

    Args:
        params (namespace): dataclass containing all parameters for run
        num_folds (int): number of cross validation runs
    """

    base_runname = params.run_name
    for fold in range(num_folds):
        params.cv_fold = fold
        
        # Reinitialize savepaths
        params.run_name = f'Fold{fold}_{base_runname}'
        params.set_savepaths()
        params.save_path_ims.mkdir(parents=True, exist_ok=True)

        # Set handler to new file
        txt_logger = logging.getLogger("pytorch_lightning")
        filehandler = logging.FileHandler(params.save_path / "logfile.log")
        txt_logger.addHandler(filehandler)


        # Run training
        rank_zero_info(f'Start fold {fold}')
        train_runner_baseline(params)
    

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(
        description='Run ProtoPnet code')

    parser.add_argument('--run_name',
                help = 'Define the run_name for saving everything (example: test_resnet)',
                default='DummyName')
    
    parser.add_argument('--param_jsonpath',
                        default='config/params_example_baseline.json',
                        required=False,
                        help='Define the parameter file used for training (example: config/params_example_baseline.json)')
    
    parser.add_argument('--datapath',
                        required=True)
    
    parser.add_argument('--savepath',
                        default='savepath_folder',
                        required=True)
                    
    parser.add_argument('--pretrained_path',
                        required=False)

    # set params
    args_dict = vars(parser.parse_args())
    params_dict = load_json(args_dict['param_jsonpath'])
    params_exp = { **params_dict,**args_dict}
    params = Parameters.from_dict(params_exp)
    params.experiment_run = 'Baselines'

    # run crossval
    run_crossvalidation(params, num_folds = 1)

