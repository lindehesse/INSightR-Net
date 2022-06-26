""" 
11-05-2022 Linde S. Hesse

Main runner code to run InsightR-Net training  
"""
import torch
import torch.utils.data
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import logging
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info

from insight_training.lightning_module_protodiab import LitModelProto
from datamodule import  MyDataModuleDiabRet
from helpers import load_json
from define_parameters import Parameters
from callbacks import MyPrintingCallback


def train_runner(params):
    """ Function to run the network training

    Args:
        params (Parameter): dataclass containing all parameters
    """

    # Set up ML Flow logger to monitor experimnets
    mlf_logger = MLFlowLogger(
        experiment_name=params.experiment_run, run_name=params.run_name)

    try:  
        # Define Dataset
        dataset = MyDataModuleDiabRet(params)

        # Set up required callbacks
        checkpoint_callback = ModelCheckpoint(filename = 'bestmodel_{epoch:02d}-{valloss:02d}', monitor='val_loss_monitor',  dirpath= params.save_path / 'checkpoints')
        checkpoint_callback_stage2 = ModelCheckpoint(filename = 'bestmodel_stage2_{epoch:02d}-{valloss:02d}',  monitor='val_loss_stage2',  dirpath=params.save_path / 'checkpoints')
        lr_callback = LearningRateMonitor(logging_interval='epoch')
        print_callback = MyPrintingCallback()
        all_callbacks = [checkpoint_callback, checkpoint_callback_stage2,
                                        lr_callback, print_callback]

        # Set up memory / time profiler
        profiler = SimpleProfiler(
            dirpath=params.save_path, filename="profiler.txt")

        # Initialize PL 
        model = LitModelProto(params, dataset.push_dataloader())

        # Set up trainer
        trainer = pl.Trainer(logger=mlf_logger,
                            gpus = 1,
                            max_epochs=params.num_train_epochs,
                            accelerator = None,
                            callbacks=all_callbacks,
                            limit_train_batches=1.0,
                            limit_val_batches=1.0,
                            log_every_n_steps=10,
                            flush_logs_every_n_steps=50, profiler=profiler, 
                            deterministic=True)
        trainer.fit(model, dataset)
        rank_zero_info('Finished Training Succesfully')

        # Test with best model from linear convex stage
        trainer.test(ckpt_path=checkpoint_callback_stage2.best_model_path)
        rank_zero_info('Completed PL testing')

        # Do some more detailed testing outside pl for best model in stage 2
        rank_zero_info('Load in best checkpoint')
        model = model.load_from_checkpoint(checkpoint_path=checkpoint_callback_stage2.best_model_path, params = params, dataloader_push = dataset.push_dataloader())

        # Plot images for first three batches
        rank_zero_info('Do test plotting for first few batches')
        test_loader = dataset.test_dataloader()
        savepath_preds = params.save_path / 'predicted_ims'
        savepath_preds.mkdir(parents=True, exist_ok=True)
       
        num_analysis = 2
        for i, batch in enumerate(test_loader):
            savenames = [str(savepath_preds / f'batch_{i}_im_{j}') for j in range(batch[0].shape[0])]
            if i < num_analysis:
                with torch.no_grad(): 
                    model(batch, save_predictions=True,savenames = savenames)
                rank_zero_info(f'Completed {i+1} batches out of {num_analysis}')
            else:
                break
                
        # Log predicted images to mlflow
        mlf_logger.experiment.log_artifact(
            run_id=mlf_logger.run_id,
            local_path=params.save_path / 'predicted_ims')

        # Log time/memory profile to mlflow
        profile_files = params.save_path.glob('*profiler*.txt')
        for file in profile_files:
            mlf_logger.experiment.log_artifact(
                run_id=mlf_logger.run_id,
                local_path=file)
        
        rank_zero_info('Finished Testing Succesfully')

    except Exception:
        rank_zero_info('Failed to run trainer', exc_info=True)

    # Log txt logger to mlflow
    mlf_logger.experiment.log_artifact(
        run_id=mlf_logger.run_id,
        local_path=params.save_path / 'logfile.log')


if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser(
        description='Run ProtoPnet code')

    parser.add_argument('--run_name',
                help = 'Define the run_name for saving everything (example: test_resnet)',
                default='DummyName')
                
    parser.add_argument('--param_jsonpath',
                        default='config/params_example_ordinal.json',
                        required=True,
                        help='Define the parameter file used for training (example: config/params_example_ordinal.json)')
    
    parser.add_argument('--datapath',
                        required=True)
    
    parser.add_argument('--savepath',
                        default = 'savepath_folder'
                        required=True)
                    
    parser.add_argument('--pretrained_path',
                        default = 'config/pretrained_model.ckpt')

    # Parse command line arguments
    args_dict = vars(parser.parse_args())
    params_dict = load_json(args_dict['param_jsonpath'])
    params_exp = { **params_dict,**args_dict}
    params = Parameters.from_dict(params_exp)

    params.set_savepaths()
    base_runname = params.run_name

    num_folds = 1
    for fold in range(num_folds):
        # Set crossvalidation
        params.cv_fold = fold
        
        # Reinitialize savepaths
        params.run_name = f'Fold{fold}_{base_runname}'
        params.set_savepaths()
        params.save_path_ims.mkdir(parents=True, exist_ok=True)

        # Set handler to new file
            # logging for outside trainer object
        txt_logger = logging.getLogger("pytorch_lightning")
        filehandler = logging.FileHandler(params.save_path / "logfile.log")
        txt_logger.addHandler(filehandler)

        # Run training
        rank_zero_info(f'Start fold {fold}')
        train_runner(params)

        # Remove handler from logger
        txt_logger.removeHandler(filehandler)
    
    