""" 
11-05-2022 Linde S. Hesse

Callback functions used for logging
"""
from pytorch_lightning.callbacks import Callback
import time
import logging
import os
from pytorch_lightning.utilities import rank_zero_only

txt_logger = logging.getLogger("pytorch_lightning")


class MyPrintingCallback(Callback):
    def on_init_start(self, trainer):
        print('\n Starting to initialize trainer')

    def on_init_end(self, trainer):
        print('\n Trainer has been succesfully initialized')

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        local_time = time.ctime(time.time())
        print(local_time, '\n Starting Training now')

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
       # local_time = time.ctime(time.time())
        #txt_logger.info(local_time + f' Start training epoch {pl_module.current_epoch}')
        self.start_time_epoch = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        local_time = time.ctime(time.time())
        train_time = time.time() - self.start_time_epoch
        txt_logger.info(
            local_time + f' Finished epoch {pl_module.current_epoch}, train_time [mins]: {train_time/60:.2f}')

        pl_module.logger.experiment.log_artifact(
            run_id=pl_module.logger.run_id, local_path=os.path.join(pl_module.params.save_path,
                                                                    'logfile.log'))
