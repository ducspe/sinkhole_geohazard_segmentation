import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from transfer_learning_train import DeadSeaUNet

# Parameters section:
TEST_NUM_DEVICES = 1
NUM_NODES = 1
args = {
    'in_channels': 3,
    'out_classes': 3,
    'batch_size': 64,
    'logs_dir': '../cloudcontainer/logs_for_deadsea_unet_test_dot_py'
}
test_checkpoint_path = "../cloudcontainer/satellite/transfer_learning_models/tr_best_fine_tuned_model.ckpt"
best_model_path = "../cloudcontainer/satellite/transfer_learning_models/tr_best_fine_tuned_model.ckpt"
mylogger = pl_loggers.TensorBoardLogger(save_dir=args['logs_dir'])
# End of parameters section.

pl.seed_everything(34)  # for reproducibility

trainer = pl.Trainer(devices=TEST_NUM_DEVICES, num_nodes=NUM_NODES, logger=mylogger)
test_tr_model = DeadSeaUNet.load_from_checkpoint(checkpoint_path=best_model_path, n_channels=args['in_channels'], n_classes=args['out_classes'], batch_size=args['batch_size'], instantiated_only_to_test_checkpoint=True)
trainer.test(test_tr_model)