"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python test.py --ckpt_file wandb/aue8088-pa1/ygeiua2t/checkpoints/epoch\=19-step\=62500.ckpt
"""
# Python packages
import argparse

# PyTorch & Pytorch Lightning
from lightning import Trainer
from torch.utils.flop_counter import FlopCounterMode
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
#import src.config as cfg

# Using hydra for config management
import os
import hydra
from omegaconf import DictConfig

torch.set_float32_matmul_precision('medium')

@hydra.main(config_name='config.yaml', config_path='config', version_base='1.1')
def test(cfg: DictConfig):
    cfg.WANDB_ENTITY = os.environ.get('WANDB_ENTITY')
    cfg.WANDB_NAME = f'{cfg.MODEL_NAME}-B{cfg.BATCH_SIZE}-{cfg.OPTIMIZER_PARAMS["type"]}'
    cfg.WANDB_NAME += f'-{cfg.SCHEDULER_PARAMS["type"]}{cfg.OPTIMIZER_PARAMS["lr"]:.1E}'

    # args = argparse.ArgumentParser()
    # args.add_argument('--ckpt_file',
    #     default='/home/sungjin/Codes/AUE8088-PA1/wandb/aue8088-pa1/zwrb3zsa/checkpoints/epoch=50-step=9996.ckpt',
    #     type = str,
    #     help = 'Model checkpoint file name')
    # args = args.parse_args()

    model = SimpleClassifier(
        cfg=cfg,
        model_name = cfg.MODEL_NAME,
        num_classes = cfg.NUM_CLASSES,
    )

    datamodule = TinyImageNetDatasetModule(
        cfg=cfg,
        batch_size = 1,
    )

    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        benchmark = True,
        inference_mode = True,
        logger = False,
    )

    model.load_from_checkpoint(checkpoint_path=cfg.CKPT)
    model.eval()
    
    trainer.validate(model, datamodule = datamodule)
    if cfg.EXPORT:
        print('Exporting model to ONNX format')
        filepath = f'{cfg.CKPT}/{cfg.MODEL_NAME}_classifier.onnx'
        input_sample = torch.randn(1, 3, 64, 64)
        model.to_onnx(filepath, input_sample, export_params=True)
        
    
    # FLOP counter
    x, y = next(iter(datamodule.test_dataloader()))
    flop_counter = FlopCounterMode(model, depth=1)

    with flop_counter:
        model(x)

if __name__ == "__main__":
    test()