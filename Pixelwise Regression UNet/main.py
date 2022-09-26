import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from lig_module.data_model import DataModule
from lig_module.lig_model import LitModel

from utils.const import COMPUTECANADA

from monai.utils import first
import matplotlib.pyplot as plt
import torch
from torchsummaryX import summary

def main(hparams: Namespace) -> None:
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy,
    # python.random and sets PYTHONHASHSEED environment variable.
    pl.seed_everything(42)

    if COMPUTECANADA:
        cur_path = Path(__file__).resolve().parent
        default_root_dir = cur_path
        checkpoint_file = (
            Path(__file__).resolve().parent
            / "checkpoint/{epoch}-{val_loss:0.5f}-{val_MAE:0.5f}-{val_MSE:0.5f}-{val_SSIM:0.5f}-{val_PSNR:0.5f}"  # noqa
        )
        if not os.path.exists(Path(__file__).resolve().parent / "checkpoint"):
            os.mkdir(Path(__file__).resolve().parent / "checkpoint")
    else:
        default_root_dir = Path("./log")
        if not os.path.exists(default_root_dir):
            os.mkdir(default_root_dir)
        
    

    # After training finishes, use best_model_path to retrieve the path to the best
    # checkpoint file and best_model_score to retrieve its score.
    
    tb_logger = loggers.TensorBoardLogger(hparams.TensorBoardLogger)

    # training
    trainer = Trainer(
        gpus=hparams.gpus,
        fast_dev_run=hparams.fast_dev_run,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
        ],
        default_root_dir=str(default_root_dir),
        logger=tb_logger,
        max_epochs=11,
    )

    model = LitModel(hparams)
    data_module = DataModule(
        hparams.batch_size,
        X_image=hparams.X_image,
        y_image=hparams.y_image,
    )
   
    trainer.fit(model, data_module)

    #sum = summary(model, torch.zeros((1, 3, 128, 128)))
    #print(sum)

    # trainer = Trainer(gpus=hparams.gpus, distributed_backend="ddp")
    # trainer.test(
    #     model=model,
    #     ckpt_path=ckpt_path,
    #     datamodule=data_module,
    # )

    # trainer.test()


if __name__ == "__main__":  # pragma: no cover
    parser = ArgumentParser(description="Trainer args", add_help=False)
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size", dest="batch_size")
    parser.add_argument(
        "--tensor_board_logger",
        dest="TensorBoardLogger",
        default="/home/jq/Desktop/log",
        help="TensorBoardLogger dir",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="whether to run 1 train, val, test batch and program ends",
    )
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--X_image", type=str, choices=["t1", "t2", "S2020_masked_vol"], default="S2020_masked_vol")
    parser.add_argument("--y_image", type=str, choices=["t1", "t2", "S2020_masked_vol"], default="S2020_masked_vol")
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "t1t2",
            "diffusion_adc",
            "diffusion_fa",
            "diffusion_2d_adc",
            "diffusion_2d_fa",
            "longitudinal",
        ],
        default="diffusion_adc",
    )
    parser.add_argument("--checkpoint_file", type=str, help="resume from checkpoint file")
    parser = LitModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
