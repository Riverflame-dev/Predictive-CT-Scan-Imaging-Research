"""Some code is borrowed and adapted from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""

from pathlib import Path
from typing import Any, List, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import Figure
from numpy import ndarray
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor


def make_imgs(img: ndarray, imin: Any = None, imax: Any = None) -> ndarray:
    """Apply a 3D binary mask to a 1-channel, 3D ndarray `img` by creating a 3-channel
    image with masked regions shown in transparent blue."""
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=int)  # img
    return scaled


def get_logger(logdir: Path) -> TensorBoardLogger:
    return TensorBoardLogger(str(logdir), name="unet")


class BrainSlices:
    def __init__(self, lightning: LightningModule, target: Tensor, prediction: Tensor):
        self.lightning = lightning
        self.target_img: ndarray = (
            target.cpu().detach().numpy().squeeze() if torch.is_tensor(target) else target
        )
        self.predict_img: ndarray = (
            prediction.cpu().detach().numpy().squeeze()
            if torch.is_tensor(prediction)
            else prediction
        )

        si, sj, sk = self.target_img.shape[:3]
        i = si // 2
        j = sj // 2
        k = sk // 2
        self.slices = [
            self.get_slice(self.target_img, i, j, k),
            self.get_slice(self.predict_img, i, j, k),
        ]

        self.shape = np.array(self.target_img.shape)

    def get_slice(self, input: ndarray, i: int, j: int, k: int) -> List[Tuple[ndarray, ...]]:
        return [
            (input[i // 2, ...], input[i, ...], input[i + i // 2, ...]),
            (input[:, j // 2, ...], input[:, j, ...], input[:, j + j // 2, ...]),
            (input[:, :, k // 2, ...], input[:, :, k, ...], input[:, :, k + k // 2, ...]),
        ]

    def plot(self) -> Figure:
        nrows, ncols = 2, 3

        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(nrows, ncols)
        for i in range(0, nrows):
            ax1 = plt.subplot(gs[i * 3])
            ax2 = plt.subplot(gs[i * 3 + 1])
            ax3 = plt.subplot(gs[i * 3 + 2])
            axes = ax1, ax2, ax3
            self.plot_row(self.slices[i], axes)
            for axis in axes:
                if i == 0:
                    axis.set_title("target image")
                else:
                    axis.set_title("predict image")

        plt.tight_layout()
        return fig

    def plot_row(self, slices: List, axes: Tuple[Any, Any, Any]) -> None:
        for (slice_, axis) in zip(slices, axes):
            imgs = [img for img in slice_]
            imgs = np.concatenate(imgs, axis=1)

            axis.imshow(imgs, cmap="bone", alpha=0.8)
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])

    def log(self, state: str, fig: Figure, loss: float, batch_idx: int) -> None:
        logger = self.lightning.logger
        summary = (
            f"{state}-Epoch:{self.lightning.current_epoch + 1}-batch:{batch_idx}-loss:{loss:0.5e}"
        )
        logger.experiment.add_figure(summary, fig, close=True)
        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)


def log_all_info(
    module: LightningModule,
    target: Union[Tensor, ndarray],
    preb: Union[Tensor, ndarray],
    loss: float,
    batch_idx: int,
    state: str,
) -> None:
    brainSlice = BrainSlices(module, target, preb)
    fig = brainSlice.plot()
    brainSlice.log(state, fig, loss, batch_idx)
