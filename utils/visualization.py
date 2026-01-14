import numpy as np
import matplotlib.pyplot as plt


def visualize(real_w0: np.ndarray, pred_w0: np.ndarray) -> None:
    """Plot ground truth, prediction, and absolute error."""
    error = pred_w0 - real_w0
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axs[0].imshow(real_w0, cmap="turbo")
    axs[0].set_title("True w0")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred_w0, cmap="turbo")
    axs[1].set_title("Predicted w0")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(np.abs(error), cmap="inferno")
    axs[2].set_title("Abs Error")
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()
