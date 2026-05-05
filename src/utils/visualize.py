import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_translation_results(ct, real_mri, fake_mri, title="Translation Results", save_path=None):
    """
    Plot CT, Real MRI, and Synthetic MRI for comparison.
    Expects tensors in range [-1, 1]
    """
    ct = (ct[0, 0].cpu().numpy() + 1.0) / 2.0
    real_mri = (real_mri[0, 0].cpu().numpy() + 1.0) / 2.0
    fake_mri = (fake_mri[0, 0].cpu().numpy() + 1.0) / 2.0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ct, cmap='gray')
    axes[0].set_title("Input CT")
    axes[0].axis('off')
    
    axes[1].imshow(real_mri, cmap='gray')
    axes[1].set_title("Real MRI")
    axes[1].axis('off')
    
    axes[2].imshow(fake_mri, cmap='gray')
    axes[2].set_title("Synthetic MRI")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig

def save_image_batch(tensor, path, nrow=8, normalize=True, range=(-1, 1)):
    from torchvision.utils import save_image
    save_image(tensor, path, nrow=nrow, normalize=normalize, value_range=range)
