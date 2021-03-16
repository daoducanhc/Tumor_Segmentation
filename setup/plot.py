import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def result(image, mask, output, title, transparency=0.38):

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(
        20, 15), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
    fig.suptitle(title, x=0.5, y=0.92, fontsize=20)

    axs[0][0].set_title("Original Image", fontdict={'fontsize': 16})
    axs[0][0].imshow(image, cmap='gray')
    axs[0][0].set_axis_off()

    seg_output = mask*transparency
    seg_image = np.add(image, seg_output)/2
    axs[0][1].set_title("Original Segment", fontdict={'fontsize': 16})
    axs[0][1].imshow(seg_image, cmap='gray')
    axs[0][1].set_axis_off()

    seg_output = output*transparency
    seg_image = np.add(image, seg_output)/2
    axs[0][2].set_title("Constructed Segment", fontdict={'fontsize': 16})
    axs[0][2].imshow(seg_image, cmap='gray')
    axs[0][2].set_axis_off()

    mask_diff = np.abs(np.subtract(mask, output))
    axs[1][0].set_title("Mask Difference", fontdict={'fontsize': 16})
    axs[1][0].imshow(mask_diff, cmap='gray')
    axs[1][0].set_axis_off()

    axs[1][1].set_title("Original Mask", fontdict={'fontsize': 16})
    axs[1][1].imshow(mask, cmap='gray')
    axs[1][1].set_axis_off()

    axs[1][2].set_title("Constructed Mask", fontdict={'fontsize': 26})
    axs[1][2].imshow(output, cmap='gray')
    axs[1][2].set_axis_off()
    
    plt.tight_layout()

    plt.show()


