"""authors: Alexander Ortlieb, Maximilian Glumann"""

import matplotlib.pyplot as plt

def plot_histogram_matching(source, reference, matched, matched_noise = None, equalized = None, diff = None):
    fig, axs = plt.subplots(2,3, figsize=(7, 4.5))

    if matched_noise is None or equalized is None or diff is None:
        axs[0,0].hist(source.flatten(), cumulative=True, bins=255, range=(0,255))
        axs[0,0].set_title('source')
        axs[0,0].set_xlabel('pixel value')
        axs[0,0].set_ylabel('frequency')
        axs[0,1].hist(reference.flatten(), cumulative=True, bins=255, range=(0,255))
        axs[0,1].set_title('reference')
        axs[0,1].set_xlabel('pixel value')
        axs[0,1].set_ylabel('frequency')
        axs[0,2].hist(matched.flatten(), cumulative=True, bins=255, range=(0,255))
        axs[0,2].set_title('matched')
        axs[0,2].set_xlabel('pixel value')
        axs[0,2].set_ylabel('frequency')
        index = 1
        axs[index,0].set_title(f'mean: {source.mean():.2f}, std: {source.std():.2f}')
        axs[index,1].set_title(f'mean: {reference.mean():.2f}, std: {reference.std():.2f}')
        axs[index,2].set_title(f'mean: {matched.mean():.2f}, std: {matched.std():.2f}')
    else:
        index = 0
        axs[index,0].set_title(f'source \n mean: {source.mean():.2f}, std: {source.std():.2f}')
        axs[index,1].set_title(f'reference \n mean: {reference.mean():.2f}, std: {reference.std():.2f}')
        axs[index,2].set_title(f'matched \n mean: {matched.mean():.2f}, std: {matched.std():.2f}')
    
    axs[index,0].imshow(source, cmap = 'gray', vmin = 0, vmax = 255)
    axs[index,1].imshow(reference, cmap = 'gray', vmin = 0, vmax = 255)
    axs[index,2].imshow(matched, cmap = 'gray', vmin = 0, vmax = 255)
    
    axs[index,0].axis('off')
    axs[index,1].axis('off')
    axs[index,2].axis('off')
    
    if matched_noise is not None and equalized is not None and diff is not None:
        axs[1,0].imshow(matched_noise, cmap = 'gray', vmin = 0, vmax = 255)
        axs[1,0].set_title(f'matched + noise \n mean: {matched_noise.mean():.2f}, std: {matched_noise.std():.2f}')
        axs[1,1].imshow(equalized, cmap = 'gray', vmin = 0, vmax = 255)
        axs[1,1].set_title(f'reconstructed \n mean: {equalized.mean():.2f}, std: {equalized.std():.2f}')
        axs[1,2].imshow(diff, cmap = 'gray')
        axs[1,2].set_title(f'reconstruction - source  \n mean: {diff.mean():.2f}, std: {diff.std():.2f}')

        axs[1,0].axis('off')
        axs[1,1].axis('off')
        axs[1,2].axis('off')
        
    plt.tight_layout()
    plt.close()
    return fig