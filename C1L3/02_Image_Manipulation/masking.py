import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 


def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    # IMPLEMENT THIS FUNCTION
    img = Image.open(path).convert('RGB')
    r,g,b = np.array(img.split())
    
    mask = ((np.array(r)>color_threshold[0]) & 
            (np.array(g)>color_threshold[1]) & 
            (np.array(b)>color_threshold[2])).astype(int)
    
    return img, mask


def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    # IMPLEMENT THIS FUNCTION
    img_mask = np.stack([mask]*3, axis=2)
    img_masked = img * img_mask
    
    f, ax = plt.subplots(1,3, figsize = (15,5))
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(img_masked)
    plt.show()
    


if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)
    