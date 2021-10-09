import glob
from PIL import Image, ImageStat
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    # IMPLEMENT THIS FUNCTION
    mean_val = []
    std_val = []
    
    for image in image_list:
        img = Image.open(image)
        stat = ImageStat.Stat(img)
        mean_val.append(stat.mean)  
        std_val.append(stat.stddev)
    
    mean = np.mean(mean_val, axis = 0)
    std = np.mean(std_val, axis = 0)
    
    return mean, std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    # IMPLEMENT THIS FUNCTION
    r_val = []
    g_val = []
    b_val = []
    for image in image_list:
        img = np.array(Image.open(image).convert('RGB'))
        r,g,b = img[...,0], img[...,1], img[...,2]
        r_val.extend(r.flatten())
        g_val.extend(g.flatten())
        b_val.extend(b.flatten())
        
    plt.figure(figsize=(15,5))
    sns.kdeplot(np.array(r_val), color = 'r')
    sns.kdeplot(np.array(g_val), color = 'g')
    sns.kdeplot(np.array(b_val), color = 'b')
    plt.show()
    
        


if __name__ == "__main__": 
    image_list = glob.glob('data/images/*')
    mean, std = calculate_mean_std(image_list)
    channel_histogram(image_list[:2])
    check_results(mean, std)