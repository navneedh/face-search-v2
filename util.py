import tensorflow as tf
from google.colab import files
from scipy.misc import imsave
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import dnnlib as dnnlib
import dnnlib.tflib as tflib
import config as config
import pickle
import os  

def gen_grid_vis(original_image, first_image, ordered_images, num_trials):
    new_im = Image.new('RGB', (458, num_trials * 64 + 128))
    index = 0
    im = Image.fromarray(first_image)
    im.thumbnail((64,64))
    new_im.paste(im, (394,0))
    im = Image.fromarray(original_image)
    im.thumbnail((64,64))
    new_im.paste(im, (394,num_trials * 64 + 64))
    
    for i in range(64, 64 + num_trials * 64,64):
        for j in range(0,448,64):
            im = Image.fromarray(ordered_images[index])
            im.thumbnail((64,64))
            if j == 384:
                new_im.paste(im, (j + 10,i))
            else:
                new_im.paste(im, (j,i))
            index += 1
        
    new_im.save("./exp" + str(experimentNum) + "/data.png")

def pixel_error(image1, image2):
    difference = image1 - image2
    error = np.linalg.norm(difference)
    return error