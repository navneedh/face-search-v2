import tensorflow as tf
# from google.colab import files
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
import sample as sp 
import util as ut


# Initialize TensorFlow
tflib.init_tf()

# Load pre-trained network.
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)

def run(experimentNum, num_trials = 20, learning_rate = 15, noise = 0.99, alpha = 0.99):
	os.mkdir("exp" + str(experimentNum))

	#Generate and save original image that you will 
	seed = 3242
	np.random.seed(19238817)

	original_z = sp.random_vector()

	print("Generating original image ...")

	o_image = sp.z_sample(Gs, original_z)
	imsave("./" + "exp" + str(experimentNum) + "/original.png", o_image)
	plt.imshow(o_image)
	plt.grid('off')
	plt.axis('off')

	#Keep track of experimental data
	z_vectors = [] # z vector after each iteration
	error_vals = [] # pixel error w.r.t original image after each iteration
	total_grid = []

	cur_z = sp.random_vector()
	r_image = sp.z_sample(Gs, cur_z)
	first_image = r_image
	imsave("./exp" + str(experimentNum) + "/reconstructed_"  +str(1)+".png", r_image)
	error_vals.append(ut.pixel_error(r_image, o_image))
	plt.imshow(r_image)
	plt.draw()
	plt.pause(0.001)

	for exp_iter in range(1,num_trials + 1):
	  
	    print("Input value between 1-3 for desired noise level")
	    print("1: Least Noise - 3: Most Noise")
	    raw_noise_level = input()
	    if int(raw_noise_level) == 1:
	        noisyVecs, noisyImages, noises = sp.gen_grid_exp(cur_z, exp_iter, 0.5)
	    elif int(raw_noise_level) == 2:
	        noisyVecs, noisyImages, noises = sp.gen_grid_exp(cur_z, exp_iter, 1.3)
	    else:
	        noisyVecs, noisyImages, noises = sp.gen_grid_exp(cur_z, exp_iter, 8)
	    temp_grid =  [0] * 6 
	    
	    # use commas to separate ranking scores 
	    raw_rankings = input() 
	    rankings = np.array([int(x) for x in raw_rankings.split(",")])
	    
	    #for visualization purposes 
	    for i,r in enumerate(rankings):
	        temp_grid[r - 1] = noisyImages[i]
	    total_grid += temp_grid
	    
	    rankings = (rankings - rankings.mean())/rankings.std()
	    noisyVecsSum = np.zeros(512).reshape(1,512)
	    for i,r in enumerate(rankings):
	        noisyVecsSum += r * (noises[i])

	    #update step 
	    learning_rate = (alpha**(exp_iter/2))
	    cur_z = cur_z + learning_rate * (noisyVecsSum/(6 * noise))
	    
	    
	    print("Generating reconstructed image ...")
	    r_image = sp.z_sample(Gs,cur_z)
	    total_grid.append(r_image)
	    z_vectors.append(cur_z)
	    imsave("./exp" + str(experimentNum) + "/reconstructed_"  +str(exp_iter + 1)+".png", r_image)
	    error_vals.append(ut.pixel_error(r_image, o_image))
	    print(error_vals)
	    plt.imshow(r_image)
	    plt.draw()
	    plt.grid('off')
	    plt.axis('off')
	    plt.pause(0.001)
	    
	ut.gen_grid_vis(o_image, first_image, total_grid, num_trials)
