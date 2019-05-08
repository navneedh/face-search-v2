import tensorflow as tf
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
import time 
# some handy functions to use along widgets
from IPython.display import display, Markdown, clear_output
# widget packages
import ipywidgets as widgets
from IPython.core.display import Image as Imdisplay

# Initialize TensorFlow
tflib.init_tf()

# Load pre-trained network.
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
	_G, _D, Gs = pickle.load(f)
	# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
	# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
	# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.


def get_rating_results(buttons):
	ratings = []
	for i in range(6):
		for j in range(6):
			if buttons[i][j].value == 1:
				print("Image " + str(i+1) + " had rating " + str(j+1))
				ratings.append(j+1)
	
	return ratings


def random_sample(Gs):
	# Pick latent vector.
	rnd = np.random.RandomState(5)
	latents = rnd.randn(1, Gs.input_shape[1])
	print(latents.shape)

	# Generate image.
	fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
	images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

	# Save image.
	os.makedirs(config.result_dir, exist_ok=True)
	png_filename = os.path.join(config.result_dir, 'random.png')
	PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
	print("Done")
	
def z_sample(Gs, z):
	#z.shape = (1,512)
	
	# Generate image.
	fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
	images = Gs.run(z, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

	# Save image.
	os.makedirs(config.result_dir, exist_ok=True)
	png_filename = os.path.join(config.result_dir, 'z_image.png')
	PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
	return images[0]

def random_vector():
	return np.random.normal(0,1,512).reshape(1,512)

def gen_grid_exp(cur_z, exp_iter, experimentNum, original, noise_level = 1):
	noise = 0.99
	seed = np.random.randint(4000)
	np.random.seed(seed)
	noisyVecs = []
	noises = []
	noisyImages = []
	new_im = Image.new('RGB', (1024,128))
	index = 0
	print("Generating grid of noisy images ...")
	print("      1    2    3    4    5    6")
	for i in range(0,768,128):
		np.random.seed(np.random.randint(4362634))
		noise_val = (random_vector() * noise_level) #most noise added
		zs = cur_z + noise_val
		zs = np.clip(zs, -5, 5)
		p_image = z_sample(Gs, zs)
		noises.append(noise_val)
		noisyVecs.append(zs)
		noisyImages.append(p_image)
		im = Image.fromarray(p_image)
		im.thumbnail((128,128))
		new_im.paste(im, (i,0))
		

	print("SIZE", noisyImages[0].shape)

	im = Image.fromarray(original)
	im.thumbnail((128,128))
	new_im.paste(im, (896,0))
	index += 1
		
	new_im.save("./exp" + str(experimentNum) + "/grid_" +str(exp_iter)+".png")
	plt.imshow(new_im)
	plt.draw()
	plt.pause(0.001)
	return noisyVecs, noisyImages, noises

def gen_images_to_rank(image_matrices, original, indexToRemove):
	new_im = Image.new('RGB', ((len(image_matrices) + 1) * 128,128))
	del image_matrices[indexToRemove - 1]
	for i in range(0,len(image_matrices) * 128,128):
		im = Image.fromarray(image_matrices[int(i/128)])
		im.thumbnail((128,128))
		new_im.paste(im, (i,0))
	
	im = Image.fromarray(original)
	im.thumbnail((128,128))
	new_im.paste(im, ((len(image_matrices) + 1) * 128,0))

	plt.imshow(new_im)
	plt.grid('off')
	plt.axis('off')
	plt.draw()
	plt.pause(0.001)



def present_noise_choices(cur_z, exp_iter, experimentNum, noise_level = 1):
	noise = 0.99
	seed = np.random.randint(4000)
	np.random.seed(seed)
	noisyVecs = []
	noises = []
	noisyImages = []
	new_im = Image.new('RGB', (1152,128))
	index = 0
	for i in range(0,384,128):
		np.random.seed(np.random.randint(4362634))
		noise_val = (random_vector() * 0.5) #most noise added
		zs = cur_z + noise_val
		zs = np.clip(zs, -5, 5)
		p_image = z_sample(Gs, zs)
		noises.append(noise_val)
		noisyVecs.append(zs)
		noisyImages.append(p_image)
		im = Image.fromarray(p_image)
		im.thumbnail((128,128))
		new_im.paste(im, (i,0))
		index += 1
	for i in range(384,768,128):
		np.random.seed(np.random.randint(4362634))
		noise_val = (random_vector() * 1.3) #most noise added
		zs = cur_z + noise_val
		zs = np.clip(zs, -5, 5)
		p_image = z_sample(Gs, zs)
		noises.append(noise_val)
		noisyVecs.append(zs)
		noisyImages.append(p_image)
		im = Image.fromarray(p_image)
		im.thumbnail((128,128))
		new_im.paste(im, (i,0))
		index += 1
	for i in range(768,1152,128):
		np.random.seed(np.random.randint(4362634))
		noise_val = (random_vector() * 8) #most noise added
		zs = cur_z + noise_val
		zs = np.clip(zs, -5, 5)
		p_image = z_sample(Gs, zs)
		noises.append(noise_val)
		noisyVecs.append(zs)
		noisyImages.append(p_image)
		im = Image.fromarray(p_image)
		im.thumbnail((128,128))
		new_im.paste(im, (i,0))
		index += 1

		
	new_im.save("./exp" + str(experimentNum) + "/grid_" +str(exp_iter)+".png")
	display(Imdisplay(filename = "./exp" + str(experimentNum) + "/grid_" +str(exp_iter)+".png", width=1900, unconfined=True))
	plt.imshow(new_im, aspect = "equal")
	plt.grid('off')
	plt.axis('off')
	plt.draw()
	plt.pause(0.001)
	return noisyVecs, noisyImages, noises


def gen_grid_vis(original_image, first_image, ordered_images, num_trials, experimentNum):
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
		
	new_im.save("./exp" + str(experimentNum) + "/data-" + str(experimentNum)+".png")

def pixel_error(image1, image2):
	difference = image1 - image2
	error = np.linalg.norm(difference)
	return error

def delete_helper(delete_array, index):
	cur_sum = 0
	for i, x in enumerate(delete_array):
		cur_sum += x
		if cur_sum == index:
			return i 


def run(experimentNum, num_trials = 20, learning_rate = 15, noise = 0.99, alpha = 0.99):

	os.mkdir("exp" + str(experimentNum))

	#Generate and save original image that you will 
	seed = 3242
	np.random.seed(19238817)

	original_z = random_vector()

	print("Generating Image to Reconstruct ... ")

	o_image = z_sample(Gs, original_z)
	imsave("./" + "exp" + str(experimentNum) + "/original.png", o_image)
	plt.imshow(o_image)
	plt.grid('off')
	plt.axis('off')
	plt.draw()
	plt.pause(0.001)

	#Keep track of experimental data
	z_vectors = [] # z vector after each iteration
	error_vals = [] # pixel error w.r.t original image after each iteration
	total_grid = []

	np.random.seed(1923438817)

	cur_z = random_vector()
	
	print("Reconstructed Image: ", 0)
	r_image = z_sample(Gs, cur_z)
	first_image = r_image
	imsave("./exp" + str(experimentNum) + "/reconstructed_"  +str(1)+".png", r_image)
	error_vals.append(pixel_error(r_image, o_image))
	plt.imshow(r_image)
	plt.draw()
	plt.pause(0.001)

	for exp_iter in range(1,num_trials + 1):

		print("Generating noise level options - Least Noise (1): Images 1-3, Middle Noise (2): Images 4-6, High Noise (3): Images 7-9")
		present_noise_choices(cur_z, exp_iter,experimentNum)
		print("Input integer between 1 (least noise) - 3 (most noise) for desired noise level")
		raw_noise_level = input()

		
		if int(raw_noise_level) == 1:
			noisyVecs, noisyImages, noises = gen_grid_exp(cur_z, exp_iter,experimentNum, o_image, 0.5)
		elif int(raw_noise_level) == 2:
			noisyVecs, noisyImages, noises = gen_grid_exp(cur_z, exp_iter, experimentNum, o_image, 1.3)
		else:
			noisyVecs, noisyImages, noises = gen_grid_exp(cur_z, exp_iter, experimentNum, o_image, 8)
		temp_grid =  [0] * 6 

		copyNoisyImages = list(noisyImages)
		raw_rankings = [0,0,0,0,0,0]
		deleted_array = [1,1,1,1,1,1]
		for rank in range(6,0,-1):
			print("Input index of image with highest similarity to original image beginning with index 1")
			best_image_index = int(input())
			raw_rankings[delete_helper(deleted_array, best_image_index)] = rank
			deleted_array[delete_helper(deleted_array, best_image_index)] = 0
			clear_output()
			if rank != 1:
				gen_images_to_rank(noisyImages, o_image, best_image_index)

		print(raw_rankings)
		rankings = np.array(raw_rankings)
		#for visualization purposes 
		for i,r in enumerate(rankings):
			temp_grid[r - 1] = copyNoisyImages[i]
		total_grid += temp_grid

		rankings = (rankings - rankings.mean())/rankings.std()
		noisyVecsSum = np.zeros(512).reshape(1,512)
		for i,r in enumerate(rankings):
			noisyVecsSum += r * (noises[i])

		#update step 
		learning_rate = (alpha**(exp_iter/2))
		cur_z = cur_z + learning_rate * (noisyVecsSum/(6 * noise))


		print("Reconstructed Image", exp_iter)
		r_image = z_sample(Gs,cur_z)
		total_grid.append(r_image)
		z_vectors.append(cur_z)
		imsave("./exp" + str(experimentNum) + "/reconstructed_"  +str(exp_iter + 1)+".png", r_image)
		error_vals.append(pixel_error(r_image, o_image))
		plt.imshow(r_image)
		plt.draw()
		plt.grid('off')
		plt.axis('off')
		plt.pause(0.001)


	print("Experiment Complete!")    
	gen_grid_vis(o_image, first_image, total_grid, num_trials, experimentNum)

if __name__ == "__main__":
	run(12)