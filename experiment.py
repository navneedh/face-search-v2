import tensorflow as tf
import imageio 
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

url = 'http://cocosci.princeton.edu/jpeterson/temp_file_hosting/263e666dc20e26dcbfa514733c1d1f81_karras2019stylegan-ffhq-1024x1024.pkl' # karras2019stylegan-ffhq-1024x1024.pkl
# https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
	_G, _D, Gs = pickle.load(f)
	# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
	# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
	# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.


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

white_image = z_sample(Gs, random_vector())
white_image.fill(255)

def gen_grid_exp(cur_z, exp_iter, experimentNum, original, cur_reconstructed_image, noise_level = 1):
	noise = 0.99
	seed = np.random.randint(4000)
	np.random.seed(seed)
	noisyVecs = []
	noises = []
	noisyImages = []
	imagesToDisplay = []			
	# new_im = Image.new('RGB', (1152,128))
	white_image_fixed = np.array(Image.fromarray(white_image).resize(size = (256,256), resample = False))
	index = 0
	print("Generating grid of noisy images ...")
	print("      1                     2                    3                    4                    5                    6                        Reconstructed   Original")
	for i in range(0,768,128):
		np.random.seed(np.random.randint(4362634))
		noise_val = (random_vector() * noise_level) #most noise added
		zs = cur_z + noise_val
		zs = np.clip(zs, -5, 5)
		p_image = z_sample(Gs, zs)
		imagesToDisplay.append(np.array(Image.fromarray(p_image).resize(size = (256,256), resample = False)))
		noises.append(noise_val)
		noisyVecs.append(zs)
		noisyImages.append(p_image)
		# im = Image.fromarray(p_image)
		# im.thumbnail((128,128))
		# new_im.paste(im, (i,0))		

	#add blank image between proposals and original
	imagesToDisplay.append(white_image_fixed)
	# im = Image.fromarray(white_image)
	# noisyImages.append(white_image)
	# im.thumbnail((128,128))
	# new_im.paste(im, (768,0))

	#add current reconstructed image to grid 
	imagesToDisplay.append(np.array(Image.fromarray(cur_reconstructed_image).resize(size = (256,256), resample = False)))	
	# im = Image.fromarray(cur_reconstructed_image)
	noisyImages.append(cur_reconstructed_image)
	# im.thumbnail((128,128))
	# new_im.paste(im, (896,0))

	#add original image to grid
	imagesToDisplay.append(np.array(Image.fromarray(original).resize(size = (256,256), resample = False)))
	# im = Image.fromarray(original)
	noisyImages.append(original)
	# im.thumbnail((128,128))
	# new_im.paste(im, (1024,0))

	image_grid = np.hstack(imagesToDisplay)
		
	# new_im.save("./exp" + str(experimentNum) + "/grid_" +str(exp_iter)+".png")
	# display(Imdisplay(filename = "./exp" + str(experimentNum) + "/grid_" +str(exp_iter)+".png", width=1000, unconfined=True))
	plt.figure(figsize=(25,50))
	plt.grid(False)
	plt.axis("off")
	plt.imshow(image_grid)
	# plt.imshow(new_im)
	plt.draw()
	plt.pause(0.001)
	return noisyVecs, noisyImages, noises

def gen_images_to_rank(image_matrices, original, indexToRemove, iteration):
	white_image_fixed = np.array(Image.fromarray(white_image).resize(size = (256,256), resample = False))
	imagesToDisplay = []
	# new_im = Image.new('RGB', ((len(image_matrices) - 1) * 128,128))
	del image_matrices[indexToRemove - 1]
	# for i in range(0,len(image_matrices) * 128,128):
	# 	im = Image.fromarray(image_matrices[int(i/128)])
	# 	im.thumbnail((128,128))
	# 	new_im.paste(im, (i,0))

	imagesToDisplay = image_matrices
	imagesToDisplay.append(white_image_fixed)
	imagesToDisplay.append(np.array(Image.fromarray(original).resize(size = (256,256), resample = False)))
	
	# im = Image.fromarray(original)
	# im.thumbnail((128,128))
	# new_im.paste(im, ((len(image_matrices) + 1) * 128,0))

	# new_im.save("temp_save.png")
	# display(Imdisplay(filename = "temp_save.png", width=1000 - (iteration * 128), unconfined=True))
	# plt.imshow(new_im)

	image_grid = np.hstack(imagesToDisplay)

	plt.figure(figsize=(25,50))
	plt.grid(False)
	plt.axis("off")
	plt.imshow(image_grid)
	# plt.imshow(new_im)
	plt.draw()
	plt.pause(0.001)



def present_noise_choices(cur_z, exp_iter, experimentNum, original, cur_reconstructed_image, noise_level = 1):
	noise = 0.99
	seed = np.random.randint(4000)
	np.random.seed(seed)
	noisyVecs = []
	noises = []
	imagesToDisplay = []
	noisyImages = []
	# new_im = Image.new('RGB', (1792,128))
	white_image_fixed = np.array(Image.fromarray(white_image).resize(size = (256,256), resample = False))
	index = 0
	print(" Low Noise                                         Medium Noise                                        High Noise                                        Reconstructed   Original")
	for i in range(0,384,128):
		np.random.seed(np.random.randint(4362634))
		noise_val = (random_vector() * 0.85) #least noise added
		zs = cur_z + noise_val
		zs = np.clip(zs, -5, 5)
		p_image = z_sample(Gs, zs)
		imagesToDisplay.append(np.array(Image.fromarray(p_image).resize(size = (256,256), resample = False)))
		noises.append(noise_val)
		noisyVecs.append(zs)
		noisyImages.append(p_image)
		# im = Image.fromarray(p_image)
		# im.thumbnail((128,128))
		# new_im.paste(im, (i,0))

	imagesToDisplay.append(white_image_fixed)
	# im = Image.fromarray(white_image)
	# im.thumbnail((128,128))
	# new_im.paste(im, (384,0))

	for i in range(512,896,128):
		np.random.seed(np.random.randint(4362634))
		noise_val = (random_vector() * 3.2) #most noise added
		zs = cur_z + noise_val
		zs = np.clip(zs, -5, 5)
		p_image = z_sample(Gs, zs)
		imagesToDisplay.append(np.array(Image.fromarray(p_image).resize(size = (256,256), resample = False)))
		noises.append(noise_val)
		noisyVecs.append(zs)
		noisyImages.append(p_image)
		# im = Image.fromarray(p_image)
		# im.thumbnail((128,128))
		# new_im.paste(im, (i,0))

	imagesToDisplay.append(white_image_fixed)
	# im = Image.fromarray(white_image)
	# im.thumbnail((128,128))
	# new_im.paste(im, (896,0))

	for i in range(1024,1408,128):
		np.random.seed(np.random.randint(4362634))
		noise_val = (random_vector() * 7) #most noise added
		zs = cur_z + noise_val
		zs = np.clip(zs, -5, 5)
		p_image = z_sample(Gs, zs)
		imagesToDisplay.append(np.array(Image.fromarray(p_image).resize(size = (256,256), resample = False )))
		noises.append(noise_val)
		noisyVecs.append(zs)
		noisyImages.append(p_image)
		# im = Image.fromarray(p_image)
		# im.thumbnail((128,128))
		# new_im.paste(im, (i,0))


	#add blank image between proposals and original
	imagesToDisplay.append(white_image_fixed)
	# im = Image.fromarray(white_image)
	noisyImages.append(white_image)
	# im.thumbnail((128,128))
	# new_im.paste(im, (1408,0))

	#add current reconstructed image to grid 	
	imagesToDisplay.append(np.array(Image.fromarray(cur_reconstructed_image).resize(size = (256,256), resample = False)))
	# im = Image.fromarray(cur_reconstructed_image)
	noisyImages.append(cur_reconstructed_image)
	# im.thumbnail((128,128))
	# new_im.paste(im, (1536,0))

	#add original image to grid
	imagesToDisplay.append(np.array(Image.fromarray(original).resize(size = (256,256), resample = False)))
	# im = Image.fromarray(original)
	noisyImages.append(original)
	# im.thumbnail((128,128))
	# new_im.paste(im, (1664,0))

	image_grid = np.hstack(imagesToDisplay)


		
	# new_im.save("noise_choices.png")
	# display(Imdisplay(filename = "noise_choices.png", width=1500, unconfined=True))
	# # plt.imshow(new_im)
	plt.figure(figsize=(25,50))
	plt.grid(False)
	plt.axis('off')
	plt.imshow(image_grid)
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
	imageio.imwrite("./" + "exp" + str(experimentNum) + "/original.png", o_image)
	# plt.imshow(o_image)
	# plt.grid('off')
	# plt.axis('off')
	# plt.draw()
	# plt.pause(0.001)

	#Keep track of experimental data
	z_vectors = [] # z vector after each iteration
	error_vals = [] # pixel error w.r.t original image after each iteration
	total_grid = []

	np.random.seed(1923438817)

	cur_z = random_vector()
	
	# print("Reconstructed Image: ", 0)
	r_image = z_sample(Gs, cur_z)
	first_image = r_image
	imageio.imwrite("./exp" + str(experimentNum) + "/reconstructed_"  +str(1)+".png", r_image)
	# error_vals.append(pixel_error(r_image, o_image))
	# plt.imshow(r_image)
	# plt.grid('off')
	# plt.axis('off')
	# plt.draw()
	# plt.pause(0.001)

	clear_output()

	for exp_iter in range(1,num_trials + 1):
		print("ITERATION #", exp_iter)
		print("Generating noise level options ... ") 
		present_noise_choices(cur_z, exp_iter,experimentNum, o_image, r_image)
		print("Input integer between 1 (least noise) - 3 (most noise) for desired noise level")
		
		raw_noise_level = input()

		clear_output()

		# present noisy proposals 
		if int(raw_noise_level) == 1:
			noisyVecs, noisyImages, noises = gen_grid_exp(cur_z, exp_iter,experimentNum, o_image, r_image, 0.85)
		elif int(raw_noise_level) == 2:
			noisyVecs, noisyImages, noises = gen_grid_exp(cur_z, exp_iter, experimentNum, o_image, r_image, 3.2)
		else:
			noisyVecs, noisyImages, noises = gen_grid_exp(cur_z, exp_iter, experimentNum, o_image, r_image, 7)
		temp_grid =  [0] * 6 

		copyNoisyImages = list(noisyImages)
		raw_rankings = [0,0,0,0,0,0]
		deleted_array = [1,1,1,1,1,1]
		for rank in range(6,0,-1):
			print("Input index of image with highest similarity to original image beginning with index 1")
			try:
				best_image_index = int(input())
				raw_rankings[delete_helper(deleted_array, best_image_index)] = rank
				deleted_array[delete_helper(deleted_array, best_image_index)] = 0

			except:
				print("Please enter a valid image index between 1 and ", rank)
				best_image_index = int(input())
				raw_rankings[delete_helper(deleted_array, best_image_index)] = rank
				deleted_array[delete_helper(deleted_array, best_image_index)] = 0

			clear_output()
			if rank != 1:
				gen_images_to_rank(noisyImages, o_image, best_image_index, 6 - rank + 1)

		rankings = np.array(raw_rankings)
		# #for visualization purposes 
		# for i,r in enumerate(rankings):
		# 	temp_grid[r - 1] = copyNoisyImages[i]
		# total_grid += temp_grid

		rankings = (rankings - rankings.mean())/rankings.std()
		noisyVecsSum = np.zeros(512).reshape(1,512)
		for i,r in enumerate(rankings):
			temp_grid[int(r) - 1] = copyNoisyImages[i]
			noisyVecsSum += r * (noises[i])
		total_grid += temp_grid

		#update step 
		learning_rate = (alpha**(exp_iter/2))
		cur_z = cur_z + learning_rate * (noisyVecsSum/(6 * noise))

		# print("Original Image")
		# o_image = z_sample(Gs, original_z)
		# imageio.imwrite("./" + "exp" + str(experimentNum) + "/original.png", o_image)
		# plt.imshow(o_image)
		# plt.grid('off')
		# plt.axis('off')
		# plt.draw()
		# plt.pause(0.001)


		# print("Reconstructed Image #", exp_iter + 1)
		r_image = z_sample(Gs,cur_z)
		total_grid.append(r_image)
		z_vectors.append(cur_z)
		imageio.imwrite("./exp" + str(experimentNum) + "/reconstructed_"  +str(exp_iter + 1)+".png", r_image)
		# error_vals.append(pixel_error(r_image, o_image))
		# plt.imshow(r_image)
		# plt.draw()
		# plt.grid('off')
		# plt.axis('off')
		# plt.pause(0.001)


	print("Experiment Complete!")    
	gen_grid_vis(o_image, first_image, total_grid, num_trials, experimentNum)

if __name__ == "__main__":
	run(12)