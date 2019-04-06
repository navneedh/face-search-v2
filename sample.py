def random_sample():
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
    print("DOne")
    
def z_sample(z):
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

def gen_grid_exp(cur_z, exp_iter, noise_level = 1):
    noise = 0.99
    seed = np.random.randint(4000)
    np.random.seed(seed)
    noisyVecs = []
    noises = []
    noisyImages = []
    new_im = Image.new('RGB', (384,64))
    index = 0
    print("Generating grid of noisy images ...")
    for i in range(0,384,64):
        np.random.seed(np.random.randint(4362634))
        noise_val = (random_vector() * noise_level) #most noise added
        zs = cur_z + noise_val
        zs = np.clip(zs, -5, 5)
        p_image = z_sample(zs)
        noises.append(noise_val)
        noisyVecs.append(zs)
        noisyImages.append(p_image)
        im = Image.fromarray(p_image)
        im.thumbnail((64,64))
        new_im.paste(im, (i,0))
        index += 1
        
    new_im.save("./exp" + str(experimentNum) + "/grid_" +str(exp_iter)+".png")
    plt.imshow(new_im)
    plt.draw()
    plt.pause(0.001)
    return noisyVecs, noisyImages, noises
