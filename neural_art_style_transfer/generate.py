__author__ = 'mateuszopala'

# download pretrained weights
# wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl


import lasagne
import numpy as np
import pickle
import scipy
from lasagne.utils import floatX
import matplotlib
import theano.sandbox.neighbours
from utils import prep_image
from neural_art_style_transfer.models import *

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_size = 600
    mean = np.array([104, 117, 123]).reshape((3, 1, 1))
    content_path = '../data/rothbard.jpg'
    style_path = '../data/matisse.jpg'
    weights_path = '../data/vgg19_normalized.pkl'
    net = build_model(img_size)
    with open(weights_path) as f:
        values = pickle.load(f)['param values']
    lasagne.layers.set_all_param_values(net['pool5'], values)

    layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layers = {k: net[k] for k in layers}

    photo = plt.imread(content_path)
    raw_img, photo = prep_image(photo, img_size, mean)
    plt.imsave('../data/raw_0.jpg', raw_img)

    art = plt.imread(style_path)
    raw_img, art = prep_image(art, img_size, mean)
    plt.imsave('../data/raw_1.jpg', raw_img)

    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

    photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                      for k, output in zip(layers.keys(), outputs)}
    art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                    for k, output in zip(layers.keys(), outputs)}

    # Get expressions for layer activations for generated image
    generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, img_size, img_size))))

    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

    # Define loss function
    losses = \
        [0.001 * content_loss(photo_features, gen_features, 'conv4_2'),
         0.2e6 * style_loss(art_features, gen_features, 'conv1_1'),
         0.2e6 * style_loss(art_features, gen_features, 'conv2_1'),
         0.2e6 * style_loss(art_features, gen_features, 'conv3_1'),
         0.2e6 * style_loss(art_features, gen_features, 'conv4_1'),
         0.2e6 * style_loss(art_features, gen_features, 'conv5_1'), 0.1e-7 * total_variation_loss(generated_image)]

    # content loss

    # style loss

    # total variation penalty

    total_loss = sum(losses)

    grad = T.grad(total_loss, generated_image)

    # Theano functions to evaluate loss and gradient
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)

    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        x0 = floatX(x0.reshape((1, 3, img_size, img_size)))
        generated_image.set_value(x0)
        return f_loss().astype('float64')


    def eval_grad(x0):
        x0 = floatX(x0.reshape((1, 3, img_size, img_size)))
        generated_image.set_value(x0)
        return np.array(f_grad()).flatten().astype('float64')


    # Initialize with a noise image
    generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, img_size, img_size))))

    x0 = generated_image.get_value().astype('float64')
    xs = []
    xs.append(x0)

    # Optimize, saving the result periodically
    for i in xrange(20):
        print(i)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
        x0 = generated_image.get_value().astype('float64')
        xs.append(x0)


    def deprocess(x):
        x = np.copy(x[0])
        x += mean

        x = x[::-1]
        x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

        x = np.clip(x, 0, 255).astype('uint8')
        return x


    for i in xrange(21):
        plt.imsave('../data/figure_%d.jpg' % i, deprocess(xs[i]))

    plt.imsave('../data/figure.jpg', deprocess(xs[-1]))
