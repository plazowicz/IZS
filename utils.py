import json
import os
import theano
import theano.tensor as T
from lasagne.utils import floatX
import skimage.transform
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as GPU_RandomStreams
import numpy as np
import h5py


def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_json(json_path):
    with open(json_path, 'r') as f:
        content = json.load(f)
    return content


def write_json(obj, json_path):
    with open(json_path, 'r') as f:
        json.dump(obj, f, indent=4)


class Sampler(object):
    def __init__(self, mode='gpu', seed=None):
        seed = seed or 69
        if mode == 'gpu':
            self.t_rng = GPU_RandomStreams(seed)
        elif mode == 'cpu':
            self.t_rng = RandomStreams(seed)
        else:
            raise NotImplementedError('Mode has to be either gpu or cpu')

    def sample_gaussian_state(self, mean, std, size):
        return self.t_rng.normal(size=size, avg=mean, std=std, dtype=theano.config.floatX)

    def sample_binary_state(self, p, n=1):
        return self.t_rng.binomial(size=p.shape, n=1, p=p, dtype=theano.config.floatX)


try:
    import PIL.Image as Image
except ImportError:
    import Image


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
        ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def save_parameters(parameters_dict, parameters_path):
    with h5py.File(parameters_path, 'w') as f:
        for key, value in parameters_dict.iteritems():
            f.create_dataset(key, data=value)


def load_parameters(parameters_path):
    with h5py.File(parameters_path, 'r') as f:
        parameters_dict = {k: v[:] for k, v in f.iteritems()}
    return parameters_dict


def prep_image(img, img_size, mean):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
    h, w, _ = img.shape
    if h < w:
        img = skimage.transform.resize(img, (img_size, w * img_size / h), preserve_range=True)
    else:
        img = skimage.transform.resize(img, (h * img_size / w, img_size), preserve_range=True)

    # Central crop
    h, w, _ = img.shape
    img = img[h // 2 - img_size // 2:h // 2 + img_size // 2, w // 2 - img_size // 2:w // 2 + img_size // 2]

    raw_img = np.copy(img).astype('uint8')

    # Shuffle axes to c01
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

    # Convert RGB to BGR
    img = img[::-1, :, :]

    img = img - mean
    return raw_img, floatX(img[np.newaxis])