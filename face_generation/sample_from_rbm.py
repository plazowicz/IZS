import numpy as np
from face_generation.rbm import BinaryRBM
from utils import load_parameters
import theano
from face_generation.loaders import LFWLoaderInMemory
from PIL import Image
from utils import tile_raster_images


if __name__ == "__main__":
    lfw_path = '../data/lfw_grayscaled_64_64'
    output_path = '../data/samples.png'
    data_loader = LFWLoaderInMemory(lfw_path)
    nb_visible = 64 ** 2
    nb_hidden = 1024
    n_samples = 10
    img_size = 64
    lr = 0.01
    test_data = np.random.random_integers(0, 255, size=(16, 64 ** 2))
    test_data = data_loader.normalise(test_data)
    parameters_path = '../data/parameters_200_epochs.hdf5'
    rbm = BinaryRBM(nb_visible, nb_hidden, learning_rate=lr, momentum_factor=0, persistent=True, batch_size=128, k=20)
    parameters_dict = load_parameters(parameters_path)
    rbm.parameters = parameters_dict

    # find out the number of test samples
    number_of_test_samples = test_data.shape[0]

    # pick random test examples, with which to initialize the persistent chain
    n_chains = test_data.shape[0]

    persistent_vis_chain = theano.shared(
        np.asarray(
            test_data,
            dtype=theano.config.floatX
        )
    )
    # end-snippet-6 start-snippet-7
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name="gibbs_vhv"
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = np.zeros(
        ((img_size + 1) * n_samples + 1, (img_size + 1) * n_chains - 1),
        dtype='uint8'
    )
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[(img_size + 1) * idx:(img_size + 1) * idx + img_size, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(img_size, img_size),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save(output_path)
