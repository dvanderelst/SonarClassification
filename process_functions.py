import math
import os
import pickle

import natsort
import numpy
import scipy.io as io
from matplotlib import pyplot
from scipy.signal import convolve
from sklearn.decomposition import PCA

import settings


def preprocess(file_name, verbose=True):
    fs = settings.fs
    noise_floor = settings.noise_floor
    integration_time = settings.integration_time
    initial_zero_time = settings.initial_zero_time
    raw_collected_samples = settings.raw_collected_samples

    initial_omission_samples = math.ceil(initial_zero_time * fs)
    integration_samples = math.ceil(fs * integration_time)
    final_samples = math.ceil(raw_collected_samples / integration_samples)

    if verbose:
        print('+-' * 10 + '+')
        print('PREPROCESSING', file_name)
        print('integration samples', integration_samples)
        print('initial zero samples', initial_omission_samples)
        print('final samples', final_samples)

    data = io.loadmat(file_name)
    data = data['Templates']
    templates = data[0, 0][0]
    azimuth = data[0, 0][1]
    elevation = data[0, 0][2]

    processed = numpy.zeros((7, 31, final_samples, 3))

    for i in range(3):
        repetition = templates[:, :, i]
        # Reshape the data
        az_box = azimuth.reshape((31, 7))
        el_box = elevation.reshape((31, 7))
        mn_box = repetition.reshape((31, 7, raw_collected_samples))
        mn_box[:, :, 0:initial_omission_samples] = 0

        az_box = numpy.transpose(az_box)
        el_box = numpy.transpose(el_box)
        mn_box = numpy.transpose(mn_box, axes=[1, 0, 2])

        # Reorder the data in ascending az/el
        col_indices = numpy.argsort(az_box[0, :])
        row_indices = numpy.argsort(el_box[:, 0])

        az_box = az_box[numpy.ix_(row_indices, col_indices)]
        el_box = el_box[numpy.ix_(row_indices, col_indices)]
        mn_box = mn_box[numpy.ix_(row_indices, col_indices)]

        # Average across directions and time
        mask = numpy.ones((3, 3, integration_samples))
        mask = mask / numpy.sum(mask)
        mn_box = convolve(mn_box, mask, mode='same')
        # Subsample
        mn_box = mn_box[:, :, ::integration_samples]
        processed[:, :, :, i] = mn_box
        processed[processed < noise_floor] = noise_floor
    return processed, az_box, el_box


def get_mean_and_variance(data):
    mns = numpy.mean(data, axis=3)
    vrs = numpy.var(data, axis=3)
    vrs = numpy.mean(vrs, axis=(0, 1))
    return mns, vrs



def noise_and_pca(data_set, n_locations):

    # %%Preprocess andrews
    pca_file = data_set + '.pck'
    data_file = data_set + '.npz'

    # Reading in all data
    files = os.listdir(data_set)
    files = natsort.natsorted(files)

    test_data, az, el = preprocess(os.path.join(data_set, files[0]), verbose=False)
    n_samples = test_data.shape[2]

    all_mns = numpy.zeros([n_locations, 7, 31, n_samples])
    all_vrs = numpy.zeros([n_locations, 7, 31, n_samples])

    for i in range(len(files)):
        data, _, _ = preprocess(os.path.join(data_set, files[i]))
        mn, vr = get_mean_and_variance(data)
        all_mns[i, :, :, :] = mn
        all_vrs[i, :, :, :] = vr


    # Reshape data to long format
    long = numpy.reshape(all_mns, (n_locations * 7 * 31, n_samples))

    # Get the variation across all measurements for each sample
    sample_variation = numpy.mean(all_vrs, axis=(0, 1, 2))

    # Save data
    numpy.savez(data_file,
                means=all_mns,
                variation=all_vrs,
                long=long,
                sample_variation = sample_variation,
                files=files)

    # %%
    pca_model = PCA()
    pca_model.fit(long)

    s = pickle.dumps(pca_model)
    f = open(pca_file, 'wb')
    f.write(s)
    f.close()
