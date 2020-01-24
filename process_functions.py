import math
import misc
import pickle
import os
import natsort
import numpy
import scipy.io as io
from scipy.signal import convolve
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import settings

def pickle_save(file, object):
    f = open(file, 'wb')
    pickle.dump(object, f)
    f.close()

def pickle_load(file):
    f = open(file, 'rb')
    object = pickle.load(f)
    f.close()
    return object

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


def process_data_set(data_set, filter_threshold = 0.1):
    # %%Preprocess andrews
    print('#' * 10)
    print(data_set)
    print('#' * 10)

    file_names = misc.folder_names(data_set, 'none')

    pca_file = os.path.join(file_names['pca_file'])
    data_file = os.path.join(file_names['npz_file'])

    # Reading in all data
    print('---> READING AND PREPROCESSING DATA')
    files = os.listdir(data_set)
    files = natsort.natsorted(files)
    n_locations = len(files)

    test_data, az, el = preprocess(os.path.join(data_set, files[0]), verbose=False)
    n_samples = test_data.shape[2]

    all_mns = numpy.zeros([n_locations, 7, 31, n_samples])
    all_vrs = numpy.zeros([n_locations, 7, 31, n_samples])

    for i in range(len(files)):
        data, _, _ = preprocess(os.path.join(data_set, files[i]))

        mns = numpy.mean(data, axis=3)
        vrs = numpy.var(data, axis=3)

        print(data.shape)
        all_mns[i, :, :, :] = mns
        all_vrs[i, :, :, :] = vrs

    # get ID vars
    print('---> DATA2LONG')
    n = numpy.arange(n_locations)
    az_line = az[0, :]
    el_line = el[:, 0]
    locs, azs, els = numpy.meshgrid(n, az_line, el_line)
    locs = numpy.transpose(locs, axes=(1, 2, 0))
    azs = numpy.transpose(azs, axes=(1, 2, 0))
    els = numpy.transpose(els, axes=(1, 2, 0))

    # Reshape data to long format
    long_data = numpy.reshape(all_mns, (n_locations * 7 * 31, n_samples))
    long_lcs = numpy.reshape(locs, (n_locations * 7 * 31))
    long_azs = numpy.reshape(azs, (n_locations * 7 * 31))
    long_els = numpy.reshape(els, (n_locations * 7 * 31))

    id_array = numpy.column_stack((long_lcs, long_azs, long_els))

    # Get the variation across all measurements for each sample
    sample_variance = numpy.mean(all_vrs, axis=(0, 1, 2))

    # Select only those templates above threshold
    summed = numpy.sum(long_data, axis=1)
    summed = numpy.array(summed)
    threshold = numpy.min(summed) + filter_threshold
    include = summed > threshold

    long_data = long_data[include, :]
    id_array = id_array[include, :]
    long_lcs = long_lcs[include]
    long_azs = long_azs[include]
    long_els = long_els[include]

    # Save data
    print('---> SAVING LONG FORMAT')
    numpy.savez(data_file,
                long_data=long_data,
                long_lcs=long_lcs,
                long_azs=long_azs,
                long_els=long_els,
                ids=id_array,
                sample_variance=sample_variance,
                include=include,
                files=files)

    # %%
    print('---> RUN AND SAVE PCA MODEL')
    pca_model = PCA()
    pca_model.fit(long_data)
    pickle_save(pca_file, pca_model)
    print('#' * 10)


def get_encoding(id_variable):
    if id_variable.ndim == 1: id_variable = id_variable.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False, categories='auto' )
    encoder.fit(id_variable)
    y = encoder.fit_transform(id_variable)
    return encoder, y