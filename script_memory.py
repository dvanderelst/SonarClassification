import numpy
import pandas
import os
from matplotlib import pyplot
from tensorflow import keras

import misc
import process_functions
import settings

for data_set in ['israel', 'royal']:
    selected_dimension = 'lcs'
    file_names = misc.folder_names(data_set, selected_dimension)
    data = numpy.load(file_names['npz_file'])
    templates = data['long_data']

    total_n_numbers = templates.shape[0] * settings.n_components

    new_model = keras.models.load_model(file_names['model_file'])
    weights = new_model.get_weights()
    total_n_weights = 0
    for w in weights:
        m = numpy.matrix(w)
        total_n_weights = total_n_weights + (m.shape[0] * m.shape[1])

    ratio = total_n_weights / total_n_numbers

    print(data_set)
    print('numbers in templates', total_n_numbers)
    print('numbers of weights', total_n_weights)
    print('ratio', ratio)
