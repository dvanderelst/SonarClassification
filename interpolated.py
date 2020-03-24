import numpy
import pandas
import os
from matplotlib import pyplot
from tensorflow import keras
from scipy.signal import convolve

from library import misc, process_functions, settings

# for data_set in ['royal', 'israel']:
#     for selected_dimension in ['lcs', 'azs','els']:

for data_set in ['israel']:
    for selected_dimension in ['azs']:
        file_names = misc.folder_names(data_set, selected_dimension)
        # Read prepared data
        data = numpy.load(file_names['npz_file'])
        pca_model = misc.pickle_load(settings.pca_templates_model_file)

        # Select inputs
        if selected_dimension == 'azs': unencoded_data = data['long_azs']
        if selected_dimension == 'els': unencoded_data = data['long_els']
        if selected_dimension == 'lcs' and data_set in ['israel', 'royal']: unencoded_data = misc.map_lcs_to_distances(data)

        # Encode targets
        encoder, targets = process_functions.get_encoding(unencoded_data)
        target_n = targets.shape[1]

        # Get PCA-ed template inputs
        templates = data['long_data']
        inputs = pca_model.transform(templates)
        n_components = inputs.shape[1]

        # Load neural network
        neural_network = keras.models.load_model(file_names['model_file'])

        predictions_matrix = neural_network.predict(inputs)
        binary_predictions = misc.binarize_prediction(predictions_matrix)
        interpreted_predictions = encoder.inverse_transform(binary_predictions)
        interpreted_predictions = interpreted_predictions.flatten()

        #pyplot.scatter(unencoded_data, interpreted_predictions)
        #pyplot.show()
        neural_network.evaluate(inputs, targets)

        results = {'target': unencoded_data, 'prediction': interpreted_predictions}
        results = pandas.DataFrame(results)
        results['dummy'] = 1

        # Get confusion table and plot it
        table, labels = misc.make_confusion_matrix(results)
        pyplot.matshow(table)
        pyplot.colorbar()
        pyplot.title(data_set + ' ' + selected_dimension)
        pyplot.show()

        m = numpy.mean(templates, axis=0)
        pyplot.plot(m)
        pyplot.show()

        m = numpy.mean(inputs, axis=0)
        pyplot.plot(m)
        pyplot.show()




