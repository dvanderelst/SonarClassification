import numpy
import pandas
import os
from matplotlib import pyplot
from tensorflow import keras

import misc
import process_functions
import settings

data_set = 'royal'

n_components = 30
generate_data = False
do_training = True
layers = [50, 100, 100, 50]
nr_epochs = 1000

for selected_dimension in ['lcs', 'azs','els']:
    file_names = misc.folder_names(data_set, selected_dimension)
    if generate_data: process_functions.process_data_set(data_set)
    if do_training:
        # Read prepared data
        data = numpy.load(file_names['npz_file'])
        pca = misc.pickle_load(file_names['pca_file'])

        # Select inputs
        if selected_dimension == 'lcs': unencoded_data = data['long_lcs']
        if selected_dimension == 'azs': unencoded_data = data['long_azs']
        if selected_dimension == 'els': unencoded_data = data['long_els']
        if selected_dimension == 'lcs' and data_set in ['israel', 'royal']: unencoded_data = misc.map_lcs_to_distances(data)

        # Encode targets
        encoder, targets = process_functions.get_encoding(unencoded_data)
        target_n = targets.shape[1]

        # Get PCA-ed template inputs
        cummulative_explained_variance = numpy.cumsum(pca.explained_variance_ratio_)
        templates = data['long_data']
        pca_templates = pca.transform(templates)
        inputs = pca_templates[:, :n_components]

        # Scale inputs to a minimum of zero
        inputs = inputs - numpy.min(inputs)

        # Make model
        model = keras.Sequential()
        noise_layer = keras.layers.GaussianNoise(input_shape=(n_components,), stddev=settings.stochaistic_noise * 1)
        output_layer = keras.layers.Dense(target_n, activation='softmax')
        model.add(noise_layer)
        for nodes in layers: model.add(keras.layers.Dense(nodes, activation='relu'))
        model.add(output_layer)

        # Train Model
        loss = keras.losses.CategoricalCrossentropy()
        model.compile('adam', loss=loss)
        training_history = model.fit(inputs, targets, epochs=nr_epochs)
        model.save(file_names['model_file'])

        predictions_matrix = model.predict(inputs)
        binary_predictions = misc.binarize_prediction(predictions_matrix)
        interpreted_predictions = encoder.inverse_transform(binary_predictions)
        interpreted_predictions = interpreted_predictions.flatten()

        results = {'target': unencoded_data, 'prediction': interpreted_predictions}
        results['dummy'] = 1
        results = pandas.DataFrame(results)
        misc.pickle_save(file_names['results_file'], results)

        # Save history
        misc.pickle_save(file_names['history_file'], training_history.history)
        # Plot history
        pyplot.plot(training_history.history['loss'])
        pyplot.title(data_set + ' ' + selected_dimension)
        pyplot.savefig(os.path.join(file_names['result_folder'],'plots',file_names['base_name'] + '_trace.png'))
        pyplot.show()

        # Plot errors
        errors = interpreted_predictions - unencoded_data
        pyplot.hist(errors, 100)
        pyplot.title(data_set + ' ' + selected_dimension)
        pyplot.savefig(os.path.join(file_names['result_folder'],'plots', file_names['base_name'] + '_errrors.png'))
        pyplot.show()

        # Get confusion table and plot it
        table, labels = misc.make_confusion_matrix(results)
        pyplot.matshow(table)
        pyplot.colorbar()
        pyplot.title(data_set + ' ' + selected_dimension)
        pyplot.savefig(os.path.join(file_names['result_folder'],'plots', file_names['base_name'] + '_matrix.png'))
        pyplot.show()

# %%
# keras.utils.plot_model(
#     model,
#     to_file='model.png',
#     show_shapes=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=True,
#     dpi=96
# )
#

# noise = numpy.random.normal(0, settings.stochaistic_noise, (100000,templates.shape[1]))
# transformed = pca.transform(noise)
# transformed = transformed[:,:n_components]
# stdv = numpy.std(transformed, axis=0)
# print(print(stdv)