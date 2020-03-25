import numpy
import pandas
from matplotlib import pyplot
from tensorflow import keras

from library import misc, process_functions, settings

pyplot.figure(figsize=(5, 5))
panel_index = 1

for data_set in ['israel', 'royal']:
    pca_model = misc.pickle_load(settings.pca_templates_model_file)
    for selected_dimension in ['azs', 'els']:
        file_names = misc.folder_names(data_set, selected_dimension)
        # Read prepared data
        data = numpy.load(file_names['npz_file_interpolated'])
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

        results = {'target': unencoded_data, 'prediction': interpreted_predictions}
        results = pandas.DataFrame(results)
        results['dummy'] = 1

        # Get confusion table and plot it
        colormap = settings.cmap
        colormap.set_bad(color='gray')

        pyplot.subplot(2, 2, panel_index)

        ######################################
        table, _ = misc.make_confusion_matrix(results, normalize=False)
        table[-1, :] = numpy.nan
        table[:, -1] = numpy.nan

        ss = numpy.nansum(table)
        sd = numpy.nansum(numpy.diag(table, k=0)) + numpy.nansum(numpy.diag(table, k=1))
        pct = '%.2f' % ((sd / ss) * 100) + '%'
        #######################################

        table, labels = misc.make_confusion_matrix(results, normalize=True)

        # Mask data from positions that were extrapolated
        table[-1, :] = None
        table[:, -1] = None

        pyplot.imshow(table, cmap=colormap, vmin=0, vmax=1)
        misc.label_confusion_matrix(labels, shift_xaxis=5)

        # Hide the gray data
        pyplot.xlim([-0.5, len(labels) - 1.5])
        pyplot.ylim([-0.5, len(labels) - 1.5])

        misc.label(0.9, 0.1, panel_index - 1, color='white')
        misc.label(0.25, 0.9, pct, color='white')
        if panel_index in [1, 3]: pyplot.ylabel('Target (interpolated positions)')
        if panel_index in [3, 4]: pyplot.xlabel('Output (original directions)')

        panel_index = panel_index + 1






pyplot.tight_layout()
pyplot.savefig(settings.interpolation_results_plot)
pyplot.show()
