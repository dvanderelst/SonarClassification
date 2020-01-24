import numpy
import pandas
from matplotlib import pyplot
from tensorflow import keras

import misc
import process_functions
import settings

data_set = 'andrews'
selected_dimension = 'azs'
n_components = 30
generate_data = True

file_names = misc.folder_names(data_set, selected_dimension)

if generate_data: process_functions.process_data_set(data_set)
# Read prepared data
data = numpy.load(file_names['npz_file'])
pca = process_functions.pickle_load(file_names['pca_file'])

# Encode targets
lcs_encoder, lcs_targets = process_functions.get_encoding(data['long_lcs'])
azs_encoder, azs_targets = process_functions.get_encoding(data['long_azs'])
els_encoder, els_targets = process_functions.get_encoding(data['long_els'])

# Get PCA-ed inputs
cummulative_explained_variance = numpy.cumsum(pca.explained_variance_ratio_)
templates = data['long_data']
pca_templates = pca.transform(templates)
inputs = pca_templates[:, :n_components]

# Scale inputs
mn = numpy.min(inputs)
inputs = inputs - mn

# Select inputs
if selected_dimension == 'lcs':
    unencoded_data = data['long_lcs']
    targets = lcs_targets
    selected_encoder = lcs_encoder

if selected_dimension == 'azs':
    unencoded_data = data['long_azs']
    targets = azs_targets
    selected_encoder = azs_encoder

if selected_dimension == 'els':
    unencoded_data = data['long_els']
    targets = els_targets
    selected_encoder = els_encoder

target_n = targets.shape[1]

layers = [25, 50, 100, 50, 25]

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
model.fit(inputs, targets, epochs=1000)
model.save(file_names['model'])

predictions_matrix = model.predict(inputs)
binary_predictions = misc.binarize_prediction(predictions_matrix)
interpreted_predictions = selected_encoder.inverse_transform(binary_predictions)
interpreted_predictions = interpreted_predictions.flatten()

results = {'target': unencoded_data, 'prediction': interpreted_predictions}
results['dummy'] = 1
results = pandas.DataFrame(results)

# Make confusion matrix and plot it
grp = results.groupby(['target', 'prediction'])
counts = grp.sum()
counts = counts.reset_index()
table = counts.pivot(index='target', columns='prediction', values='dummy')
table[numpy.isnan(table)] = 0

pyplot.matshow(table)
pyplot.colorbar()
pyplot.show()

# Error histogram
errors = unencoded_data - interpreted_predictions
pyplot.hist(errors, 100)
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
# print(print(stdv))
