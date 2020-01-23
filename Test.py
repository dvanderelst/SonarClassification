import process_functions
import numpy
from matplotlib import pyplot
import settings
import misc
import pandas
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb

n_components = 30
generated_data = False
selected_dimension = 'els'

if generated_data: process_functions.process_data_set('andrews')

# Read prepared data
data = numpy.load('andrews.npz')
pca = process_functions.pickle_load('andrews.pck')

# Encode targets
lcs_encoder, lcs_targets = process_functions.get_encoding(data['long_lcs'])
azs_encoder, azs_targets = process_functions.get_encoding(data['long_azs'])
els_encoder, els_targets = process_functions.get_encoding(data['long_els'])

# Get PCA-ed inputs
cummulative_explained_variance = numpy.cumsum(pca.explained_variance_ratio_)
templates = data['long_data']
pca_templates = pca.transform(templates)
inputs = pca_templates[:,:n_components]

# Scale inputs
mn = numpy.min(inputs)
inputs = inputs - mn

# Select inputs
if selected_dimension == 'locs':
    unencoded_data = data['long_data']
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

loss = keras.losses.CategoricalCrossentropy()
model.compile('adam', loss=loss)
model.fit(inputs, targets, epochs=750)

predictions_matrix = model.predict(inputs)
binary_predictions = misc.binarize_prediction(predictions_matrix)
interpreted_predictions = selected_encoder.inverse_transform(binary_predictions)
interpreted_predictions = interpreted_predictions.flatten()

results = {'target':unencoded_data, 'prediction':interpreted_predictions}
results['dummy'] = 1
results = pandas.DataFrame(results)

grp = results.groupby(['target','prediction'])
counts = grp.sum()
counts = counts.reset_index()
table = counts.pivot(index='target', columns='prediction', values='dummy')

pyplot.matshow(table)
pyplot.colorbar()
pyplot.show()

# Select inputs
e = (unencoded_data - interpreted_predictions)
pyplot.hist(e, 100)
pyplot.show()

#%%
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

noise = numpy.random.normal(0, settings.stochaistic_noise, (100000,templates.shape[1]))
transformed = pca.transform(noise)
transformed = transformed[:,:n_components]
stdv = numpy.std(transformed, axis=0)
print(print(stdv))