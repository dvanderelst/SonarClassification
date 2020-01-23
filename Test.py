import process_functions
import numpy
from matplotlib import pyplot
import settings
from tensorflow import keras

n_components = 25
generated_data = False

if generated_data: process_functions.process_data_set('andrews')

data = numpy.load('andrews.npz')
pca = process_functions.pickle_load('andrews.pck')
loc_encoder, loc_targets = process_functions.get_encoding(data['long_locs'])
azs_encoder, azs_targets = process_functions.get_encoding(data['long_azs'])
els_encoder, els_targets = process_functions.get_encoding(data['long_els'])

cummulative_explained_variance = numpy.cumsum(pca.explained_variance_ratio_)

templates = data['long_data']
pca_templates = pca.transform(templates)
inputs = pca_templates[:,:n_components]

input_n = inputs.shape[1]
output_n = loc_targets.shape[1]

l1 = keras.layers.Dense(n_components * 2, input_shape=(input_n,))
l2 = keras.layers.Dense(n_components * 5, activation='relu')
l3 = keras.layers.Dense(n_components * 5, activation='relu')
l4 = keras.layers.Dense(n_components * 2, activation='relu')
l5 = keras.layers.Dense(output_n, activation='softmax')

mn = numpy.min(inputs)
inputs = inputs - mn
mx = numpy.max(inputs)
inputs = inputs / mx

loss = keras.losses.CategoricalCrossentropy()

model = keras.Sequential([l1, l2, l3, l4, l5])
model.compile('adam', loss=loss)

model.fit(inputs, loc_targets, epochs=1000)
predictions = model.predict(inputs)

#%%
keras.utils.plot_model(
    model,
    to_file='model.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=True,
    dpi=96
)
