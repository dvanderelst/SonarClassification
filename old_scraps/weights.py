from library import misc

# royal_components are in the rows
# components_array, shape (n_components, n_features)



data_set = 'royal'
selected_dimension = 'lcs'
file_names = misc.folder_names(data_set, selected_dimension)

pca = misc.pickle_load(file_names['pca_file'])

components = pca.components_

#new_model = keras.models.load_model(file_names['model_file'])
#weights = new_model.get_weights()
#%%
#x = weights[4]
#x = numpy.abs(x)
#pyplot.imshow(x)
#pyplot.show()