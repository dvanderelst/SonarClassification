import numpy
import os
from matplotlib import pyplot
from sklearn.decomposition import PCA

from library import misc, process_functions, settings

generate_data_israel = False
generate_data_royal = False
generate_data_andrews = False

pyplot.style.use(settings.style)
pca_plot_file = os.path.join(settings.figure_folder, 'pca.pdf')

if generate_data_israel: process_functions.process_data_set('israel')
if generate_data_royal: process_functions.process_data_set('royal')
if generate_data_andrews: process_functions.process_data_set('andrews')

data_set = 'israel'
file_names = misc.folder_names(data_set, None)
data = numpy.load(file_names['npz_file'])
templates_isreael = data['long_data']

data_set = 'royal'
file_names = misc.folder_names(data_set, None)
data = numpy.load(file_names['npz_file'])
templates_royal = data['long_data']

data_set = 'andrews'
file_names = misc.folder_names(data_set, None)
data = numpy.load(file_names['npz_file'])
templates_andrews = data['long_data']

all_templates = numpy.row_stack((templates_royal, templates_isreael))
print('---> RUN AND SAVE PCA MODEL')
pca_model = PCA(n_components=settings.n_components)
pca_model.fit(all_templates)
misc.pickle_save(settings.pca_file, pca_model)


transformed = pca_model.transform(all_templates)
reconstructed = pca_model.inverse_transform(transformed)
ratio = numpy.cumsum(pca_model.explained_variance_ratio_)

x = reconstructed.flatten()
y = all_templates.flatten()
r0 = numpy.corrcoef(x, y)[0, 1]

# map andrews data

transformed_andrews = pca_model.transform(templates_andrews)
reconstructed_andrews = pca_model.inverse_transform(transformed_andrews)

x = reconstructed_andrews.flatten()
y = templates_andrews.flatten()
r1 = numpy.corrcoef(x, y)[0, 1]

print(r0 ** 2, r1 ** 2)

# %% plots
# Plot Explained variance
pyplot.plot(ratio)
pyplot.xlabel('Nr of PCs')
pyplot.ylabel('Proportion of explained variance')
pyplot.tight_layout()
pyplot.savefig(pca_plot_file)
pyplot.show()

# Plot components
components = pca_model.components_
for i in range(settings.n_components):
    title = "$PC_{%i}$" % (i + 1)
    pyplot.subplot(4, 5, i + 1)
    pyplot.plot(components[i, :])
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.title(title)

pyplot.tight_layout()
pyplot.savefig('results/PCA.pdf')
pyplot.show()
