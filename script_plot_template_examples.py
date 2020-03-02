import numpy
import os
from matplotlib import pyplot
from library import misc, settings

#
# Synthetic echoes, templates and reconstruction
#

mx_echoes = 0.05
mx_templates = 3.5
mn_templates = 0.15

index0 = 10
index1 = 50
index2 = 400

pyplot.style.use(settings.style)

print('-----> LOADING DATA')
loaded = numpy.load(settings.synthetic_echoes_file)
synthetic_echoes = loaded['echoes_synthetic']

loaded = numpy.load(settings.scaled_synthetic_templates_file)
synthetic_templates = loaded['templates_synthetic']

loaded = numpy.load(settings.reconstructed_synthetic_templates_file)
reconstructed_synthetic_templates = loaded['reconstructed_synthetic']

pca_model = misc.pickle_load(settings.pca_file)
cummulative_variance = numpy.cumsum(pca_model.explained_variance_ratio_)

x = synthetic_templates.flatten()
y = reconstructed_synthetic_templates.flatten()

print('-----> GETTING DATA FROM ARRAYS')

echo_0 = synthetic_echoes[index0]
echo_1 = synthetic_echoes[index1]
echo_2 = synthetic_echoes[index2]

template_0 = synthetic_templates[index0, :]
template_1 = synthetic_templates[index1, :]
template_2 = synthetic_templates[index2, :]

reconstructed_template_0 = reconstructed_synthetic_templates[index0, :]
reconstructed_template_1 = reconstructed_synthetic_templates[index1, :]
reconstructed_template_2 = reconstructed_synthetic_templates[index2, :]

print('-----> PLOTTING')


pyplot.figure(figsize=(12, 7))

print('-----> PLOTTING 2')
pyplot.subplot(2, 4, 2)
pyplot.plot(echo_0)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])
pyplot.ylabel('Echo amplitude, model units')

print('-----> PLOTTING 3')
pyplot.subplot(2, 4, 3)
pyplot.plot(echo_1)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])

print('-----> PLOTTING 4')
pyplot.subplot(2, 4, 4)
pyplot.plot(echo_2)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])

print('-----> PLOTTING 6')
pyplot.subplot(2, 4, 6)
pyplot.plot(template_0, linewidth=3)
pyplot.plot(reconstructed_template_0)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])
pyplot.legend(['Template', 'Reconstructed from PCs'])
pyplot.ylabel('Template amplitude, model units')

print('-----> PLOTTING 7')
pyplot.subplot(2, 4, 7)
pyplot.plot(template_1, linewidth=3)
pyplot.plot(reconstructed_template_1)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])

print('-----> PLOTTING 8')
pyplot.subplot(2, 4, 8)
pyplot.plot(template_2, linewidth=3)
pyplot.plot(reconstructed_template_2)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])

print('-----> PLOTTING 1')
pyplot.subplot(2, 4, 1)
pyplot.plot(cummulative_variance)
pyplot.ylabel('Cumm. variance')
pyplot.xlabel('Number of components')

print('-----> PLOTTING 5')
#pyplot.subplot(2, 4, 5)
#pyplot.scatter(x, y, s=0.1)
#pyplot.ylabel('Template values')
#pyplot.xlabel('Reconstructed template values')

pyplot.tight_layout()
pyplot.savefig(settings.synthetic_templates_plot)
pyplot.show()

# %%
# Empirical templates and reconstruction
#

selected_indices = [500, 7000, 10000]

min_value = 0.15
max_value = 0.45

data_set = 'israel'
file_names = misc.folder_names(data_set, None)
data = numpy.load(file_names['npz_file'])
template_israel = data['long_data']

data_set = 'royal'
file_names = misc.folder_names(data_set, None)
data = numpy.load(file_names['npz_file'])
templates_royal = data['long_data']

all_templates = numpy.row_stack((templates_royal, template_israel))
reconstructed_templates = numpy.load(settings.reconstructed_templates_file)

pyplot.figure(figsize=(10, 5))
for i in range(len(selected_indices)):
    index = selected_indices[i]
    template = all_templates[index, :]
    recon = reconstructed_templates[index, :]

    pyplot.subplot(1, len(selected_indices), i + 1)
    pyplot.plot(template, linewidth=3)
    pyplot.plot(recon)
    pyplot.ylim([min_value, max_value])
    pyplot.xlabel('Sample')
    if i == 0: pyplot.legend(['Template', 'Reconstructed from PCs'])

pyplot.tight_layout()
pyplot.savefig(settings.templates_plot)
pyplot.show()
