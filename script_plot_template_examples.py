import numpy
import os
from matplotlib import pyplot
from library import misc, settings

#
# Synthetic echoes, templates and reconstruction
#

pyplot.style.use(settings.style)

synthetic_echoes = misc.pickle_load(settings.synthetic_echoes_file)
synthetic_indices = misc.pickle_load(settings.synthetic_echoes_indices)
synthetic_templates = numpy.load(settings.scaled_synthetic_templates_file)
reconstructed_synthetic_templates = numpy.load(settings.reconstructed_synthetic_templates_file)

pca_model = misc.pickle_load(settings.pca_file)
cummulative_variance = numpy.cumsum(pca_model.explained_variance_ratio_)

x = synthetic_templates.flatten()
y = reconstructed_synthetic_templates.flatten()

mx_echoes = 0.25
mx_templates = 0.35
mn_templates = 0.15

label0 = '5_5_0'
label1 = '10_10_0'
label2 = '250_250_0'

echo_0 = synthetic_echoes[label0]
echo_1 = synthetic_echoes[label1]
echo_2 = synthetic_echoes[label2]

index0 = synthetic_indices[label0]
index1 = synthetic_indices[label1]
index2 = synthetic_indices[label2]

template_0 = synthetic_templates[index0, :]
template_1 = synthetic_templates[index1, :]
template_2 = synthetic_templates[index2, :]

reconstructed_template_0 = reconstructed_synthetic_templates[index0, :]
reconstructed_template_1 = reconstructed_synthetic_templates[index1, :]
reconstructed_template_2 = reconstructed_synthetic_templates[index2, :]

pyplot.figure(figsize=(12,7))

pyplot.subplot(2,4,2)
pyplot.plot(echo_0)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])
pyplot.ylabel('Echo amplitude, model units')

pyplot.subplot(2,4,3)
pyplot.plot(echo_1)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])

pyplot.subplot(2,4,4)
pyplot.plot(echo_2)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])

pyplot.subplot(2,4,6)
pyplot.plot(template_0, linewidth=3)
pyplot.plot(reconstructed_template_0)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])
pyplot.legend(['Template', 'Reconstructed from PCs'])
pyplot.ylabel('Template amplitude, model units')

pyplot.subplot(2,4,7)
pyplot.plot(template_1, linewidth=3)
pyplot.plot(reconstructed_template_1)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])

pyplot.subplot(2,4,8)
pyplot.plot(template_2, linewidth=3)
pyplot.plot(reconstructed_template_2)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])

pyplot.subplot(2,4,1)
pyplot.plot(cummulative_variance)
pyplot.ylabel('Cumm. variance')
pyplot.xlabel('Number of components')

pyplot.subplot(2,4,5)
pyplot.scatter(x,y, s=0.1)
pyplot.ylabel('Template values')
pyplot.xlabel('Reconstructed template values')

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

pyplot.figure(figsize=(10,5))
for i in range(len(selected_indices)):
    index = selected_indices[i]
    template = all_templates[index, :]
    recon = reconstructed_templates[index,:]

    pyplot.subplot(1, len(selected_indices), i + 1)
    pyplot.plot(template, linewidth=3)
    pyplot.plot(recon)
    pyplot.ylim([min_value, max_value])
    pyplot.xlabel('Sample')
    if i == 0: pyplot.legend(['Template', 'Reconstructed from PCs'])

pyplot.tight_layout()
pyplot.savefig(settings.templates_plot)
pyplot.show()