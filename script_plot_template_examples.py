import numpy
from matplotlib import pyplot
from library import misc, settings


#
# Synthetic echoes, templates and reconstruction
#

index0 = 10
index1 = 100
index2 = 250

pyplot.style.use(settings.style)

print('-----> LOADING DATA')
loaded = numpy.load(settings.synthetic_echoes_file)
synthetic_echoes = loaded['synthetic_echoes']

loaded = numpy.load(settings.synthetic_templates_pca_results)
synthetic_templates = loaded['original']
reconstructed_synthetic_templates = loaded['reconstructed']

#average_spectrum = synthetic_templates.flatten()
#y = reconstructed_synthetic_templates.flatten()

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

mx_echoes = numpy.max(numpy.concatenate((echo_0, echo_1, echo_2)))
mx_templates = numpy.max(numpy.concatenate((template_0, template_1, template_2)))
mn_templates = numpy.min(numpy.concatenate((template_0, template_1, template_2)))
mn_templates = mn_templates * 0.99
mx_templates = mx_templates * 1.01

print('-----> PLOTTING')

fig = pyplot.figure(figsize=(9, 4))

gs = fig.add_gridspec(2, 3, height_ratios=[0.25, 1] )

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])

print('-----> PLOTTING 2')

pyplot.sca(ax0)
pyplot.plot(echo_0, color=settings.default_line_color01)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])
pyplot.ylabel('Amplitude')
misc.label(0.9, 0.8, 0, fontsize=14)

print('-----> PLOTTING 3')
pyplot.sca(ax1)
pyplot.plot(echo_1, color=settings.default_line_color01)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])
misc.label(0.9, 0.8, 1, fontsize=14)

print('-----> PLOTTING 4')
pyplot.sca(ax2)
pyplot.plot(echo_2, color=settings.default_line_color01)
pyplot.ylim([-mx_echoes, mx_echoes])
pyplot.xticks([])
pyplot.yticks([])
misc.label(0.9, 0.8, 2, fontsize=14)

print('-----> PLOTTING 6')
pyplot.sca(ax3)
pyplot.plot(template_0, linewidth=3, color=settings.default_line_color01)
pyplot.plot(reconstructed_template_0, color=settings.default_line_color02, alpha=0.75)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])
pyplot.legend(['Template', 'Reconstructed'], frameon=False)
pyplot.ylabel('Template amplitude')
misc.label(0.1, 0.9, 3, fontsize=14)

print('-----> PLOTTING 7')
pyplot.sca(ax4)
pyplot.plot(template_1, linewidth=3, color=settings.default_line_color01)
pyplot.plot(reconstructed_template_1, color=settings.default_line_color02, alpha=0.75)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])
misc.label(0.1, 0.9, 4, fontsize=14)

print('-----> PLOTTING 8')
pyplot.sca(ax5)
pyplot.plot(template_2, linewidth=3, color=settings.default_line_color01)
pyplot.plot(reconstructed_template_2, color=settings.default_line_color02, alpha=0.75)
pyplot.ylim([mn_templates, mx_templates])
pyplot.xticks([])
pyplot.yticks([])
misc.label(0.1, 0.9, 5, fontsize=14)

pyplot.tight_layout()
pyplot.savefig(settings.synthetic_templates_plot)
pyplot.show()


# %%
# Empirical templates and reconstruction
#

selected_indices = [500, 7000, 9000]

min_value = 0.15
max_value = 0.45

empirical_templates = misc.load_all_empirical_templates()
pca_model = misc.pickle_load(settings.pca_templates_model_file)
transformed, reconstructed, correlation = misc.project_and_reconstruct(pca_model, empirical_templates)

#%%
pyplot.figure(figsize=(8, 2.5))
for i in range(len(selected_indices)):
    index = selected_indices[i]
    template = empirical_templates[index, :]
    recon = reconstructed[index, :]

    pyplot.subplot(1, len(selected_indices), i + 1)
    pyplot.plot(template, linewidth=3, color=settings.default_line_color01)
    pyplot.plot(recon, color=settings.default_line_color02, alpha=0.75)
    pyplot.ylim([min_value, max_value])
    pyplot.xticks([])
    pyplot.yticks([])
    if i == 0: pyplot.legend(['Template', 'Reconstructed'],frameon=False)
    misc.label(0.1, 0.9, i, fontsize=14)

pyplot.tight_layout()
pyplot.savefig(settings.empirical_templates_plot)
pyplot.show()
