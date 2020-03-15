import scipy.stats as stats
from matplotlib import pyplot
import os
from library import misc, settings

pyplot.style.use(settings.style)

data_set = 'royal'

loss_function = 'Categorical Cross Entropy'

files_lcs = misc.folder_names(data_set, 'lcs')
files_azs = misc.folder_names(data_set, 'azs')
files_els = misc.folder_names(data_set, 'els')
results_azs = misc.pickle_load(files_azs['results_file'])
results_els = misc.pickle_load(files_els['results_file'])
results_lcs = misc.pickle_load(files_lcs['results_file'])
perfect_memory = misc.pickle_load(files_lcs['perfect_memory_file'])


# %% Plot errors

fig, axes = pyplot.subplots(nrows=3, ncols=3)
fig.set_figheight(10)
fig.set_figwidth(12)

pyplot.sca(axes[0, 0])
counts = misc.get_error_histogram(results_azs, normalize=True, cummmulative=True)
counts_pm = misc.get_error_histogram(perfect_memory['results_azs'], normalize=True, cummmulative=True)
pyplot.plot(counts.error, counts.number, color=settings.azs_color)
pyplot.plot(counts_pm.error, counts_pm.number, color='k', alpha=0.5)
_, entries = misc.plot_inference_lines(counts.error, counts.number, [30, 45, 90])
pyplot.ylim([0, 1.1])
pyplot.xlabel('Error (degrees)')
pyplot.ylabel('Proportion of templates')
pyplot.legend(['Neural network', 'Perfect memory'] + entries)
pyplot.title('Azimuth')

pyplot.sca(axes[0, 1])
counts = misc.get_error_histogram(results_els, normalize=True, cummmulative=True)
counts_pm = misc.get_error_histogram(perfect_memory['results_els'], normalize=True, cummmulative=True)
pyplot.plot(counts.error, counts.number, color=settings.els_color)
pyplot.plot(counts_pm.error, counts_pm.number, color='k', alpha=0.5)
_, entries = misc.plot_inference_lines(counts.error, counts.number, [15, 30])
pyplot.ylim([0, 1.1])
pyplot.xlabel('Error (degrees)')
pyplot.ylabel('Proportion of templates')
pyplot.legend(['Neural network', 'Perfect memory'] + entries)
pyplot.title('Elevation')

pyplot.sca(axes[0, 2])
counts = misc.get_error_histogram(results_lcs, normalize=True, cummmulative=True)
counts_pm = misc.get_error_histogram(perfect_memory['results_lcs'], normalize=True, cummmulative=True)
pyplot.plot(counts.error, counts.number, color=settings.lcs_color)
pyplot.plot(counts_pm.error, counts_pm.number, color='k', alpha=0.5)
_, entries = misc.plot_inference_lines(counts.error, counts.number, [0.5, 1, 2])
pyplot.ylim([0, 1.1])
pyplot.xlabel('Error (meters)')
pyplot.ylabel('Proportion of templates')
pyplot.legend(['Neural network', 'Perfect memory'] + entries)
pyplot.title('Location')

pyplot.sca(axes[1, 0])
table, azs_labels = misc.make_confusion_matrix(results_azs, normalize=True)
azs_entropy = stats.entropy(table, base=2)  # works over the rows. Ie gives the entropy of each row.
pyplot.imshow(table, cmap=settings.colormap, vmin=0, vmax=1)
misc.label_confusion_matrix(azs_labels)
pyplot.ylabel('Target')
pyplot.xlabel('Output')

pyplot.sca(axes[1, 1])
table, els_labels = misc.make_confusion_matrix(results_els, normalize=True)
els_entropy = stats.entropy(table, base=2)  # axis=0
pyplot.imshow(table, cmap=settings.colormap, vmin=0, vmax=1)
misc.label_confusion_matrix(els_labels)
pyplot.xlabel('Output')

pyplot.sca(axes[1, 2])
table, lcs_labels = misc.make_confusion_matrix(results_lcs, normalize=True)
lcs_entropy = stats.entropy(table, base=2)  # axis=0
pyplot.imshow(table, cmap=settings.colormap, vmin=0, vmax=1)
misc.label_confusion_matrix(lcs_labels)
pyplot.xlabel('Output')

pyplot.sca(axes[2, 0])
pyplot.plot(azs_labels, azs_entropy, color=settings.azs_color)
pyplot.xlabel('Azimuth (degrees)')
pyplot.ylabel('Entropy (bits)')
pyplot.ylim([0, 5])

pyplot.sca(axes[2, 1])
pyplot.plot(els_labels, els_entropy, color=settings.els_color)
pyplot.xlabel('Elevation (degrees)')
pyplot.ylim([0, 5])

pyplot.sca(axes[2, 2])
pyplot.plot(lcs_labels, lcs_entropy, color=settings.lcs_color)
pyplot.xlabel('Location (meters)')
pyplot.ylim([0, 5])

pyplot.tight_layout()
if data_set == 'israel': pyplot.savefig(settings.israel_results_plot)
if data_set == 'royal': pyplot.savefig(settings.royal_results_plot)
pyplot.show()
