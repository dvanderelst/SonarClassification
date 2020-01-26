import scipy.stats as stats
from matplotlib import pyplot
import os
import misc
import settings

pyplot.style.use(settings.style)

data_set = 'royal'

loss_function = 'Categorical Cross Entropy'

files_lcs = misc.folder_names(data_set, 'lcs')
files_azs = misc.folder_names(data_set, 'azs')
files_els = misc.folder_names(data_set, 'els')
results_azs = misc.pickle_load(files_azs['results_file'])
results_els = misc.pickle_load(files_els['results_file'])
results_lcs = misc.pickle_load(files_lcs['results_file'])

output_file = os.path.join(settings.figure_folder, data_set + '_performance.pdf')

# %% Plot errors

fig, axes = pyplot.subplots(nrows=3, ncols=3)
fig.set_figheight(10)
fig.set_figwidth(10)

pyplot.sca(axes[0, 0])
counts = misc.get_error_histogram(results_azs, normalize=True, cummmulative=True)
pyplot.plot(counts.error, counts.number, color=settings.azs_color)
pyplot.ylim([0, 1])
pyplot.xlabel('Error (degrees)')
pyplot.ylabel('Proportion of templates')

pyplot.sca(axes[0, 1])
counts = misc.get_error_histogram(results_els, normalize=True, cummmulative=True)
pyplot.plot(counts.error, counts.number, color=settings.els_color)
pyplot.ylim([0, 1])
pyplot.xlabel('Error (degrees)')
pyplot.ylabel('Proportion of templates')

pyplot.sca(axes[0, 2])
counts = misc.get_error_histogram(results_lcs, normalize=True, cummmulative=True)
pyplot.plot(counts.error, counts.number, color=settings.lcs_color)
pyplot.ylim([0, 1])
pyplot.xlabel('Error (meters)')
pyplot.ylabel('Proportion of templates')

pyplot.sca(axes[1, 0])
table, labels = misc.make_confusion_matrix(results_azs, normalize=True)
azs_entropy = stats.entropy(table, base=2)  # axis=0
pyplot.imshow(table, cmap=settings.colormap)

pyplot.sca(axes[1, 1])
table, labels = misc.make_confusion_matrix(results_els, normalize=True)
els_entropy = stats.entropy(table, base=2)  # axis=0
pyplot.imshow(table, cmap=settings.colormap)

pyplot.sca(axes[1, 2])
table, labels = misc.make_confusion_matrix(results_lcs, normalize=True)
lcs_entropy = stats.entropy(table, base=2)  # axis=0
pyplot.imshow(table, cmap=settings.colormap)

pyplot.sca(axes[2, 0])
pyplot.plot(azs_entropy, color=settings.azs_color)
pyplot.ylim([0, 5])

pyplot.sca(axes[2, 1])
pyplot.plot(els_entropy, color=settings.els_color)
pyplot.ylim([0, 5])

pyplot.sca(axes[2, 2])
pyplot.plot(lcs_entropy, color=settings.lcs_color)
pyplot.ylim([0, 5])

pyplot.tight_layout()
pyplot.savefig(output_file)
pyplot.show()
