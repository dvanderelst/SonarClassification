import misc
import numpy
import settings
import os
from matplotlib import pyplot

pca_plot_file = os.path.join(settings.figure_folder, 'pca.pdf')

pyplot.style.use(settings.style)
pyplot.figure(figsize=(4,4))
for data_set in ['israel', 'royal']:

    files = misc.folder_names(data_set, 'lcs')
    pca = misc.pickle_load(files['pca_file'])
    cvar = numpy.cumsum(pca.explained_variance_ratio_)
    if data_set == 'royal': pyplot.plot(cvar, '--k')
    if data_set == 'israel': pyplot.plot(cvar, '-k')

pyplot.legend(['Israel', 'Royal'])
pyplot.xlabel('Nr of PCs')
pyplot.ylabel('Proportion of explained variance')
pyplot.tight_layout()
pyplot.savefig(pca_plot_file)
pyplot.show()