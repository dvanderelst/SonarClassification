# components are in the rows
# components_array, shape (n_components, n_features)

import numpy
import os
from matplotlib import pyplot
from library import misc, settings
from PyAstronomy.pyaC import zerocross1d

pyplot.style.use(settings.style)
n = settings.n_components

pca_model = misc.pickle_load(settings.pca_file)
components = pca_model.components_

mx = numpy.max(numpy.abs(components))
n = components.shape[1]
x = numpy.array(range(n))

pyplot.figure(figsize=(10, 7))

for i in range(20):
    component = components[i, :]
    crossings = zerocross1d(x, component)
    n_crossings = crossings.size
    title = '$PC_{%i}$' % (i + 1)

    pyplot.subplot(4, 5, i + 1)
    pyplot.plot(component)
    pyplot.ylim([-mx, mx])
    pyplot.hlines(0, 0, n, alpha=0.25)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.text(60, mx / 1.5, 'zc = %i' % n_crossings)
    pyplot.title(title)

pyplot.tight_layout()
pyplot.savefig(settings.pca_components_plot)
pyplot.show()
