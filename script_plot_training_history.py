import numpy
import os
from matplotlib import pyplot
from library import smoothn, misc, settings

pyplot.style.use(settings.style)

pyplot.figure(figsize=(7,3))

for data_set in ['israel', 'royal']:
    if data_set == 'israel': pyplot.subplot(1, 2, 1)
    if data_set == 'royal': pyplot.subplot(1, 2, 2)
    for dimension in ['azs', 'els', 'lcs']:
        files = misc.folder_names(data_set, dimension)
        history_file = files['history_file']
        history = misc.pickle_load(history_file)
        if data_set == 'israel': linestyle = settings.israel_linestyle
        if data_set == 'royal': linestyle = settings.royal_linestyle

        if dimension == 'azs': color = settings.azs_color
        if dimension == 'els': color = settings.els_color
        if dimension == 'lcs': color = settings.lcs_color

        trace = numpy.array(history['loss'])
        trace = smoothn.smoothn(trace, s0=1)[0]

        pyplot.plot(trace, color=color)
        pyplot.ylim([0, 4])
        #pyplot.xscale('symlog')
        pyplot.xlabel('Epoch')
        if data_set == 'israel': pyplot.ylabel('Loss')
        if data_set == 'israel': pyplot.title('Israel Data')
        if data_set == 'royal': pyplot.title('Royal Fort Data')

pyplot.tight_layout()
pyplot.legend(['Azimuth', 'Elevation', 'Location'])
pyplot.savefig(settings.training_history_plot)
pyplot.show()
