import numpy
from matplotlib import pyplot
from scipy.signal import hilbert

from library import misc
from library import settings
from library import generate_functions
from pyBat import Wiegrebe

number_of_samples = settings.raw_collected_samples
emission = generate_functions.create_emission()
emission_padding = number_of_samples - emission.shape[0]
padded_emission = numpy.pad(emission, (0, emission_padding), 'constant')
sample_frequency = settings.sample_frequency
emission_frequency_mean = settings.emission_frequency_mean

wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4)

do_calc = True

if do_calc:

    print('---> LOADING SYNTHETIC DATA')
    # THESE HAVE ALREADY BEEN DECHIRPED .....
    loaded = numpy.load(settings.synthetic_echoes_file)
    synthetic_echoes = loaded['synthetic_echoes']
    n = synthetic_echoes.shape[0]

    shuffles = [False, False, True, True]
    filters = [True, False, True, False]
    results = {}
    pcs = []
    labels = []

    for filter, shuffle in zip(filters, shuffles):
        spectra = []
        templates = []
        for i in range(n):
            print(i, filter, shuffle)
            signal = synthetic_echoes[i, :]
            if shuffle: numpy.random.shuffle(signal)
            template = wiegrebe.run_model(signal, apply_attenuation=True)
            if not filter: template = numpy.mean(wiegrebe.compressed, axis=0)
            spectrum = misc.template2fft(template)
            spectra.append(spectrum)
            templates.append(template)

        spectra = numpy.array(spectra)
        templates = numpy.array(templates)
        label = '%s_%s' % (filter, shuffle)
        results[label] = [spectra, templates]

        model, stats = misc.run_pca(templates, criterion=settings.pca_criterion)
        pcs.append(stats["suggestion"])
        labels.append(label)

    results['pcs'] = pcs
    results['labels'] = labels
    misc.pickle_save(settings.mechanism_data_file, results)
# %%
from library import settings
from library import smoothn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

results = misc.pickle_load(settings.mechanism_data_file)

pyplot.style.use(settings.style)

pcs = results['pcs']
labels = results['labels']
colors = [settings.default_line_color01, settings.default_line_color02, settings.default_line_color02, settings.default_line_color02]
linestyles = ['-', '--', '-', '--']

text_labels = ['1 kHz filter', 'No 1 kHz filter', '1 kHz filter\nshuffled', 'No 1 kHz filter\nshuffled']

pyplot.close('alll')
pyplot.figure()

#pyplot.subplot(2, 1, 1)
pyplot.bar([0, 1, 2, 3], pcs, color=colors)
pyplot.xticks([0, 1, 2, 3], text_labels)
pyplot.ylabel('Number of components')
pyplot.grid(False)
pyplot.ylim(0, 500)

for i, v in enumerate(pcs):
    pyplot.text(i, v + 10, str(v), color='black', fontweight='bold', horizontalalignment='center')


inset_axes = inset_axes(pyplot.gca(),
                    width="30%", # width = 30% of parent_bbox
                    height="20%",
                    loc=2,
                    borderpad=2)


frequency_axis = numpy.fft.fftfreq(settings.raw_collected_samples, d=1 / 219000) / 1000
positive = frequency_axis > 0
frequency_axis = frequency_axis[positive]
data = results['True_False']
spectra = data[0]
average = numpy.mean(spectra, axis=0)
average = average[positive]
average = smoothn.smoothn(average, s=2)[0]
inset_axes.plot(frequency_axis, average, color=settings.default_line_color01)
inset_axes.set_xlim(0,5)
inset_axes.set_xlabel('Freq. (kHz)')
inset_axes.set_yticks([])
inset_axes.set_xticks([0,1,2,3,4,5])
inset_axes.set_title('Av. Template Spectrum',fontsize=10)

# pyplot.subplot(2, 1, 2)
# frequency_axis = numpy.fft.fftfreq(settings.raw_collected_samples, d=1 / 219000) / 1000
# positive = frequency_axis > 0
# frequency_axis = frequency_axis[positive]

# for label in labels:
#     i = labels.index(label)
#     data = results[label]
#     spectra = data[0]
#     average = numpy.mean(spectra, axis=0)
#     average = average[positive]
#     average = smoothn.smoothn(average, s=2)[0]
#     pyplot.plot(frequency_axis, average, linestyles[i], color=colors[i])
#
#
# pyplot.xlim(0, 5)
# pyplot.xlabel('Frequency (kHz)')
# pyplot.legend(text_labels)

#pyplot.tight_layout()
pyplot.savefig(settings.mechanism_plot)
pyplot.show()
