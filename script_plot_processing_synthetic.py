import math

import numpy
from matplotlib import pyplot


from library import misc
from library import generate_functions
from library import settings
from pyBat import Wiegrebe

sample_frequency = settings.sample_frequency
number_of_samples = settings.raw_collected_samples
initial_zero_time = settings.initial_zero_time
emission_frequency_mean = settings.emission_frequency_mean
initial_zero_samples = math.ceil(initial_zero_time * sample_frequency)

emission = generate_functions.create_emission()

emission_padding = number_of_samples - emission.shape[0]
padded_emission = numpy.pad(emission, (0, emission_padding), 'constant')

wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4, emission=padded_emission)

generative_model_parameters = {}
generative_model_parameters['n_seed_points'] = [100]
generative_model_parameters['n_cloud_points'] = [100]
distances = generate_functions.generative_model(generative_model_parameters)
echo_sequence, impulse_response = generate_functions.distances2echo_sequence(distances, emission)
echo_sequence[0:initial_zero_samples] = 0
template = generate_functions.echo_sequence2template(echo_sequence, wiegrebe)

# normalize stuff for easier plotting
echo_sequence = echo_sequence/numpy.max(numpy.abs(echo_sequence))

bm = wiegrebe.filtered
bm = bm - numpy.min(bm)
bm = bm / numpy.max(bm)


template = template - numpy.min(template)
template = template / numpy.max(template)

correlations = misc.pickle_load(settings.channel_correlations_file)
correlations = numpy.array(correlations)

synthetic_template_pca = misc.pickle_load(settings.pca_templates_model_file)
cumulative = numpy.cumsum(synthetic_template_pca.explained_variance_ratio_)

# %%
pyplot.style.use(settings.style)

fig = pyplot.figure(figsize=(20, 10))

gs = fig.add_gridspec(8, 3)
ax = fig.add_subplot(gs[:, 0])
ax.plot(echo_sequence, color=settings.default_line_color01)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-1.1,1.1])
misc.label(0.95,0.95, 0, fontsize=20)

for x in range(8):
    f = wiegrebe.frequencies[x]
    title = 'CF= %i Khz' % (f/1000)
    signal = bm[x,:]
    ax = fig.add_subplot(gs[x, 1])
    ax.plot(signal, color=settings.default_line_color01)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-0.1, 1])
    ax.text(5000, 0.5, title)
    misc.label(0.95, 0.8, x+1, fontsize=20)


# ax = fig.add_subplot(gs[0:4, 2])
# ax.plot(cumulative, color=settings.default_line_color)
# ax.set_ylabel('Cumm. expl. variance')
# ax.set_xlabel('Nr of components')

ax = fig.add_subplot(gs[0:2, 2])
r = ax.hist(correlations, color=settings.default_line_color01)
ax.set_ylabel('Echo Seq. Count')
ax.set_xlabel('Cross-channel correlation/variance')
ax.set_xlim([0.75, 1])
mn = numpy.mean(correlations)
h = numpy.max(r[0])
ax.vlines(mn,0,h,colors=settings.default_line_color02)
misc.label(0.05, 0.8, x+1, fontsize=20)

ax = fig.add_subplot(gs[2:6, 2])
ax.plot(template, color=settings.default_line_color01)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-0.1,1.1])
misc.label(0.95, 0.95, x+2, fontsize=20)


ax = fig.add_subplot(gs[6:8, 2])
ax.plot(cumulative, color=settings.default_line_color01)
ax.set_ylabel('Cumm. expl. variance')
ax.set_xlabel('Nr of components')
misc.label(0.95, 0.8, x+3, fontsize=20)




pyplot.tight_layout()
pyplot.savefig(settings.synthetic_processing_plot)
pyplot.show()
