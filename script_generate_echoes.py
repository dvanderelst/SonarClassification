import math
from matplotlib import pyplot
import numpy

from library import generate_functions
from library import misc
from library import settings
from pyBat import Call
from pyBat import Wiegrebe

pyplot.style.use(settings.style)

repeats = 10

sample_frequency = settings.sample_frequency
number_of_samples = settings.raw_collected_samples
initial_zero_time = settings.initial_zero_time
emission_frequency_mean = settings.emission_frequency_mean

integration_samples = math.ceil(sample_frequency * settings.integration_time)

caller = Call.Call(None, [emission_frequency_mean])
wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4)
pca_model = misc.pickle_load(settings.pca_file)
emission = generate_functions.create_emission()

all_fits = numpy.zeros((3, 3))
all_inverse_fits = numpy.zeros((3, 3))

n_seed_levels = [50, 100, 150]
n_cloud_levels = [50, 100, 150]

for seed_level_i in range(len(n_seed_levels)):
    for cloud_level_i in range(len(n_cloud_levels)):
        fits = []
        inverse_fits = []
        print(seed_level_i, cloud_level_i)
        for i in range(repeats):
            # Make distances
            generative_model_parameters = {}
            generative_model_parameters['n_seed_points'] = n_seed_levels[seed_level_i]
            generative_model_parameters['n_cloud_points'] = n_cloud_levels[cloud_level_i]
            distances = generate_functions.generative_model(generative_model_parameters)

            # Make echo sequence
            echo_sequence = generate_functions.distances2echo_sequence(distances, caller, emission)

            # Create template
            template = generate_functions.echo_sequence2template(echo_sequence, wiegrebe)
            inverse_template = numpy.flip(template)

            # Evaluate
            template, reconstructed, explained_variance = generate_functions.evaluate_fit(template, pca_model)
            fits.append(explained_variance)

            inverse_template, inverse_reconstructed, explained_variance = generate_functions.evaluate_fit(inverse_template, pca_model)
            inverse_fits.append(explained_variance)

        all_fits[seed_level_i, cloud_level_i] = numpy.mean(fits)
        all_inverse_fits[seed_level_i, cloud_level_i] = numpy.mean(inverse_fits)

#%%
pyplot.imshow(all_fits)
pyplot.colorbar()
pyplot.show()

pyplot.imshow(all_inverse_fits)
pyplot.colorbar()
pyplot.show()


pyplot.figure()
pyplot.plot(template)
pyplot.plot(reconstructed)
pyplot.plot(inverse_reconstructed)
pyplot.show()