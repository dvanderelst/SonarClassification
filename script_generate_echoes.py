import math

from matplotlib import pyplot

from library import generate_echoes_functions
from library import misc
from library import settings
from pyBat import Call
from pyBat import Wiegrebe

repeats = 3

sample_frequency = settings.sample_frequency
number_of_samples = settings.raw_collected_samples
initial_zero_time = settings.initial_zero_time
emission_frequency_mean = settings.emission_frequency_mean

integration_samples = math.ceil(sample_frequency * settings.integration_time)

caller = Call.Call(None, [emission_frequency_mean])
wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4)
pca_model = misc.pickle_load(settings.pca_file)
emission = generate_echoes_functions.create_emission()

for i in range(repeats):
    # Make distances
    generative_model_parameters = {}
    generative_model_parameters['n_seed_points'] = 50
    generative_model_parameters['n_cloud_points'] = 100
    distances = generate_echoes_functions.generative_model(generative_model_parameters)

    # Make echo sequence
    echo_sequence = generate_echoes_functions.distances2echo_sequence(distances, caller, emission)

    # Create template
    template = generate_echoes_functions.echo_sequence2template(echo_sequence, wiegrebe)

    # Evaluate
    template, reconstructed, explained_variance = generate_echoes_functions.evaluate_fit(template, pca_model)
    
    pyplot.figure()
    pyplot.plot(template)
    pyplot.plot(reconstructed)
    pyplot.show()