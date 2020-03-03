import math

import numpy
from numpy import random
from scipy.signal import chirp

from library import settings
from pyBat import Acoustics
from pyBat import Signal

from scipy.signal import convolve

def create_emission():
    emission_duration = settings.emission_duration
    sample_frequency = settings.sample_frequency
    emission_frequency_high = settings.emission_frequency_high
    emission_frequency_low = settings.emission_frequency_low

    emission_samples = int(sample_frequency * emission_duration)
    emission_time = numpy.linspace(0, emission_duration, emission_samples)
    emission = chirp(emission_time, f0=emission_frequency_high, f1=emission_frequency_low, t1=emission_duration, method='quadratic')
    emission_window = Signal.signal_ramp(emission_samples, 10)
    emission = emission * emission_window
    return emission


def generative_model(parameters):
    minimum_distance = 1
    maximum_distance = 7
    standard_deviation = 0.1
    n_seed_points = parameters.pop('n_seed_points')
    n_cloud_points = parameters.pop('n_cloud_points')
    keys = parameters.keys()
    if len(keys) > 0: print('***********Warning', keys)
    distances = numpy.array([])
    seed_points = random.uniform(minimum_distance, maximum_distance, n_seed_points)
    for loc in seed_points:
        g = random.normal(loc=loc, scale=standard_deviation, size=n_cloud_points)
        distances = numpy.concatenate((distances, g))

    # add close echo to ensure that even seqs with low n have a strong echo
    #close = random.uniform(minimum_distance, minimum_distance+1, 1)
    #g = numpy.array([minimum_distance])
    #distances = numpy.concatenate((distances, close))
    return distances

def distances2echo_sequence(distances, caller, emission):
    emission_duration = settings.emission_duration
    sample_frequency = settings.sample_frequency
    number_of_samples = settings.raw_collected_samples
    number_of_zero_samples = math.ceil(settings.initial_zero_time * sample_frequency)
    # Get intensities
    azimuths = numpy.zeros(distances.shape)
    elevations = numpy.zeros(distances.shape)
    call_result = caller.call(azimuths, elevations, distances)
    left_db = call_result['echoes_left']
    delays = call_result['delays']

    # Make impulse response
    left_db[left_db<0] = 0
    ir_result = Acoustics.make_impulse_response(delays, left_db, emission_duration, sample_frequency)
    impulse_response = ir_result['ir_result']
    shape = impulse_response.shape[0]
    padding = number_of_samples - shape
    if padding > 0: impulse_response = numpy.pad(impulse_response, (0, padding), 'constant')
    impulse_response = impulse_response[0: number_of_samples]

    # Generate echo sequency
    echo_sequence = numpy.convolve(emission, impulse_response, mode='same')
    echo_sequence[0:number_of_zero_samples] = 0
    return echo_sequence, impulse_response


def echo_sequence2template(echo_sequence, wiegrebe):
    sample_frequency = settings.sample_frequency
    integration_time = settings.integration_time
    #integration_samples = math.ceil(sample_frequency * integration_time)

    # Run Wiegrebe model
    wiegrebe_result = wiegrebe.run_model(echo_sequence, dechirp=True)
    wiegrebe_result = wiegrebe_result.reshape(1, -1)

    # Subsample
    #mask = numpy.ones((1, integration_samples))
    #mask = mask / numpy.sum(mask)
    #wiegrebe_result = convolve(wiegrebe_result, mask, mode='same')
    #wiegrebe_result = wiegrebe_result[:, ::integration_samples]
    wiegrebe_result = wiegrebe_result[0]
    return wiegrebe_result



def evaluate_fit(template, pca_model, remove_begin_samples = 17, remove_end_samples=17):
    # Apply pca model
    transformed = pca_model.transform(template)
    reconstructed = pca_model.inverse_transform(transformed)
    template = template[0, remove_begin_samples:-remove_end_samples]
    reconstructed = reconstructed[0, remove_begin_samples:-remove_end_samples]
    explained_variance = numpy.corrcoef(template, reconstructed) ** 2
    explained_variance = explained_variance[0, 1]
    return template, reconstructed, explained_variance