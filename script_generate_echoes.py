import math

import numpy
from matplotlib import pyplot
from numpy import random

from scipy.signal import chirp
from scipy.signal import convolve

from library import settings
from library import misc

from pyBat import Acoustics
from pyBat import Call
from pyBat import Signal
from pyBat import Wiegrebe

emission_duration = 0.005
emission_frequency_low = 25000
emission_frequency_high = 75000
minumum_distance = 0.5

sample_frequency = settings.sample_frequency
number_of_samples = settings.raw_collected_samples
initial_zero_time = settings.initial_zero_time

emission_frequency_mean = round((emission_frequency_low + emission_frequency_high) / 2)
integration_samples = math.ceil(sample_frequency * settings.integration_time)
# Generate reflectors

distances = numpy.array([])
seed_points = random.uniform(minumum_distance, 6, 15)
for loc in seed_points:
    g = random.normal(loc=loc, scale=0.15, size=250)
    distances = numpy.concatenate((distances, g))

distances = distances[distances > minumum_distance]
pyplot.hist(distances, 50)
pyplot.show()

# Get echo intensities

caller = Call.Call('pd01', [emission_frequency_mean])
azimuths = numpy.zeros(distances.shape)
elevations = numpy.zeros(distances.shape)
call_result = caller.call(azimuths, elevations, distances)
left_db = call_result['echoes_left']
delays = call_result['delays']

# Make impulse response

ir_result = Acoustics.make_impulse_response(delays, left_db, emission_duration, sample_frequency)
impulse_response = ir_result['ir_result']

pyplot.figure()
pyplot.scatter(ir_result['impulse_time'], ir_result['ir_result'])
pyplot.show()

# Make emission

emission_samples = int(sample_frequency * emission_duration)
emission_time = numpy.linspace(0, emission_duration, emission_samples)
emission = chirp(emission_time, f0=emission_frequency_high, f1=emission_frequency_low, t1=emission_duration, method='quadratic')
emission_window = Signal.signal_ramp(emission_samples, 10)
emission = emission * emission_window

pyplot.figure()
pyplot.plot(emission)
pyplot.show()

# Generate echo sequency

number_of_zero_samples = math.ceil(settings.initial_zero_time * sample_frequency)
echo_sequence = numpy.convolve(emission, impulse_response, mode='same')
echo_sequence = echo_sequence[0:number_of_samples]
echo_sequence[0:number_of_zero_samples] = 0

pyplot.figure()
pyplot.plot(echo_sequence)
pyplot.show()

#%% Run Wiegrebe model

wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4)
wiegrebe_result = wiegrebe.run_model(echo_sequence)
wiegrebe_result = wiegrebe_result.reshape(1, -1)

pyplot.figure()
pyplot.plot(wiegrebe_result)
pyplot.show()

# Subsample
mask = numpy.ones((1, integration_samples))
mask = mask / numpy.sum(mask)
mn_box = convolve(wiegrebe_result, mask, mode='same')

print(wiegrebe_result.shape)
wiegrebe_result = wiegrebe_result[:, ::(integration_samples)]
print(wiegrebe_result.shape)

pca_model = misc.pickle_load(settings.pca_file)
transformed = pca_model.transform(wiegrebe_result)
reconstructed = pca_model.inverse_transform(transformed)

wiegrebe_result = wiegrebe_result[0,17:]
reconstructed = reconstructed[0,17:]

pyplot.figure()
pyplot.plot(wiegrebe_result)
pyplot.plot(reconstructed)
pyplot.legend(['Cochlear output','projected and reconstructed'])
pyplot.show()

numpy.corrcoef(wiegrebe_result, reconstructed)

explained_variance = numpy.corrcoef(wiegrebe_result, reconstructed) ** 2
print(explained_variance)