from matplotlib import pyplot
import scipy.io as io
import numpy
import math
from scipy.signal import convolve

file_name = 'andrews/clutter1.mat'

fs = 219000
integration_time = 350 / 1000000  # integration constant
initial_omission_time = 6 / 1000  # initial time to omit

initial_omission_samples = math.ceil(initial_omission_time * fs)
integration_samples = math.ceil(fs * integration_time)
final_samples = math.ceil((7499 - initial_omission_samples) / integration_samples)

print('+-' * 10 + '+')
print('PREPROCESSING', file_name)
print('integration samples', integration_samples)
print('initial omission samples', initial_omission_samples)
print('final samples', final_samples)

data = io.loadmat(file_name)
data = data['Templates']
templates = data[0, 0][0]
azimuth = data[0, 0][1]
elevation = data[0, 0][2]

processed = numpy.zeros((7, 31, final_samples, 3))

for i in range(3):
    repetition = templates[:, :, i]
    # Reshape the data
    az_box = azimuth.reshape((31, 7))
    el_box = elevation.reshape((31, 7))
    mn_box = repetition.reshape((31, 7, 7499))
    mn_box = mn_box[:, :, initial_omission_samples:]

    az_box = numpy.transpose(az_box)
    el_box = numpy.transpose(el_box)
    mn_box = numpy.transpose(mn_box, axes=[1, 0, 2])

    col_indices = numpy.argsort(az_box[0, :])
    row_indices = numpy.argsort(el_box[:, 0])

    az_box = az_box[numpy.ix_(row_indices, col_indices)]
    el_box = el_box[numpy.ix_(row_indices, col_indices)]
    mn_box = mn_box[numpy.ix_(row_indices, col_indices)]

    # Average across directions and time
    mask = numpy.ones((3, 3, integration_samples))
    mask = mask / numpy.sum(mask)
    mn_box = convolve(mn_box, mask, mode='same')
    # Subsample
    mn_box = mn_box[:, :, ::integration_samples]
    processed[:, :, :, i] = mn_box