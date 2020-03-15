from numpy.fft import fft, ifft
from numpy import conj, pad, abs
from library import generate_functions
from matplotlib import pyplot

pyplot.figure(figsize=(10,10))

emission = generate_functions.create_emission()
signal = generate_functions.create_emission()

emission = pad(emission, (0, 100), 'constant')
signal = pad(signal, (50, 50), 'constant')

a = conj(fft(emission))
b = fft(signal)
c = ifft(a * b)
dechriped = abs(c)


pyplot.subplot(2,2,1)
pyplot.plot(emission)
pyplot.title('Emission')

pyplot.subplot(2,2,2)
pyplot.plot(signal)
pyplot.title('Signal == Emission')

pyplot.subplot(2,2,3)
pyplot.plot(dechriped)
pyplot.title('Dechirped signal (magnitude)')

pyplot.show()