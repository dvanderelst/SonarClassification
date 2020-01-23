fs = 219000
raw_collected_samples = 7499
noise_floor = 0.15 # should be determined on open space

# It turns out, or so it seems, that the noise before and after the transformation
# by pca has the same sdv. I should check whether this makes mathematical sense.
stochaistic_noise = 0.02 # stdv, See paper for determination, before pca mapping


integration_time = 350 / 1000000  # integration constant
initial_zero_time = 6 / 1000  # initial time to omit