import numpy
import pandas
import scipy.spatial.distance as distance

from library import misc, process_functions, settings

data_set = 'israel'
generate_data = False
iterations = 10

print('Running perfect memory for', data_set)

file_names = misc.folder_names(data_set, None)
output_file = file_names['perfect_memory_file']
if generate_data: process_functions.process_data_set(data_set)

# Read prepared data
data = numpy.load(file_names['npz_file'])
corridor_distances = misc.map_lcs_to_distances(data)

pca = misc.pickle_load(file_names['pca_file'])

correct_ids = data['ids']
correct_ids[:, 0] = corridor_distances

templates = data['long_data']
pca_templates = pca.transform(templates)
n_components = settings.n_components
inputs = pca_templates[:, :n_components]

# Scale inputs to a minimum of zero
inputs = inputs - numpy.min(inputs)

summed_ids = numpy.zeros(correct_ids.shape)
for repeat in range(iterations):
    print('iteration', repeat, '/', iterations)
    # Make noisey templates
    noise = numpy.random.normal(0, settings.stochaistic_noise, inputs.shape)
    noisy_templates = inputs + noise
    # rows: inputs, cols: noisy_templates
    distances = distance.cdist(inputs, noisy_templates)
    # for each input, what noisy template is closest?
    closest = numpy.argmin(distances, axis=1)
    summed_ids = summed_ids + correct_ids[closest, :]
inferred_ids = summed_ids/iterations

# lcs results
results_lcs = {'target': correct_ids[:, 0], 'prediction': inferred_ids[:, 0]}
results_lcs = pandas.DataFrame(results_lcs)
results_lcs['dummy'] = 1

# azs results
results_azs = {'target': correct_ids[:, 1], 'prediction': inferred_ids[:, 1]}
results_azs = pandas.DataFrame(results_azs)
results_azs['dummy'] = 1

# els results
results_els = {'target': correct_ids[:, 2], 'prediction': inferred_ids[:, 2]}
results_els = pandas.DataFrame(results_els)
results_els['dummy'] = 1

# save results
output_data = {}
output_data['results_lcs'] = results_lcs
output_data['results_azs'] = results_azs
output_data['results_els'] = results_els
misc.pickle_save(output_file, output_data)
print('done')