import math
import numpy

from library import misc
from matplotlib import pyplot
from library import generate_functions
from library import settings
from pyBat import Call
from pyBat import Wiegrebe

from sklearn.decomposition import PCA

repeats = 50
n_seed_levels = [50, 100, 150, 200]
n_cloud_levels = [50, 50, 50, 50]

print('---> LOAD ANDREWS DATA')

data_set = 'andrews'
file_names = misc.folder_names(data_set, None)
data = numpy.load(file_names['npz_file'])
templates_andrews = data['long_data']

print('---> CREATE SYNTHETIC TEMPLATES')

sample_frequency = settings.sample_frequency
number_of_samples = settings.raw_collected_samples
initial_zero_time = settings.initial_zero_time
emission_frequency_mean = settings.emission_frequency_mean

integration_samples = math.ceil(sample_frequency * settings.integration_time)

caller = Call.Call(None, [emission_frequency_mean])
wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4)
emission = generate_functions.create_emission()

templates_synthetic = []

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
            templates_synthetic.append(template)

templates_synthetic = numpy.array(templates_synthetic)
numpy.save(settings.synthetic_templates_file, templates_synthetic)

print('---> SCALE SYNTHETIC TEMPLATES')

mean_andrews = numpy.mean(templates_andrews, axis=0)
mean_synthetic = numpy.mean(templates_synthetic, axis=0)

pyplot.plot(mean_andrews)
pyplot.plot(mean_synthetic)
pyplot.show()

r_andrews = numpy.max(mean_andrews) - numpy.min(mean_andrews)
r_synthetic = numpy.max(mean_synthetic) - numpy.min(mean_synthetic)
scale = r_andrews / r_synthetic

templates_synthetic = templates_synthetic * scale
templates_synthetic = templates_synthetic + settings.noise_floor
mean_andrews = numpy.mean(templates_andrews, axis=0)
mean_synthetic = numpy.mean(templates_synthetic, axis=0)

pyplot.plot(mean_andrews)
pyplot.plot(mean_synthetic)
pyplot.show()

print('---> RUN AND SAVE PCA MODEL')

pca_model = PCA(n_components=settings.n_components)
transformed_synthetic = pca_model.fit_transform(templates_synthetic)

misc.pickle_save(settings.pca_file, pca_model)

reconstructed_synthetic = pca_model.inverse_transform(transformed_synthetic)

transformed_andrews = pca_model.transform(templates_andrews)
reconstructed_andrews = pca_model.inverse_transform(transformed_andrews)

print('---> GET CORRELATIONS')

x = reconstructed_synthetic.flatten()
y = templates_synthetic.flatten()
r0 = numpy.corrcoef(x, y)[0, 1]

x = reconstructed_andrews.flatten()
y = templates_andrews.flatten()
r1 = numpy.corrcoef(x, y)[0, 1]

print(r0, r1)
