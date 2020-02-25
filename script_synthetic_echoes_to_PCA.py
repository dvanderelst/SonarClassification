import math
import numpy

from library import misc
from matplotlib import pyplot
from library import generate_functions
from library import settings
from pyBat import Call
from pyBat import Wiegrebe

pyplot.style.use(settings.style)

# repeats = 30
# n_seed_levels = [5, 10, 20, 40, 80, 160, 320]
# n_cloud_levels = [5, 10, 20, 40, 80, 160, 320]

repeats = 3
n_seed_levels = [5, 10, 20]
n_cloud_levels = [5, 10, 20]

print('---> CREATE SYNTHETIC TEMPLATES')

sample_frequency = settings.sample_frequency
number_of_samples = settings.raw_collected_samples
initial_zero_time = settings.initial_zero_time
emission_frequency_mean = settings.emission_frequency_mean

integration_samples = math.ceil(sample_frequency * settings.integration_time)

caller = Call.Call(None, [emission_frequency_mean])
emission = generate_functions.create_emission()

emission_padding = settings.raw_collected_samples - emission.shape[0]
padded_emission = numpy.pad(emission, (0, emission_padding), 'constant')

wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4, emission=padded_emission)

templates_synthetic = []
echoes_synthetic = {}
echoes_synthetic_indices = {}
index = 0
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

            # Store echoes
            a = n_seed_levels[seed_level_i]
            b = n_cloud_levels[cloud_level_i]
            c = i
            label = '%i_%i_%i' % (a, b, c)
            echoes_synthetic[label] = echo_sequence
            echoes_synthetic_indices[label] = index

            # Create template
            template = generate_functions.echo_sequence2template(echo_sequence, wiegrebe)
            templates_synthetic.append(template)

            index = index + 1

misc.pickle_save(settings.synthetic_echoes_file, echoes_synthetic)
misc.pickle_save(settings.synthetic_echoes_indices, echoes_synthetic_indices)
templates_synthetic = numpy.array(templates_synthetic)
numpy.save(settings.synthetic_templates_file, templates_synthetic)
# %%
import numpy
from library import settings
from library import misc
from matplotlib import pyplot

from sklearn.decomposition import PCA

print('---> LOAD ISRAEL and ROYAL DATA')

data_set = 'israel'
file_names = misc.folder_names(data_set, None)
data = numpy.load(file_names['npz_file'])
template_israel = data['long_data']

data_set = 'royal'
file_names = misc.folder_names(data_set, None)
data = numpy.load(file_names['npz_file'])
templates_royal = data['long_data']

all_templates = numpy.row_stack((templates_royal, template_israel))

print('---> SCALE SYNTHETIC TEMPLATES')
templates_synthetic = numpy.load(settings.synthetic_templates_file)

# Plot means
mean_empirical = numpy.mean(all_templates, axis=0)
mean_synthetic = numpy.mean(templates_synthetic, axis=0)

pyplot.plot(mean_empirical)
pyplot.plot(mean_synthetic)
pyplot.legend(['Empirical', 'Synthetic'])
pyplot.show()

# Plot correct mean and offset and plot again
r_empirical = numpy.max(mean_empirical) - numpy.min(mean_empirical)
r_synthetic = numpy.max(mean_synthetic) - numpy.min(mean_synthetic)
scale = r_empirical / r_synthetic
offset = numpy.min(mean_empirical)
templates_synthetic = templates_synthetic * scale + offset

# Plot again
mean_empirical = numpy.mean(all_templates, axis=0)
mean_synthetic = numpy.mean(templates_synthetic, axis=0)
pyplot.plot(mean_empirical)
pyplot.plot(mean_synthetic)
pyplot.legend(['Empirical', 'Synthetic'])
pyplot.show()

print('---> RUN AND SAVE PCA MODEL')

pca_model = PCA(n_components=settings.n_components)
transformed_synthetic = pca_model.fit_transform(templates_synthetic)

misc.pickle_save(settings.pca_file, pca_model)

reconstructed_synthetic = pca_model.inverse_transform(transformed_synthetic)

transformed_empirical = pca_model.transform(all_templates)
reconstructed_empirical = pca_model.inverse_transform(transformed_empirical)

print('---> GET CORRELATIONS')

x = reconstructed_synthetic.flatten()
y = templates_synthetic.flatten()
r0 = numpy.corrcoef(x, y)[0, 1]

x = reconstructed_empirical.flatten()
y = all_templates.flatten()
r1 = numpy.corrcoef(x, y)[0, 1]

print(r0, r1)

numpy.save(settings.scaled_synthetic_templates_file, templates_synthetic)
numpy.save(settings.reconstructed_synthetic_templates_file, reconstructed_synthetic)
