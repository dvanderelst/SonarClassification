import math
import numpy
from library import settings

from library import generate_functions
from pyBat import Call
from pyBat import Wiegrebe

repeats = 10

n_seed_levels = [5, 10, 20, 50, 100, 200]
n_cloud_levels = [5, 10, 20, 50, 100, 200]

sample_frequency = settings.sample_frequency
number_of_samples = settings.raw_collected_samples
initial_zero_time = settings.initial_zero_time
emission_frequency_mean = settings.emission_frequency_mean
initial_zero_samples = math.ceil(initial_zero_time * sample_frequency)

caller = Call.Call(None, [emission_frequency_mean])
emission = generate_functions.create_emission()

emission_padding = number_of_samples - emission.shape[0]
padded_emission = numpy.pad(emission, (0, emission_padding), 'constant')

wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4, emission=padded_emission)

synthetic_impulses = []
synthetic_echoes = []
synthetic_templates = []

for seed_level_i in range(len(n_seed_levels)):
    for cloud_level_i in range(len(n_cloud_levels)):
        print(seed_level_i, cloud_level_i)
        for i in range(repeats):
            generative_model_parameters = {}
            generative_model_parameters['n_seed_points'] = n_seed_levels[seed_level_i]
            generative_model_parameters['n_cloud_points'] = n_cloud_levels[cloud_level_i]
            distances = generate_functions.generative_model(generative_model_parameters)
            echo_sequence, impulse_response = generate_functions.distances2echo_sequence(distances, caller, emission)
            echo_sequence[0:initial_zero_samples] = 0
            template = generate_functions.echo_sequence2template(echo_sequence, wiegrebe)
            synthetic_echoes.append(echo_sequence)
            synthetic_templates.append(template)

print('---> SAVE ECHOES AND TEMPLATES')

synthetic_impulses = numpy.array(synthetic_impulses)
synthetic_echoes = numpy.array(synthetic_echoes)
synthetic_templates = numpy.array(synthetic_templates)

numpy.savez_compressed(settings.synthetic_impulses_file, synthetic_impulses=synthetic_impulses)
numpy.savez_compressed(settings.synthetic_echoes_file, synthetic_echoes=synthetic_echoes)
numpy.savez_compressed(settings.synthetic_templates_file, synthetic_templates=synthetic_templates)

# %%
import importlib
import numpy
from library import settings
from library import misc
from matplotlib import pyplot


importlib.reload(settings)

print('-----> LOADING EMPIRICAL DATA')
empirical_templates = misc.load_all_templates()

print('-----> LOADING SYNTHETIC DATA')
loaded = numpy.load(settings.synthetic_echoes_file)
synthetic_echoes = loaded['synthetic_echoes']

loaded = numpy.load(settings.synthetic_templates_file)
synthetic_templates = loaded['synthetic_templates']

print('-----> SCALE SYNTHETIC TEMPLATES')

# Plot means
mean_empirical = numpy.mean(empirical_templates, axis=0)
mean_synthetic = numpy.mean(synthetic_templates, axis=0)

pyplot.plot(mean_empirical)
pyplot.plot(mean_synthetic)
pyplot.legend(['Empirical', 'Synthetic'])
pyplot.show()

# Plot correct mean and offset and plot again
r_empirical = numpy.max(mean_empirical) - numpy.min(mean_empirical)
r_synthetic = numpy.max(mean_synthetic) - numpy.min(mean_synthetic)
scale = r_empirical / r_synthetic
offset = numpy.min(mean_empirical)
synthetic_templates = synthetic_templates * scale + offset

# Plot again
mean_empirical = numpy.mean(empirical_templates, axis=0)
mean_synthetic = numpy.mean(synthetic_templates, axis=0)
pyplot.plot(mean_empirical)
pyplot.plot(mean_synthetic)
pyplot.legend(['Empirical', 'Synthetic'])
pyplot.show()

print('-----> RUN PCAs FOR SYNTHETIC DATA')
pca_model_echoes = PCA()
transformed_synthetic_echoes = pca_model_echoes.fit_transform(synthetic_echoes)
reconstructed_synthetic_echoes = pca_model_echoes.inverse_transform(transformed_synthetic_echoes)
pca_model_echoes.fit_transform(synthetic_echoes)
cummmulative_echoes = numpy.cumsum(pca_model_echoes.explained_variance_ratio_)
suggestion_echoes = numpy.min(numpy.where(cummmulative_echoes > 0.99)[0])
print('+ SUGGESTED NR OF PCs for waveforms:', suggestion_echoes)

pca_model_templates = PCA()
transformed_synthetic_templates = pca_model_templates.fit_transform(synthetic_templates)
reconstructed_synthetic_templates = pca_model_templates.inverse_transform(transformed_synthetic_templates)
cummmulative_templates = numpy.cumsum(pca_model_templates.explained_variance_ratio_)
suggestion_templates = numpy.min(numpy.where(cummmulative_templates > 0.99)[0])
print('+ SUGGESTED NR OF PCs for templates:', suggestion_templates)

print('------> RUN PCAs FOR COMPARISON WITH EMPIRICAL DATA')
transformed_empirical_templates = pca_model_templates.transform(empirical_templates)
reconstructed_empirical_templates = pca_model_templates.inverse_transform(transformed_empirical_templates)

print('---> GET CORRELATIONS')

x = reconstructed_synthetic_templates.flatten()
y = synthetic_templates.flatten()

n = x.size
selected = numpy.random.randint(0, n, [1,10000])
x = x[selected]
y = y[selected]

r0 = numpy.corrcoef(x, y)[0, 1]

x = reconstructed_empirical_templates.flatten()
y = empirical_templates.flatten()

n = x.size
selected = numpy.random.randint(0, n, [1,10000])
x = x[selected]
y = y[selected]

r1 = numpy.corrcoef(x, y)[0, 1]

print('+ Correlations:', r0, r1)

print('---> SAVE DATA')
misc.pickle_save(settings.pca_templates_file, pca_model_templates)
misc.pickle_save(settings.pca_echoes_file, pca_model_echoes)
numpy.savez_compressed(settings.scaled_synthetic_templates_file, synthetic_templates=synthetic_templates)
numpy.savez_compressed(settings.reconstructed_synthetic_templates_file, reconstructed_synthetic_templates=reconstructed_synthetic_templates)
numpy.savez_compressed(settings.reconstructed_synthetic_echoes_file, reconstructed_synthetic_echoes=reconstructed_synthetic_echoes)

print('---> PLOTS')

pyplot.figure()
pyplot.subplot(1,2,1)
pyplot.plot(cummmulative_echoes)
pyplot.plot(cummmulative_templates)

pyplot.show()