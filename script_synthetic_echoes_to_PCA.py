import math
import numpy
from library import settings
from library import misc
from library import generate_functions
from pyBat import Call
from pyBat import Wiegrebe

repeats = 10
n_seed_levels = [5, 10, 20, 50, 100, 200]
n_cloud_levels = [5, 10, 20, 50, 100, 200]

#repeats = 3
#n_seed_levels = [5, 10]
#n_cloud_levels = [5, 10]


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

            synthetic_impulses.append(impulse_response)
            synthetic_echoes.append(echo_sequence)
            synthetic_templates.append(template)

synthetic_impulses = numpy.array(synthetic_impulses)
synthetic_echoes = numpy.array(synthetic_echoes)
synthetic_templates = numpy.array(synthetic_templates)

print('---> SCALE SYNTHETIC TEMPLATES')
empirical_templates = misc.load_all_templates()
synthetic_templates = misc.scale_synthetic_templates(empirical_templates, synthetic_templates)

print('---> SAVE ECHOES AND TEMPLATES')

numpy.savez_compressed(settings.synthetic_impulses_file, synthetic_impulses=synthetic_impulses)
numpy.savez_compressed(settings.synthetic_echoes_file, synthetic_echoes=synthetic_echoes)
numpy.savez_compressed(settings.synthetic_templates_file, synthetic_templates=synthetic_templates)

# %%
import importlib
import numpy
from library import settings
from library import misc

importlib.reload(settings)
importlib.reload(misc)

print('---> LOADING SYNTHETIC DATA')
loaded = numpy.load(settings.synthetic_impulses_file)
synthetic_impulses = loaded['synthetic_impulses']

loaded = numpy.load(settings.synthetic_echoes_file)
synthetic_echoes = loaded['synthetic_echoes']

loaded = numpy.load(settings.synthetic_templates_file)
synthetic_templates = loaded['synthetic_templates']

print('---> RUN PCA FOR SYNTHETIC IMPULSE RESPONSES')
output = settings.synthetic_impulses_pca_results
pca_model_impulses, results_impulses = misc.run_pca(synthetic_impulses, criterion=0.99, save_file=output)
misc.pickle_save(settings.pca_impulses_model_file, pca_model_impulses)

print('---> RUN PCA FOR SYNTHETIC ECHOES')
output = settings.synthetic_echoes_pca_results
pca_model_echoes, results_echoes = misc.run_pca(synthetic_echoes, criterion=0.99, save_file=output)
misc.pickle_save(settings.pca_echoes_model_file, pca_model_echoes)

print('---> RUN PCA FOR SYNTHETIC TEMPLATES')
output = settings.synthetic_templates_pca_results
pca_model_templates, results_templates = misc.run_pca(synthetic_templates, criterion=0.99, save_file=output)
misc.pickle_save(settings.pca_templates_model_file, pca_model_templates)

#%%
print('---> LOADING EMPIRICAL DATA')
empirical_templates = misc.load_all_templates()
transformed, reconstructed, correlation = misc.project_and_reconstruct(pca_model_templates, empirical_templates)
