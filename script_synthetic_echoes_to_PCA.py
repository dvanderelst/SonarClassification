import numpy
from matplotlib import pyplot
from library import generate_functions
from library import settings
from pyBat import Call
from pyBat import Wiegrebe

pyplot.style.use(settings.style)

reflector_distances = numpy.linspace(1,6, 500)

print('---> CREATE SYNTHETIC TEMPLATES')

sample_frequency = settings.sample_frequency
number_of_samples = settings.raw_collected_samples
initial_zero_time = settings.initial_zero_time
emission_frequency_mean = settings.emission_frequency_mean

caller = Call.Call(None, [emission_frequency_mean])
emission = generate_functions.create_emission()

emission_padding = settings.raw_collected_samples - emission.shape[0]
padded_emission = numpy.pad(emission, (0, emission_padding), 'constant')

wiegrebe = Wiegrebe.ModelWiegrebe(sample_frequency, emission_frequency_mean, 4, emission=padded_emission)

templates_synthetic = []
echoes_synthetic = []

for reflector_distance in reflector_distances:
    print('Current distance:', reflector_distance)
    # Make echo sequence
    distances = numpy.array([reflector_distance])
    echo_sequence = generate_functions.distances2echo_sequence(distances, caller, emission)

    # Create template
    template = generate_functions.echo_sequence2template(echo_sequence, wiegrebe)
    templates_synthetic.append(template)
    echoes_synthetic.append(echo_sequence)

print('---> SAVE ECHOES AND TEMPLATES')

templates_synthetic = numpy.array(templates_synthetic)
echoes_synthetic = numpy.array(echoes_synthetic)

numpy.savez_compressed(settings.synthetic_templates_file, templates_synthetic=templates_synthetic)
numpy.savez_compressed(settings.synthetic_echoes_file, echoes_synthetic=echoes_synthetic)
# %%
import importlib
import numpy
from library import settings
from library import misc
from matplotlib import pyplot
from sklearn.decomposition import PCA

importlib.reload(settings)

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
loaded = numpy.load(settings.synthetic_templates_file)
templates_synthetic = loaded['templates_synthetic']

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

print('------> RUN FOR ALL PCs')
pca_model = PCA()
transformed_synthetic = pca_model.fit_transform(templates_synthetic)
cummmulative = numpy.cumsum(pca_model.explained_variance_ratio_)
pyplot.plot(cummmulative)
pyplot.show()

suggestion = numpy.min(numpy.where(cummmulative>0.99)[0])
print('------> SUGGESTED NR OF PCs:', suggestion)

print('------> RUN FOR SELECTED PCs')
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

print('------> Correlations:', r0, r1)

print('---> SAVE DATA')
numpy.savez_compressed(settings.scaled_synthetic_templates_file, templates_synthetic=templates_synthetic)
numpy.savez_compressed(settings.reconstructed_synthetic_templates_file, reconstructed_synthetic=reconstructed_synthetic)

#Dont save this, as it results in a big file that takes ages to sync.
#numpy.savez_compressed(settings.reconstructed_templates_file, reconstructed_empirical=reconstructed_empirical)

print('---> DONE')