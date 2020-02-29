from library import misc
from library import settings
from matplotlib import pyplot
import pandas
import numpy


echoes = misc.pickle_load(settings.synthetic_echoes_file)
indices = misc.pickle_load(settings.synthetic_echoes_indices)
templates_synthetic = numpy.load(settings.synthetic_templates_file)

pca_model = misc.pickle_load(settings.pca_file)


transformed_synthetic = pca_model.fit_transform(templates_synthetic)
reconstructed_synthetic = pca_model.inverse_transform(transformed_synthetic)
#%%
n = reconstructed_synthetic.shape[0]
keys = list(echoes.keys())

correlations = []
n_seeds = []
n_clouds = []
n_repeats = []

for i in range(n):
    k = keys[i]
    k = k.split('_')
    n_seed = int(k[0])
    n_cloud = int(k[1])
    n_repeat = int(k[2])

    x = reconstructed_synthetic[i, :]
    y = templates_synthetic[i, :]
    r = numpy.corrcoef(x, y)[0, 1]

    correlations.append(r)
    n_seeds.append(n_seed)
    n_clouds.append(n_cloud)
    n_repeats.append(n_repeat)

results = {'n_seed': n_seeds, 'n_cloud': n_clouds, 'n_repeat': n_repeats, 'corr': correlations}
results = pandas.DataFrame(results)
#%%
grp = results.groupby(['n_seed', 'n_cloud'])
mns = grp.mean()
mns = mns.reset_index()

table = mns.pivot_table(index='n_seed', columns='n_cloud', values='corr')
print(table)

pyplot.imshow(table)
pyplot.show()