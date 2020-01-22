import os
import pickle
import settings
import natsort
import numpy
from matplotlib import pyplot
from sklearn.decomposition import PCA

import process_functions

# %%Preprocess andrews
files = os.listdir("andrews")
files = natsort.natsorted(files)

test_data, _, _ = process_functions.preprocess(os.path.join('andrews', 'clutter1.mat'), verbose=False)
n_samples = test_data.shape[2]

all_mns = numpy.zeros([12, 7, 31, n_samples])
all_vrs = numpy.zeros([12, 7, 31, n_samples])

for i in range(len(files)):
    data, az, el = process_functions.preprocess(os.path.join('andrews', files[i]))
    mn, vr = process_functions.get_mean_and_variance(data)
    all_mns[i, :, :, :] = mn
    all_vrs[i, :, :, :] = vr

long = numpy.reshape(all_mns, (12 * 7 * 31, n_samples))
sample_variation = numpy.mean(all_vrs, axis=(0, 1, 2))

numpy.savez('andrews_data.npz',
            means=all_mns,
            variation=all_vrs,
            long=long,
            sample_variation = sample_variation,
            files=files)

# %%
pca_model = PCA()
pca_model.fit(long)

s = pickle.dumps(pca_model)
f = open('andrews_pca.pck', 'wb')
f.write(s)
f.close()

cme = numpy.cumsum(pca_model.explained_variance_ratio_)
pyplot.subplot(1,2,1)
pyplot.plot(cme)
pyplot.subplot(1,2,2)
pyplot.plot(sample_variation)
pyplot.show()