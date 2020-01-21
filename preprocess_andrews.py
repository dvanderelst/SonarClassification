import os
import pickle

import natsort
import numpy
from matplotlib import pyplot
from sklearn.decomposition import PCA

import process_functions

# %%Preprocess andrews
files = os.listdir("andrews")
files = natsort.natsorted(files)

test_data, test_az, test_el = process_functions.preprocess(os.path.join('andrews', files[0]))
length_of_subsampled_data = test_data.shape[2]

result = {}
all_mns = numpy.zeros((12, 7, 31, length_of_subsampled_data))
all_vrs = numpy.zeros((12, 7, 31, length_of_subsampled_data))
i = 0
for name in files:
    data, az, el = process_functions.preprocess(os.path.join('andrews', name))
    mn, vr = process_functions.get_mean_and_variance(data)
    all_mns[i, :, :, :] = mn
    all_vrs[i, :, :, :] = vr
    i = i + 1

long = numpy.reshape(all_mns, (12 * 7 * 31, length_of_subsampled_data))
numpy.savez('andrews_data.npz', means=all_mns, variation=all_vrs, long=long, files=files)

# %%
pca_model = PCA()
pca_model.fit(long)

s = pickle.dumps(pca_model)
f = open('andrews_pca.pck', 'wb')
f.write(s)
f.close()

cme = numpy.cumsum(pca_model.explained_variance_ratio_)
pyplot.plot(cme)
pyplot.show()

v = numpy.mean(all_vrs, axis=(0, 1, 2))
pyplot.plot(v)
pyplot.show()
