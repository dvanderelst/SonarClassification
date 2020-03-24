from library import process_functions, misc
import numpy

#process_functions.process_data_set('royal')
process_functions.process_data_set('israel')

#process_functions.process_data_set('royal', interpolate_directions=True)
process_functions.process_data_set('israel', interpolate_directions=True)

#%%
from matplotlib import pyplot

data_set = 'israel'
file_names = misc.folder_names(data_set, None)


data1 = numpy.load(file_names['npz_file'])
data1 = data1['long_data']
print(data1.shape)

data2 = numpy.load(file_names['npz_file_interpolated'])
data2 = data2['long_data']
print(data2.shape)

i = 250

pyplot.plot(data1[i,:])
pyplot.plot(data2[i,:])
pyplot.show()
