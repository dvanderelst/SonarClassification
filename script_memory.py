import numpy
import pandas
import os
from matplotlib import pyplot
from tensorflow import keras
import pandas
import misc
import process_functions
import settings


ns = []
ws = []
ws_nb = []
rs = []

for data_set in ['israel', 'royal']:
    selected_dimension = 'lcs'
    file_names = misc.folder_names(data_set, selected_dimension)
    data = numpy.load(file_names['npz_file'])
    templates = data['long_data']

    total_n_numbers = templates.shape[0] * settings.n_components

    new_model = keras.models.load_model(file_names['model_file'])
    weights = new_model.get_weights()
    total_n_weights = 0
    total_n_weights_no_bias = 0
    for w in weights:
        m = numpy.matrix(w)
        total_n_weights = total_n_weights + (m.shape[0] * m.shape[1])
        if m.shape[0] > 1: total_n_weights_no_bias = total_n_weights_no_bias +  (m.shape[0] * m.shape[1])
    total_n_weights = total_n_weights * 3
    total_n_weights_no_bias = total_n_weights_no_bias * 3

    ratio = total_n_weights / total_n_numbers
    string_ratio = '%.1f'%(ratio*100) + '%'

    print(data_set)
    print('numbers in templates', total_n_numbers)
    print('numbers of weights', total_n_weights)
    print('numbers of weights, no bias', total_n_weights_no_bias)
    print('ratio', ratio)


    ns.append(total_n_numbers)
    ws.append(total_n_weights)
    ws_nb.append(total_n_weights_no_bias)
    rs.append(string_ratio)


output = {}
output['Site'] = ['Israel', 'Royal']
output['Template size'] = ns
output['Weights'] = ws
output['Weights (no bias)'] = ws_nb
output['Ratio'] = rs

output = pandas.DataFrame(output)
latex = output.to_latex(index=False)
print(latex)


