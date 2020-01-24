import numpy
import os

def binarize_prediction(prediction):
    n = prediction.shape[0]
    new = numpy.zeros(prediction.shape)
    maxes = numpy.argmax(prediction, axis=1)
    for i in range(n): new[i, maxes[i]] = 1
    return new


def folder_names(data_set, dimension):
    result = {}

    result_folder = data_set + '_results'
    npz_file = os.path.join(result_folder, data_set + '.npz')
    pca_file = os.path.join(result_folder, data_set + '.pck')
    model_file = os.path.join(result_folder, data_set + '_' + dimension + '.5h')

    result['data_folder'] = data_set
    result['result_folder'] = result_folder
    result['npz_file'] = npz_file
    result['pca_file'] = pca_file
    result['model'] = model_file
    return result


