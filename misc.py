import pickle
import pandas
import numpy
import os
import re
from sklearn.preprocessing import Normalizer


def map_lcs_to_distances(data):
    distances = extract_distances_from_filenames(data['files'])
    long_lcs = data['long_lcs']
    new_long_lcs = long_lcs.copy()
    new_long_lcs = new_long_lcs.astype(float)
    lcs_unique = numpy.unique(long_lcs)
    for lc, dist in zip(lcs_unique, distances):
        indices = numpy.where(long_lcs == lc)[0]
        new_long_lcs[indices] = dist
    return new_long_lcs

def extract_distances_from_filenames(filenames):
    distances = []
    for text in filenames:
        expression = "[+-]?[0-9]+\.?[0-9]*"
        match = re.match( expression, text, re.M|re.I)
        match =  float(match[0])
        distances.append(match)
    return distances


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
    pca_file = os.path.join(result_folder, data_set + '.pca')

    model_file = os.path.join(result_folder, data_set + '_' + dimension + '.5h')
    res_file = os.path.join(result_folder, data_set + '_' + dimension + '.pd')
    hist_file = os.path.join(result_folder, data_set + '_' + dimension + '.hist')

    log_folder = os.path.join(result_folder, 'log_' + data_set + '_' + dimension)

    result['data_folder'] = data_set
    result['result_folder'] = result_folder
    result['log_folder'] = log_folder
    result['npz_file'] = npz_file
    result['pca_file'] = pca_file
    result['model_file'] = model_file
    result['results_file'] = res_file
    result['history_file'] = hist_file
    return result


def pickle_save(file, object):
    f = open(file, 'wb')
    pickle.dump(object, f)
    f.close()


def pickle_load(file):
    f = open(file, 'rb')
    object = pickle.load(f)
    f.close()
    return object


def make_confusion_matrix(results, normalize=True):
    N = Normalizer()
    grp = results.groupby(['target', 'prediction'])
    counts = grp.sum()
    counts = counts.reset_index()
    table = counts.pivot(index='target', columns='prediction', values='dummy')
    labels = list(table.index)
    table[numpy.isnan(table)] = 0
    if normalize: table = N.fit_transform(table) ** 2
    if not normalize: table = numpy.array(table)
    return table, labels

def get_error_histogram(results, normalize=True, cummmulative=False):
    errors = results['prediction'] - results['target']
    if cummmulative: errors = numpy.abs(errors)
    counts = errors.value_counts(sort=False, normalize=normalize)
    counts = counts.reset_index()
    counts.columns = ['error', 'number']
    counts = counts.sort_values(by=['error'])
    if cummmulative: counts['number'] = numpy.cumsum(counts['number'])
    return counts