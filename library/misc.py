import os
import pickle
import re
from library import settings
import numpy
import scipy.interpolate as interpolate
from matplotlib import pyplot
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
        match = re.match(expression, text, re.M | re.I)
        match = float(match[0])
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
    base_name = data_set + '_' + str(dimension)
    if dimension is None: base_name = data_set

    npz_file = os.path.join(result_folder, data_set + '.npz')
    pca_file = os.path.join(result_folder, data_set + '.pca')

    model_file = os.path.join(result_folder, base_name + '.5h')
    res_file = os.path.join(result_folder, base_name + '.pd')
    hist_file = os.path.join(result_folder, base_name + '.hist')

    perf_file = os.path.join(result_folder, data_set + '.pm')

    log_folder = os.path.join(result_folder, 'log_' + base_name)

    result['data_folder'] = data_set
    result['result_folder'] = result_folder
    result['log_folder'] = log_folder
    result['npz_file'] = npz_file
    #result['pca_file'] = pca_file
    result['model_file'] = model_file
    result['results_file'] = res_file
    result['history_file'] = hist_file
    result['perfect_memory_file'] = perf_file
    result['base_name'] = base_name
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
    values = results.target.unique()
    values = numpy.sort(values)
    n = len(values)
    table = numpy.zeros((n, n))
    for ti in range(n):
        for pi in range(n):
            tv = values[ti]
            pv = values[pi]
            selected = results.query("target==@tv and prediction==@pv")
            selected_n = selected.shape[0]
            #print(selected_n)
            table[ti, pi] = selected_n
    labels = values
    table[numpy.isnan(table)] = 0
    if normalize: table = N.fit_transform(table) ** 2
    if not normalize: table = numpy.array(table)
    return table, labels


def label_confusion_matrix(labels):
    if len(labels) == 7: ticks = list(range(0, 7, 2))
    if len(labels) == 31: ticks = list(range(6, 31, 6))
    if len(labels) == 50: ticks = list(range(10, 50, 10))
    if len(labels) == 40: ticks = list(range(10, 40, 10))
    ticks_locs = numpy.array(ticks)


    new_labels = []
    for x in labels: new_labels.append('%.1f'%x)
    new_labels = numpy.array(new_labels)

    print(pyplot.xticks())

    pyplot.xticks(ticks_locs, new_labels[ticks_locs])
    pyplot.yticks(ticks_locs, new_labels[ticks_locs])

    pyplot.xlim([-0.5, len(labels)-0.5])
    pyplot.ylim([-0.5, len(labels)-0.5])


def get_error_histogram(results, normalize=True, cummmulative=False):
    errors = results['prediction'] - results['target']
    if cummmulative: errors = numpy.abs(errors)
    counts = errors.value_counts(sort=False, normalize=normalize)
    counts = counts.reset_index()
    counts.columns = ['error', 'number']
    counts = counts.sort_values(by=['error'])
    if cummmulative: counts['number'] = numpy.cumsum(counts['number'])
    return counts


def zlim(lims):
    try:
        ax = pyplot.gca()
        ax.set_zlim(lims[0], lims[1])
    except:
        pyplot.set


def plot_inference_lines(error, number, xs):
    f = interpolate.interp1d(error, number)
    ys = f(xs)
    ax = pyplot.gca()
    legend_entries = []
    i = 0
    for (x, y) in zip(xs, ys):
        string_x = "x=%4.2f"%x
        string_y = "y=%4.2f"%y
        string = string_x + ', ' + string_y
        color = settings.qualitative_colors[i, :]
        pyplot.plot([x, x], [0, y], '--', color=color, alpha=1)
        pyplot.plot([0, x], [y, y], '--', color=color, alpha=1, label='_nolegend_')
        legend_entries.append(string)
        i+=1
    #print(list(pyplot.xticks()[0]))
    #pyplot.xticks(list(pyplot.xticks()[0]) + list(xs))
    #pyplot.yticks(list(pyplot.yticks()[0]) + list(ys))

    return ys, legend_entries
