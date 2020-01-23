import numpy


def binarize_prediction(prediction):
    n = prediction.shape[0]
    new = numpy.zeros(prediction.shape)
    maxes = numpy.argmax(prediction, axis=1)
    for i in range(n): new[i, maxes[i]] = 1
    return new
