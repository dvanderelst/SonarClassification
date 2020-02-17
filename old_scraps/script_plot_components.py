from matplotlib import pyplot
from library import misc, settings

# royal_components are in the rows
# components_array, shape (n_components, n_features)




n = settings.n_components
royal_file_names = misc.folder_names('royal', None)
israel_file_names = misc.folder_names('israel', None)


royal_pca = misc.pickle_load(royal_file_names['pca_file'])
royal_components = royal_pca.components_
royal_components = royal_components[0:n, :]

israel_pca = misc.pickle_load(israel_file_names['pca_file'])
israel_components = israel_pca.components_
israel_components = israel_components[0:n, :]

for i in range(20):
    pyplot.subplot(4,5,i+1)
    pyplot.plot(israel_components[i,:])
    pyplot.plot(royal_components[i,:])
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.title(i)

pyplot.tight_layout()
pyplot.savefig('PCA.pdf')
pyplot.show()
