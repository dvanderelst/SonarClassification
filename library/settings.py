
import  numpy

fs = 219000
raw_collected_samples = 7499
noise_floor = 0.18 # should be determined on open space

# It turns out, or so it seems, that the noise before and after the transformation
# by pca has the same sdv. I should check whether this makes mathematical sense.
stochaistic_noise = 0.02 # stdv, See paper for determination, before pca mapping

integration_time = 350 / 1000000  # integration constant
initial_zero_time = 6 / 1000  # initial time to omit

n_components = 20

####################################################
##PLOT SETTINGS
####################################################
import palettable.colorbrewer.qualitative as qualitative
i = numpy.linspace(0,1,4)
qualitative_map = qualitative.Dark2_4.get_mpl_colormap()
qualitative_colors = qualitative_map(i)

figure_folder = '/home/dieter/Dropbox/Apps/ShareLaTeX/Paper_Adarsh/figures'
pca_file = 'results/PCA.pca'

style = 'ggplot'
colormap = 'hot'
azs_color = '#e41a1c'
els_color = '#377eb8'
lcs_color = '#4daf4a'
royal_linestyle = 'dashdot'
israel_linestyle = 'solid'
