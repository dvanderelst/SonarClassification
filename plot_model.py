from tensorflow import keras
from library import misc, settings
import os

files = misc.folder_names('israel', 'azs')
model = keras.models.load_model(files['model_file'])

output = os.path.join(settings.figure_folder, 'model.pdf')

keras.utils.plot_model(
    model,
    to_file=output,
    show_shapes=True,
    show_layer_names=False,
    rankdir='TB',
    expand_nested=True,
    dpi=96
)
