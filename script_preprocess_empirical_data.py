from library import process_functions

process_functions.process_data_set('royal')
process_functions.process_data_set('israel')

process_functions.process_data_set('royal', interpolate_directions=True)
process_functions.process_data_set('israel', interpolate_directions=True)
