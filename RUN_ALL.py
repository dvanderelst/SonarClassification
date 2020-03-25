def run(file_name):
    file_name = file_name + '.py'
    with open(file_name) as f:
        code = compile(f.read(), file_name, 'exec')
        exec(code)

#%%
import script_preprocess_empirical_data

# Get space based on synthetic echoes
run('script_synthetic_echoes_to_PCA')
run('script_plot_processing_synthetic')
#%%
# Visualize reconstruction of syn and emp templates
run('script_plot_template_examples')
#%%Train the neural networks
run('script_run_training')
run('script_plot_training_history')

#%%

run('script_run_perfect_memory')
#%%
import script_plot_training_results
run('script_plot_training_results')
#%%
# Assess where the reduction happens
run('script_run_identify_mechanism')

#%%
run('script_evaluate_interpolation')

#%%

run('script_tabulate_memory_use')
run('script_plot_components')

