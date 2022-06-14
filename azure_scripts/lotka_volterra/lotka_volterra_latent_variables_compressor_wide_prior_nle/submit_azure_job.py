from azureml.core import (ComputeTarget, Environment, Experiment,
                          ScriptRunConfig, Workspace, runconfig)

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
import time

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')


environment=runconfig.RunConfiguration()
environment.environment=Environment.get(workspace=ws, name="tf27-jax-py38-cuda11-gpu", version="6")


# for model_seed in range(2):
# #for model_seed in range(1):
#     for n_simulations in [20,50,100,200,500,1000,10000,100000]:
#     #for n_simulations in [20]:
     
#         # Configure the ScriptRunConfig and specify the compute cluster and environment
#         src = ScriptRunConfig(
#             source_directory="azure_scripts", 
#             script="lotka_volterra_with_latent_variables.py",
#             arguments=[
#                 "--batch_size", 256,
#                 "--n_simulations",n_simulations,
#                 "--n_epochs",40, # not needed
#                 "--n_steps",20000,
#                 "--dimension", 4, # not needed
#                 "--bijector_layers_size", 128, # not needed
#                 "--bijector_layers_shape", 2, # not needed
#                 "--nf_layers", 3, # not needed
#                 "--n_components", 32, # not needed
#                 "--score_weight", 1e-7, #0 or 1e-7
#                 "--model_seed",int(model_seed+8),
#                 "--initial_learning_rate",0.002, # not needed
                
#             ],
#             compute_target=ComputeTarget(name="justine-k80", workspace=ws), 
#             environment=Environment.get(workspace=ws, name="tf27-jax-py38-cuda11-gpu", version="18"))

#         # Create a new experiment
#         exp = Experiment(name=f"060622_nle_score", workspace=ws)

#         # Submit run 
#         run = exp.submit(src)

#         time.sleep(5)

# Print info in the terminal until completion 
# Useful for quick feedback on a short job, otherwise better to comment it out
# run.wait_for_completion(show_output=True)





#data_store = ws.get_default_datastore()
#params = OutputFileDatasetConfig('params_nf')


for model_seed in range(20):
#for model_seed in range(1):
    for n_simulations in [20,50,100,200,500,1000,10000,100000]:

      params=PipelineData(f"paramsnd{model_seed}b{n_simulations}")
      
      #output=PipelineData('output', is_directory=True)

      # Step to run a Python script
      step1 = PythonScriptStep(name = 'train nf',
                              source_directory = 'azure_scripts',
                              script_name = 'lotka_volterra_with_latent_variables.py',
                              compute_target = 'justine-k80', 
                              arguments = [
                                 "--batch_size", 256,
                                 "--n_simulations",n_simulations,
                                 "--n_epochs",40, # not needed
                                 "--n_steps",20000,
                                 "--dimension", 4, # not needed
                                 "--bijector_layers_size", 128, # not needed
                                 "--bijector_layers_shape", 2, # not needed
                                 "--nf_layers", 3, # not needed
                                 "--n_components", 32, # not needed
                                 "--score_weight", 0, #0 or 1e-7
                                 "--model_seed",model_seed,
                                 "--initial_learning_rate",0.002 ,
                                 #"--output_dir", output,
                                 "--output_file", params
                              ],
                              runconfig=environment,
                              #outputs=[params,output]
                              outputs=[params]
      )
                              

      # Step to train a model
      step2 = PythonScriptStep(name = 'run mcmc',
                              source_directory = 'azure_scripts',
                              script_name = 'mcmc.py',
                              compute_target = 'justine-D4',
                              arguments = [
                                 #"--input_dir", output,
                                 "--input_file", params,],
                              runconfig=environment,
                              # inputs=[params,output]
                              inputs=[params]
                        
                              )

      # Construct the pipeline
      train_pipeline = Pipeline(workspace = ws, steps = [step1,step2])

      # Create an experiment and run the pipeline
      exp = Experiment(workspace = ws, name = '120622_lotvolt_latvar_cprs_nle_wide_prior_without_score')
      pipeline_run = exp.submit(train_pipeline)

      time.sleep(5)



