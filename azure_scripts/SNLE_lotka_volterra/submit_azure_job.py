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
environment.environment=Environment.get(workspace=ws, name="tf27-jax-py38-cuda11-gpu", version="6") # CHANGER VERSION
# verifier params output file etc 

nrounds = 10
n_steps=20000
score_weight = 0
batch_size = 256
start_grad = 1
seq = 1 # 0: not sequential -> sample from prior 



for model_seed in range(6):
   for sim_budget in [100,200,500,1000,2000]:
   # for sim_budget in [100000]:
      
      # N = sim_budget // nrounds
      steps = []
      params = []
      sample = []
      new_thetas = []

      #init
      r = 0
      params.append(PipelineData(f"paramsnd{model_seed}a{sim_budget}a{r}"))
      sample.append(PipelineData(f"sample{model_seed}a{sim_budget}a{r}"))
      new_thetas.append(PipelineData(f"thetas{model_seed}a{sim_budget}a{r}"))
      # Step to run a Python script
      step1 = PythonScriptStep(name = 'train nf % d' %r,
                              source_directory = 'azure_scripts',
                              script_name = 'train_nf.py',
                              compute_target = 'justine-k80', 
                              arguments = [
                                 "--batch_size", batch_size,
                                 "--n_simulations",sim_budget,
                                 "--n_steps",n_steps,
                                 "--score_weight",0,
                                 "--model_seed",int(model_seed),
                                 "--output_file", params[r],
                                 "--input_file",'' , # juste pcq la osef c'est r = 0
                                 "--sample_in", '', 
                                 "--sample_out", sample[r],
                                 "--new_thetas", '',  # juste pcq la osef c'est r = 0
                                 "--rounds", r, 
                                 "--start_grad",start_grad, 
                                 "--seq", seq
                              ],
                              runconfig=environment,
                              outputs=[params[r], sample[r]]      )

      
                              
      # Step to train a model
      step2 = PythonScriptStep(name = 'run mcmc %d' %r,
                              source_directory = 'azure_scripts',
                              script_name = 'mcmc.py',
                              compute_target = 'justine-D4',
                              arguments = [
                                 "--n_simulations",100,
                                 "--input_file", params[r],
                                 "--thetas", new_thetas[r],
                                 "--rounds", r,
                                 "--model_seed",int(model_seed)],
                              runconfig=environment,
                              inputs=[params[r]], 
                              outputs=[new_thetas[r]]
                        
      )

      steps.append(step2)


      

      for r in range(1, nrounds):

         params.append(PipelineData(f"paramsnd{model_seed}a{sim_budget}a{r}"))
         sample.append(PipelineData(f"sample{model_seed}a{sim_budget}a{r}"))
         new_thetas.append(PipelineData(f"thetas{model_seed}a{sim_budget}a{r}"))

         # Step to run a Python script
         step1 = PythonScriptStep(name = 'train nf % d' %r,
                                 source_directory = 'azure_scripts',
                                 script_name = 'train_nf.py',
                                 compute_target = 'justine-k80', 
                                 arguments = [
                                    "--batch_size", batch_size,
                                    "--n_simulations",100,
                                    "--n_steps",n_steps,
                                    "--score_weight", score_weight,
                                    "--model_seed",int(model_seed),
                                    "--output_file", params[r],
                                    "--input_file", params[r-1],
                                    "--sample_in", sample[r-1], 
                                    "--sample_out", sample[r], 
                                    "--new_thetas", new_thetas[r-1], 
                                    "--rounds", r,
                                    "--start_grad",start_grad,
                                    "--seq", seq
                                 ],
                                 runconfig=environment,
                                 outputs=[params[r], sample[r]], 
                                 inputs=[sample[r-1], new_thetas[r-1], params[r-1]]
         )

         
                                 
         # Step to train a model
         step2 = PythonScriptStep(name = 'run mcmc %d' %r,
                                 source_directory = 'azure_scripts',
                                 script_name = 'mcmc.py',
                                 compute_target = 'justine-D4',
                                 arguments = [
                                    "--n_simulations",100,
                                    "--input_file", params[r],
                                    "--thetas", new_thetas[r],
                                    "--rounds", r,
                                    "--model_seed",int(model_seed)],
                                 runconfig=environment,
                                 inputs=[params[r]], 
                                 outputs=[new_thetas[r]]
                           
         )

         steps.append(step2)

         
      # Construct the pipeline
      train_pipeline = Pipeline(workspace = ws, steps = steps)

      # Create an experiment and run the pipeline
      exp = Experiment(workspace = ws, name = '120922_10round_seq_nle_0score')
      pipeline_run = exp.submit(train_pipeline)

      time.sleep(5)



