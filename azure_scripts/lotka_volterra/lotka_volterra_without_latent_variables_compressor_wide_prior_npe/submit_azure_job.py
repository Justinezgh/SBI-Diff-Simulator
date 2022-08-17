from azureml.core import (ComputeTarget, Environment, Experiment,
                          ScriptRunConfig, Workspace)
import time

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

for model_seed in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
    for n_simulations in [20,50,100,200,500,1000,10000,100000]:
    
        # Configure the ScriptRunConfig and specify the compute cluster and environment
        src = ScriptRunConfig(
            source_directory="azure_scripts", 
            script="lotka_volterra_without_latent_variables.py",
            arguments=[
                "--batch_size", 256,
                "--n_simulations",n_simulations,
                "--n_epochs",40, # not needed
                "--n_steps",20000,
                "--dimension", 4, # not needed
                "--bijector_layers_size", 128, # not needed
                "--bijector_layers_shape", 2, # not needed
                "--nf_layers", 3, # not needed
                "--n_components", 32, # not needed
                "--score_weight",1e-7, # 0 or 1e-7
                "--model_seed",model_seed,
                "--initial_learning_rate",0.002, # not needed
                
            ],
            compute_target=ComputeTarget(name="justine-k80", workspace=ws), 
            environment=Environment.get(workspace=ws, name="tf27-jax-py38-cuda11-gpu", version="6"))

        # Create a new experiment
        exp = Experiment(name=f"100622_lotvolt_without_latvar_newcprsr_sim_score", workspace=ws)

        # Submit run 
        run = exp.submit(src)

        time.sleep(5)

# Print info in the terminal until completion 
# Useful for quick feedback on a short job, otherwise better to comment it out
# run.wait_for_completion(show_output=True)
