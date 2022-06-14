from azureml.core import (ComputeTarget, Environment, Experiment,
                          ScriptRunConfig, Workspace)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

# Configure the ScriptRunConfig and specify the compute cluster and environment
src = ScriptRunConfig(
    source_directory="azure_scripts", 
    script="lotka_volterra.py",
    arguments=[
        "--batch_size", 1000,
        "--n_simulations", 5e5,
        "--n_epochs", 5,
        "--dimension", 4,
        "--bijector_layers_size", 128,
        "--bijector_layers_shape", 2,
        "--nf_layers", 3,
        "--n_components", 32,
        "--score_weight", 0.0
    ],
    compute_target=ComputeTarget(name="justine-k80", workspace=ws), 
    environment=Environment.get(workspace=ws, name="tf27-jax-py38-cuda11-gpu", version="6"))

# Create a new experiment
exp = Experiment(name=f"lotka-volterra", workspace=ws)

# Submit run 
run = exp.submit(src)

# Print info in the terminal until completion 
# Useful for quick feedback on a short job, otherwise better to comment it out
# run.wait_for_completion(show_output=True)
