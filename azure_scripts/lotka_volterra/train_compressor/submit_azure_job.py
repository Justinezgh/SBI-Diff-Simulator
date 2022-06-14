from azureml.core import (ComputeTarget, Environment, Experiment,
                          ScriptRunConfig, Workspace)
import time

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

for comp_nf_layers in [3]:
    for normalization_compressor in [30]:
        for normalization_reg in [0.04]:
            for model_seed in range(1):
                for n_simulations in [100000]:
                    # Configure the ScriptRunConfig and specify the compute cluster and environment
                    src = ScriptRunConfig(
                        source_directory="azure_scripts", 
                        script="lotka_volterra_with_latent_variables.py",
                        arguments=[
                            "--batch_size", 256, #not needed
                            "--n_simulations",n_simulations, # not needed
                            "--n_epochs",40, # not needed
                            "--n_steps",20000, #not needed
                            "--dimension", 4, # not needed
                            "--bijector_layers_size", 128, # not needed
                            "--bijector_layers_shape", 2, # not needed
                            "--nf_layers", 3, # not needed
                            "--n_components", 32, # not needed
                            "--score_weight", 0, #0 or 1e-7 not needed
                            "--model_seed",model_seed, #not needed
                            "--initial_learning_rate",0.002, # not needed
                            "--normalization_compressor",normalization_compressor,
                            "--normalization_reg",normalization_reg,
                            "--comp_nf_layers", comp_nf_layers
                            
                        ],
                        compute_target=ComputeTarget(name="justine-k80", workspace=ws), 
                        environment=Environment.get(workspace=ws, name="tf27-jax-py38-cuda11-gpu", version="6"))

                    # Create a new experiment
                    exp = Experiment(name=f"train_compressor", workspace=ws)

                    # Submit run 
                    run = exp.submit(src)

                    time.sleep(5)

# Print info in the terminal until completion 
# Useful for quick feedback on a short job, otherwise better to comment it out
# run.wait_for_completion(show_output=True)
