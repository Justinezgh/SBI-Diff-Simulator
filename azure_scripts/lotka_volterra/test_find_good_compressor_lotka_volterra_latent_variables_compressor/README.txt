Find the good compressor architecture to compress lotka Volterra's observations into 4 summary statistics using mutual information. 

Compressors are trained to approximate the posterior in the specific case: 

Task: lotka volterra
Latent variables: yes
Prior: wide
Sbi method: npe without score  



BEST ARCHITECTURE: 
normalization_compressor = 30
normalization_reg = 0.04
comp_nf_layers = 3


Job name on azure: test_compressor2