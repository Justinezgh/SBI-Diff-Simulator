import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
import numpyro.distributions as dist

__all__=['lensingLogNormal']

def make_power_map(pk_fn, N, field_size, zero_freq_val=0.0):
  k = 2*jnp.pi*jnp.fft.fftfreq(N, d=field_size / N)
  kcoords = jnp.meshgrid(k,k)
  k = jnp.sqrt(kcoords[0]**2 + kcoords[1]**2)
  ps_map = pk_fn(k)
  ps_map = ps_map.at[0,0].set(zero_freq_val)
  return ps_map * (N / field_size)**2

def make_lognormal_power_map(power_map, shift, zero_freq_val=0.0):
  power_spectrum_for_lognorm = jnp.fft.ifft2(power_map).real
  power_spectrum_for_lognorm = jnp.log(1 + power_spectrum_for_lognorm/shift**2)
  power_spectrum_for_lognorm = jnp.abs(jnp.fft.fft2(power_spectrum_for_lognorm))
  power_spectrum_for_lognorm = power_spectrum_for_lognorm.at[0,0].set(0.)
  return power_spectrum_for_lognorm

def lensingLogNormal(N=64,               # number of pixels on the map
          map_size=10,         # map size in deg.
          gal_per_arcmin2=30,   
          sigma_e=0.2, 
          shift=0.05,
          model_type='lognormal'): # either 'lognormal' or 'gaussian'      
    
    pix_area = (map_size * 60 / N)**2 # arcmin2 
    map_size = map_size / 180 * jnp.pi    # radians

    theta = numpyro.sample('theta', dist.Independent(dist.Normal(jnp.array([0.3,0.8]), 0.05*jnp.ones(2)),1))
    
    cosmo = jc.Planck15(Omega_c=theta[0], sigma8=theta[1])
    # Creating a given redshift distribution
    pz = jc.redshift.smail_nz(0.5, 2., 1.0)
    tracer = jc.probes.WeakLensing([pz])
    
    # Defining the function that will compute the power spectrum of the field
    # Create an interpolation array for the cls to avoid far too many computations
    ell_tab = jnp.logspace(0, 4.5, 128)
    cell_tab = jc.angular_cl.angular_cl(cosmo, ell_tab, [tracer])[0]
    P = lambda k: jc.scipy.interpolate.interp(k.flatten(), ell_tab, cell_tab).reshape(k.shape)
    
    # Sampling latent variables
    z = numpyro.sample('z', dist.MultivariateNormal(loc=jnp.zeros((N,N)), precision_matrix=jnp.eye(N)))

    # Convolving by the power spectrum
    power_map = make_power_map(P, N, map_size) 
    if model_type == 'lognormal':
      power_map =  make_lognormal_power_map(power_map, shift)

    field = jnp.fft.ifft2(jnp.fft.fft2(z) * jnp.sqrt(power_map)).real

    if model_type == 'lognormal':
      field = shift * (jnp.exp(field - jnp.var(field) / 2) - 1)

    # Adding "observational noise"
    y = numpyro.sample('y', dist.Independent(dist.Normal(field, sigma_e/jnp.sqrt(gal_per_arcmin2 * pix_area)), 2))
    
    return y