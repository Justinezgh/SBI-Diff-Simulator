
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


from tensorflow_datasets.core.utils import gcs_utils

import jax
from functools import partial
from pathlib import Path
from sbids.tasks.lensinglognormal import lensingLogNormal
from sbids.tasks import get_samples_and_scores

from tqdm import tqdm 

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

_CITATION = """
"""

_DESCRIPTION = """
"""

class LensingLogNormalDatasetConfig(tfds.core.BuilderConfig):

  def __init__(self, *, N, map_size, gal_per_arcmin2, sigma_e, model_type, proposal, score_type, **kwargs):
    v1 = tfds.core.Version("0.0.1")
    super(LensingLogNormalDatasetConfig, self).__init__(
        description=("Lensing simulations."),
        version=v1,
        **kwargs)
    self.N = N
    self.map_size = map_size
    self.gal_per_arcmin2 = gal_per_arcmin2
    self.sigma_e = sigma_e
    self.model_type = model_type
    self.proposal = proposal
    self.score_type = score_type



class LensingLogNormalDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.0.1': 'Initial release.',
  }
  BUILDER_CONFIGS = [LensingLogNormalDatasetConfig(name="toy_model_with_proposal", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=30, 
                                            sigma_e=0.2,
                                            model_type='lognormal', 
                                            proposal = True, 
                                            score_type = 'density'),
                    LensingLogNormalDatasetConfig(name="year_1_score_density", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=10, 
                                            sigma_e=0.26,
                                            model_type='lognormal', 
                                            proposal = True, 
                                            score_type = 'density'),
                     LensingLogNormalDatasetConfig(name="year_10_score_density", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=27, 
                                            sigma_e=0.26, 
                                            model_type='lognormal', 
                                            proposal = True, 
                                            score_type = 'density'),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'simulation': tfds.features.Tensor(shape=[self.builder_config.N, self.builder_config.N], dtype=tf.float32),
            'theta': tfds.features.Tensor(shape=[2], dtype=tf.float32),
            'score': tfds.features.Tensor(shape=[2], dtype=tf.float32),
        }),
        supervised_keys=None,  
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    return [
        tfds.core.SplitGenerator(name=tfds.Split.TRAIN, 
                                 gen_kwargs={'size': 500000}),

    ]

  def _generate_examples(self, size):
    """Yields examples."""

    SOURCE_FILE = Path(__file__)
    SOURCE_DIR = SOURCE_FILE.parent
    ROOT_DIR = SOURCE_DIR.parent.resolve()
    DATA_DIR = ROOT_DIR / "data"

    if self.builder_config.name == 'toy_model_with_proposal':
      FILE = "sample_power_spectrum_toy_model.npy"
    elif self.builder_config.name == "year_1_score_density":
      FILE = "sample_power_spectrum_year_1.npy"
    elif self.builder_config.name == "year_10_score_density":
      FILE = "sample_power_spectrum_year_10.npy"

  
    bs = 100
    if self.builder_config.proposal == True:
        thetas = np.load(DATA_DIR / FILE)
        thetas = thetas[:(len(thetas)//bs)*bs]
        if size > len(thetas):
          size = len(thetas)
        thetas = thetas.reshape([-1,bs,2])
    else: 
        thetas = np.array([None]).repeat(size // bs)

    model = partial(lensingLogNormal, 
                    self.builder_config.N, 
                    self.builder_config.map_size,
                    self.builder_config.gal_per_arcmin2,
                    self.builder_config.sigma_e,
                    self.builder_config.model_type)

    @jax.jit 
    def get_batch(key, thetas):
      (_, samples), scores = get_samples_and_scores(model = model, 
                                              key = key, 
                                              batch_size = bs, 
                                              score_type = self.builder_config.score_type,
                                              thetas = thetas) 

      return samples['y'], samples['theta'], scores

    master_key = jax.random.PRNGKey(2948570986789)


    for i in range(size // bs):    
      key, master_key = jax.random.split(master_key)
      simu, theta, score = get_batch(key, thetas[i])  

      for j in range(bs):                                  
        yield '{}-{}'.format(i,j), {
              'simulation': simu[j],
              'theta': theta[j],
              'score': score[j]
          }
