
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


from tensorflow_datasets.core.utils import gcs_utils

import jax
from functools import partial
from pathlib import Path
from sbids.tasks.lensinglognormal import lensingLogNormal
from sbids.tasks import get_samples_and_scores

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False


class LensingLogNormalDatasetConfig(tfds.core.BuilderConfig):

  def __init__(self, *, N, map_size, gal_per_arcmin2, sigma_e, model_type, proposal, **kwargs):
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



class LensingLogNormalDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.0.1': 'Initial release.',
  }
  BUILDER_CONFIGS = [LensingLogNormalDatasetConfig(name="year_1", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=10, 
                                            sigma_e=0.26,
                                            model_type='lognormal', 
                                            proposal = True),
                     LensingLogNormalDatasetConfig(name="year_10", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=27, 
                                            sigma_e=0.26, 
                                            model_type='lognormal', 
                                            proposal = True),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'simulation': tfds.features.Image(shape=(self.builder_config.N, self.builder_config.N, 1)),
            'theta': tfds.features.Tensor(shape=[2], dtype=tf.float32),
            'score': tfds.features.Tensor(shape=[2], dtype=tf.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('image', 'label'),  # Set to `None` to disable rien compris
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    # return {
    #     'train': self._generate_examples(path / 'train_imgs'),
    # }
    return [
        tfds.core.SplitGenerator(name=tfds.Split.TRAIN, 
                                 gen_kwargs={'size': 15000}),
        # tfds.core.SplitGenerator(name=tfds.Split.TEST, 
        # ),
    ]

  def _generate_examples(self, size):
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    
    _BASEDIR = Path(__file__).parent.resolve()
    PARAM_FILE = "sample_full_field.npy"

    if self.builder_config.proposal == True:
        thetas = np.load(_BASEDIR / PARAM_FILE)
    else: 
        thetas = None

    model = partial(lensingLogNormal, 
                    self.builder_config.N, 
                    self.builder_config.map_size,
                    self.builder_config.gal_per_arcmin2,
                    self.builder_config.sigma_e,
                    self.builder_config.model_type)

    master_key = jax.random.PRNGKey(2948570986789)
    for i in range(size):    
      key, master_key = jax.random.split(master_key)
      (_, samples), scores = get_samples_and_scores(model, 
                                              key, 
                                              batch_size=1, 
                                              thetas = thetas)                                     
      yield '{}'.format(i), {
            'simulation': samples['y'][0],
            'theta': samples['theta'][0],
            'score': scores[0]
        }
