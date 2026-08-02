import tensorflow as tf

import numpy as np
import pandas as pd
import os
import glob
import json
import functools

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def organism_path(organism):
    return "/home/users/zyfeng/DataSet/Input/"+organism+"/"


def get_dataset(organism, subset, num_threads=8):
  metadata = get_metadata(organism)
  dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  return dataset


def get_metadata(organism):
  path = os.path.join(organism_path(organism), 'statistics.json')
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def tfrecord_files(organism, subset):
  return sorted(tf.io.gfile.glob(os.path.join(organism_path(organism), 'tfrecords', f'{subset}-*.tfr')), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile."""
  feature_map = {'sequence': tf.io.FixedLenFeature([], tf.string),'target': tf.io.FixedLenFeature([], tf.string),}
  example = tf.io.parse_example(serialized_example, feature_map);
  sequence = tf.io.decode_raw(example['sequence'], tf.bool);
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4));
  sequence = tf.cast(sequence, tf.float32)

  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,(metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)

  return {'sequence': sequence,'target': target}

def _reduced_shape(shape, axis):
  if axis is None:
    return tf.TensorShape([])
  return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
  """Contains shared code for PearsonR and R2."""

  def __init__(self, reduce_axis=None, name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation (say
        (0, 1). If not specified, it will compute the correlation across the
        whole tensor.
      name: Metric name.
    """
    super(CorrelationStats, self).__init__(name=name)
    self._reduce_axis = reduce_axis
    self._shape = None  # Specified in _initialize.

  def _initialize(self, input_shape):
    # Remaining dimensions after reducing over self._reduce_axis.
    self._shape = _reduced_shape(input_shape, self._reduce_axis)

    weight_kwargs = dict(shape=self._shape, initializer='zeros')
    self._count = self.add_weight(name='count', **weight_kwargs)
    self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
    self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
    self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                             **weight_kwargs)
    self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
    self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                             **weight_kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Update the metric state.

    Args:
      y_true: Multi-dimensional float tensor [batch, ...] containing the ground
        truth values.
      y_pred: float tensor with the same shape as y_true containing predicted
        values.
      sample_weight: 1D tensor aligned with y_true batch dimension specifying
        the weight of individual observations.
    """
    if self._shape is None:
      # Explicit initialization check.
      self._initialize(y_true.shape)
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    self._product_sum.assign_add(
        tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

    self._true_sum.assign_add(
        tf.reduce_sum(y_true, axis=self._reduce_axis))

    self._true_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

    self._pred_sum.assign_add(
        tf.reduce_sum(y_pred, axis=self._reduce_axis))

    self._pred_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

    self._count.assign_add(
        tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

  def result(self):
    raise NotImplementedError('Must be implemented in subclasses.')

  def reset_states(self):
    if self._shape is not None:
      tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                        for v in self.variables])


class PearsonR(CorrelationStats):
  """Pearson correlation coefficient.

  Computed as:
  ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
  """

  def __init__(self, reduce_axis=(0,), name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                   name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    pred_mean = self._pred_sum / self._count

    covariance = (self._product_sum
                  - true_mean * self._pred_sum
                  - pred_mean * self._true_sum
                  + self._count * true_mean * pred_mean)

    true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
    pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
    tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
    correlation = covariance / tp_var

    return correlation


class R2(CorrelationStats):
  """R-squared  (fraction of explained variance)."""

  def __init__(self, reduce_axis=None, name='R2'):
    """R-squared metric.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(R2, self).__init__(reduce_axis=reduce_axis,
                             name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    total = self._true_squared_sum - self._count * tf.math.square(true_mean)
    residuals = (self._pred_squared_sum - 2 * self._product_sum
                 + self._true_squared_sum)

    return tf.ones_like(residuals) - residuals / total


class MetricDict:
  def __init__(self, metrics):
    self._metrics = metrics

  def update_state(self, y_true, y_pred):
    for k, metric in self._metrics.items():
      metric.update_state(y_true, y_pred)

  def result(self):
    return {k: metric.result() for k, metric in self._metrics.items()}

f = open(os.path.join(SRC_DIR, 'HumanData.txt'))
a = f.readlines();f.close()
del a[0]
CAGE = [[],[]];ChIPH = [[],[]];ChIPT = [[],[]];ATAC = [[],[]]

for i in range(len(a)):
    a[i] = a[i].strip('\n').split('\t')
    if "CAGE" in a[i][8]:
        CAGE[0].append(a[i][3])
        CAGE[1].append(i)
    elif a[i][8].split(':')[0] == "DNASE" or a[i][8].split(':')[0] == "ATAC":
        ATAC[0].append(a[i][3])
        ATAC[1].append(i)
    elif  a[i][8].split(':')[0] == "CHIP":
        if "CHIP:H2" in a[i][8] or "CHIP:H3" in a[i][8] or "CHIP:H4" in a[i][8]:
            ChIPH[0].append(a[i][3])
            ChIPH[1].append(i)
        else:
            ChIPT[0].append(a[i][3])
            ChIPT[1].append(i)

CAGE1 = [];ChIPH1 = [];ChIPT1 = [];ATAC1 = []

f = open(os.path.join(SRC_DIR, 'Human_wigs_class.txt'))
wigs = f.readlines();f.close()
RNA2 = [];CAGE2 = [];ChIPH2 = [];ChIPT2 = [];ATAC2 = []
for i in range(len(wigs)):
    wigs[i] = wigs[i].strip('\n').split('\t')
    if wigs[i][2] == "RNA":
        RNA2.append(i)
    if wigs[i][2] == "CAGE" and wigs[i][1] in CAGE[0]:
        CAGE2.append(i)
        CAGE1.append(CAGE[1][CAGE[0].index(wigs[i][1])])
    if wigs[i][2] == "ChIP_Histone" and wigs[i][1] in ChIPH[0]:
        ChIPH2.append(i)
        ChIPH1.append(ChIPH[1][ChIPH[0].index(wigs[i][1])])
    if wigs[i][2] == "ChIP_TF" and wigs[i][1] in ChIPT[0]:
        ChIPT2.append(i)
        ChIPT1.append(ChIPT[1][ChIPT[0].index(wigs[i][1])])
    if wigs[i][2] == "ATAC/DNASE" and wigs[i][1] in ATAC[0]:
        ATAC2.append(i)
        ATAC1.append(ATAC[1][ATAC[0].index(wigs[i][1])])

All1 = CAGE1 + ChIPH1 + ChIPT1 + ATAC1
All2 = CAGE2 + ChIPH2 + ChIPT2 + ATAC2

def evaluate_model0(model, dataset, max_steps=None):
    @tf.function
    def predict(x):
        return model(x, is_training=False)["human"]
    
    metric_all0 = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_cage0 = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_atac0 = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_chiph0 = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_chipt0 = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    
    for i, batch in enumerate(dataset):
        if max_steps is not None and i > max_steps:
            break
        predicted_values = predict(batch['sequence'])
        metric_all0.update_state(tf.gather(batch['target'],All2,axis=2), tf.gather(predicted_values,All1,axis=2))
        metric_cage0.update_state(tf.gather(batch['target'],CAGE2,axis=2), tf.gather(predicted_values,CAGE1,axis=2))
        metric_atac0.update_state(tf.gather(batch['target'],ATAC2,axis=2), tf.gather(predicted_values,ATAC1,axis=2))
        metric_chiph0.update_state(tf.gather(batch['target'],ChIPH2,axis=2), tf.gather(predicted_values,ChIPH1,axis=2))
        metric_chiph0.update_state(tf.gather(batch['target'],ChIPT2,axis=2), tf.gather(predicted_values,ChIPT1,axis=2))
    
    pcc_all0 = [v.numpy().mean() for k, v in metric_all0.result().items()][0]
    pcc_cage0 = [v.numpy().mean() for k, v in metric_cage0.result().items()][0]
    pcc_atac0 = [v.numpy().mean() for k, v in metric_atac0.result().items()][0]
    pcc_chiph0 = [v.numpy().mean() for k, v in metric_chiph0.result().items()][0]
    pcc_chipt0 = [v.numpy().mean() for k, v in metric_chipt0.result().items()][0]
    
    return pcc_all0,pcc_cage0,pcc_atac0,pcc_chiph0,pcc_chipt0

def evaluate_model1(model, dataset, max_steps=None):
    @tf.function
    def predict(x):
        return model(x, is_training=False)["human1"]
    
    metric_all = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_cage = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_rna = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_atac = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_chiph = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    metric_chipt = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    
    for i, batch in enumerate(dataset):
        if max_steps is not None and i > max_steps:
            break
        predicted_values = predict(batch['sequence'])
        metric_all.update_state(batch['target'], predicted_values)
        metric_cage.update_state(tf.gather(batch['target'],CAGE2,axis=2), tf.gather(predicted_values,CAGE2,axis=2))
        metric_rna.update_state(tf.gather(batch['target'],RNA2,axis=2), tf.gather(predicted_values,RNA2,axis=2))
        metric_atac.update_state(tf.gather(batch['target'],ATAC2,axis=2), tf.gather(predicted_values,ATAC2,axis=2))
        metric_chiph.update_state(tf.gather(batch['target'],ChIPH2,axis=2), tf.gather(predicted_values,ChIPH2,axis=2))
        metric_chipt.update_state(tf.gather(batch['target'],ChIPT2,axis=2), tf.gather(predicted_values,ChIPT2,axis=2))
        
    pcc_all = [v.numpy().mean() for k, v in metric_all.result().items()][0]
    pcc_cage = [v.numpy().mean() for k, v in metric_cage.result().items()][0]
    pcc_rna = [v.numpy().mean() for k, v in metric_rna.result().items()][0]
    pcc_atac = [v.numpy().mean() for k, v in metric_atac.result().items()][0]
    pcc_chiph = [v.numpy().mean() for k, v in metric_chiph.result().items()][0]
    pcc_chipt = [v.numpy().mean() for k, v in metric_chipt.result().items()][0]
    
    return pcc_all,pcc_cage,pcc_rna,pcc_atac,pcc_chiph,pcc_chipt
