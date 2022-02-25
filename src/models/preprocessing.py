"""TFX taxi preprocessing.

This file defines a template for TFX Transform component and uses features
defined in features.py.
"""

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_probability as tfp

from models import features


def _fill_in_missing(x: tf.sparse.SparseTensor,
                     fill_value=None):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with `fill_value`, or '' or 0, and converts to a dense tensor.

    :param SparseTensor x:
        Of rank 2.  Its dense shape should have size at most 1 in the
        second dimension.
    :param fill_value:
        Specified value to impute missing values.

    :return:
        A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    if fill_value is not None:
        default_value = fill_value
    else:
        default_value = '' if x.dtype == tf.string else 0

    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    :param inputs:
      Map from feature keys to raw not-yet-transformed features.
    :return:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in features.DENSE_FLOAT_FEATURE_KEYS:
        # If sparse make it dense, impute missing values, and apply zscore.
        outputs[features.transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key],
                             fill_value=tf.cast(tft.mean(inputs[key]),
                                                inputs[key].dtype)))

    for key, num_buckets in features.BUCKET_FEATURE_DICT.items():
        outputs[features.transformed_name(key)] = tft.bucketize(
            _fill_in_missing(
                inputs[key],
                fill_value=tf.cast(tft.mean(inputs[key]), inputs[key].dtype)),
            num_buckets)

    for key, size in features.VOCAB_FEATURE_DICT.items():
        outputs[features.transformed_name(key)] = tft\
          .compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=size,
            num_oov_buckets=features.OOV_SIZE)

    worst_status = _fill_in_missing(
        inputs[features.WORST_STATUS], fill_value=1)
    outputs[features.transformed_name(features.WORST_STATUS)] = tf.cast(
        tf.greater(worst_status, 1), tf.int64)

    for key in features.BOOL_FEATURE_KEYS:
        val = _fill_in_missing(inputs[key], fill_value=0)
        outputs[features.transformed_name(key)] = tf.cast(
            tf.greater(val, 0), tf.int64)

    # for key, value in features.INT_FEATURE_DICT.items():
    #     outputs[features.transformed_name(key)] = _fill_in_missing(
    #         inputs[key], fill_value=1)

    # # NOTE: One-Hot encoding strategy
    # # Convert strings to indices and convert to one-hot vectors
    # for key, vocab_size in features.VOCAB_FEATURE_DICT.items():
    #     indices = tft.compute_and_apply_vocabulary(
    #         _fill_in_missing(inputs[key]), num_oov_buckets=features.OOV_SIZE)
    #     one_hot = tf.one_hot(indices, vocab_size + features.OOV_SIZE)
    #     outputs[features.transformed_name(key)] = tf.reshape(
    #         one_hot, [-1, vocab_size + features.OOV_SIZE])

    # # Bucketize this feature and convert to one-hot vectors
    # for key, num_buckets in _BUCKET_FEATURE_DICT.items():
    #     indices = tft.bucketize(inputs[key], num_buckets)
    #     one_hot = tf.one_hot(indices, num_buckets)
    #     outputs[key] = tf.reshape(one_hot, [-1, num_buckets])

    # Did the transaction default or not?
    outputs[features.transformed_name(features.LABEL_KEY)] = inputs[features.LABEL_KEY]  # noqa: E501

    return outputs
