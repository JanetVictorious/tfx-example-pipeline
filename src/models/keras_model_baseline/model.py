"""TFX template model.

A sequential keras model which uses features defined in features.py and network
parameters defined in constants.py.
"""

import os
from absl import logging
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

from models import features, preprocessing
from models.keras_model_baseline import constants

# TFX Transform will call this function.
preprocessing_fn = preprocessing.preprocessing_fn

# Callback for the search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


def _get_hyperparameters() -> kt.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hp = kt.HyperParameters()

    # Defines search space.
    hp.Choice(name='learning_rate', values=[1e-3, 1e-4], default=1e-3)
    hp.Int(name='units1', min_value=16, max_value=48, step=16, default=48)
    hp.Choice(name='drop_out1', values=[0.5, 0.6], default=0.5)
    # hp.Fixed(name='units_1', value=16)
    # hp.Fixed(name='units_2', value=8)
    return hp


def _get_tf_examples_serving_signature(
        model, tf_transform_output: tft.TFTransformOutput):
    """Returns a serving signature that accepts `tensorflow.Example`."""

    # We need to track the layers in the model in order to save it.
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()

        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(features.LABEL_KEY)

        raw_features = tf.io.parse_example(serialized_tf_example,
                                           raw_feature_spec)

        # Transform raw features
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""

    # We need to track the layers in the model in order to save it.
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def _input_fn(file_pattern, data_accessor,
              tf_transform_output, batch_size=200):
    """Generates features and label for tuning/training.

    :param file_pattern:
        List of paths or patterns of input tfrecord files.
    :param data_accessor:
        DataAccessor for converting input to RecordBatch.
    :param tf_transform_output:
        A TFTransformOutput.
    :param batch_size:
        representing the number of consecutive elements of returned
        dataset to combine in a single batch
    :return:
        A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=features.transformed_name(features.LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema).repeat()


def _make_model(hparams: kt.HyperParameters, output_bias=None):
    """Creates a simple neural network with a densly connected hidden layer,
    a dropout layer to reduce overfitting, and an output sigmoid layer that
    returns the probability

    :param float learning_rate:
        Learning rate of the Adam optimizer.
    :return:
        A Sequential Keras model.
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Keras needs the feature definitions at compile time

    # Define input layers for numeric keys
    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)  # noqa: E501
        for colname in features.transformed_names(
            features.DENSE_FLOAT_FEATURE_KEYS)
    }

    # Define input layers for bucket keys
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in features.transformed_names(features.BUCKET_FEATURE_DICT.keys())  # noqa: E501
    })

    # Define input layers for vocab keys
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in features.transformed_names(features.VOCAB_FEATURE_DICT.keys())  # noqa: E501
    })

    # Define input layer for worst status
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in features.transformed_names([features.WORST_STATUS])
    })

    # Define input layers for boolean keys
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in features.transformed_names(features.BOOL_FEATURE_KEYS)
    })

    # Create model
    inputs = [tf.keras.Input(shape=(1,), name=f) for f in input_layers.keys()]  # noqa: E501
    inputs_cc = tf.keras.layers.concatenate(inputs)
    h1 = tf.keras.layers.Dense(hparams.get('units1'), activation='relu')(inputs_cc)  # noqa: E501
    do1 = tf.keras.layers.Dropout(hparams.get('drop_out1'))(h1)
    output = tf.keras.layers.Dense(
        1, activation='sigmoid', bias_initializer=output_bias)(do1)
    model = tf.keras.Model(inputs, output)

    # Define training parameters
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hparams.get('learning_rate')),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.AUC(name='prc', curve='PR')])

    # Print model summary
    model.summary(print_fn=logging.info)

    return model


# TFX Tuner will call this function
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
    """Build the tuner using the KerasTuner API.

    :param fn_args:
        Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    :return:
        A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting
                    the model , e.g., the training and validation dataset.
                    Required args depend on the above tuner's implementation.
    """
    # Define tuner search strategy
    tuner = kt.Hyperband(_make_model,
                         hyperparameters=_get_hyperparameters(),
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory=fn_args.working_dir,
                         project_name='kt_hyperband')

    # Load transform output
    tf_transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_graph, constants.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                             tf_transform_graph, constants.EVAL_BATCH_SIZE)

    return tfx.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'callbacks': [stop_early],
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })


# TFX Trainer will call this function
def run_fn(fn_args: tfx.components.FnArgs) -> None:
    """Train the model based on given args.

    :param fn_args:
        Holds args used to train the model as name/value pairs.
        Refer here for the complete attributes:
        https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/fn_args_utils.py
    """

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Create batches of data
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_output, constants.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                             tf_transform_output, constants.EVAL_BATCH_SIZE)

    # Load best hyperparameters
    # hparams = fn_args.hyperparameters.get('values')
    if fn_args.hyperparameters:
        hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        # This is a shown case when hyperparameters is decided and Tuner is
        # removed from the pipeline. User can also inline the hyperparameters
        # directly in _build_keras_model.
        hparams = _get_hyperparameters()
    logging.info('HyperParameters for training: %s' % hparams.get_config())

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _make_model(hparams=hparams,
                            output_bias=constants.INITIAL_BIAS)

    # Callback for TensorBoard, write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    # Load weights
    # model.load_weights(constants.INITIAL_WEIGHTS)

    # NOTE: Uncomment this to set bias to zero and exclude careful
    # initialization
    # model.layers[-1].bias.assign([0.0])

    model.fit(train_dataset,
              steps_per_epoch=fn_args.train_steps,
              validation_data=eval_dataset,
              validation_steps=fn_args.eval_steps,
              callbacks=[tensorboard_callback],
              epochs=constants.NUM_EPOCHS,
              class_weight=constants.CLASS_WEIGHT)

    signatures = {
        'serving_default':
            _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features':
            _get_transform_features_signature(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf',
               signatures=signatures)
