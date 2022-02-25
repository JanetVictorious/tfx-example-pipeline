"""TFX taxi template pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""

import os
from typing import Any, Dict, List, Optional

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.proto import example_gen_pb2
from tfx.orchestration import metadata
from tfx.dsl.components.common.importer import Importer
from tfx.types.standard_artifacts import HyperParameters

from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.dsl.components.base import executor_spec
from utils.custom_executor import Executor

from ml_metadata.proto import metadata_store_pb2
from google.protobuf import text_format

from models import features

from utils.mlmd_helpers import get_latest_artifacts


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    # TODO: (Optional) Uncomment here to use BigQuery as a data source.
    # query: str,
    module_file: str,
    train_args: tfx.proto.TrainArgs,
    eval_args: tfx.proto.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: str,
    schema_path: Optional[str] = None,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[str]] = None,
    ai_platform_training_args: Optional[Dict[str, str]] = None,
    ai_platform_serving_args: Optional[Dict[str, Any]] = None,
    enable_tuning: bool = False,
    hparams_path: Optional[str] = None,
    eval_config_path: Optional[str] = None
) -> tfx.dsl.Pipeline:
    """Implements the chicago taxi pipeline with TFX."""

    components_list = []

    # +----------------+
    # |   ExampleGen   |
    # +----------------+

    # Specify 80/20 split for the train and eval set
    # For splitting techniques see: https://www.tensorflow.org/tfx/guide/examplegen#splitting_method  # noqa: E501
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
        ]))

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = tfx.components.CsvExampleGen(input_base=data_path,
                                               output_config=output)
    # NOTE: (Optional) Uncomment here to use FileBasedExampleGen, i.e. other
    # formats than CSV, or with customized executor.
    # example_gen = FileBasedExampleGen(
    #     input_base=data_path,
    #     output_config=output,
    #     # custom_executor_spec=executor_spec.ExecutorClassSpec(Executor)
    #     custom_executor_spec=executor_spec.BeamExecutorSpec(Executor))
    # NOTE: (Optional) Uncomment here to use BigQuery as a data source.
    # example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
    #     query=query)
    components_list.append(example_gen)

    # +-------------------+
    # |   StatisticsGen   |
    # +-------------------+

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])
    # NOTE: Uncomment here to add StatisticsGen to the pipeline.
    components_list.append(statistics_gen)

    # +--------------+
    # |   SchemGen   |
    # +--------------+

    if schema_path is None:
        # Generates schema based on statistics files.
        schema_gen = tfx.components.SchemaGen(
            statistics=statistics_gen.outputs['statistics'],
            exclude_splits=['eval'],
            infer_feature_shape=True)
        # NOTE: Uncomment here to add SchemaGen to the pipeline.
        components_list.append(schema_gen)
    else:
        # Import user provided schema into the pipeline.
        schema_gen = tfx.components.ImportSchemaGen(schema_file=schema_path)
        # NOTE: (Optional) Uncomment here to add ImportSchemaGen to the
        # pipeline.
        components_list.append(schema_gen)

    # +----------------------+
    # |   ExampleValidator   |
    # +----------------------+

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(  # noqa: F841
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    # NOTE: (Optional) Uncomment here to add ExampleValidator to the pipeline.
    components_list.append(example_validator)

    # +---------------+
    # |   Transform   |
    # +---------------+

    # Performs transformations and feature engineering in training and serving.
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file)
    # NOTE: Uncomment here to add Transform to the pipeline.
    components_list.append(transform)

    # +-----------+
    # |   Tuner   |
    # +-----------+

    # Tunes the hyperparameters for model training based on user-provided
    # Python function. Note that once the hyperparameters are tuned, you can
    # drop the Tuner component from pipeline and feed Trainer with tuned
    # hyperparameters.
    if enable_tuning and hparams_path is None:
        tuner = tfx.components.Tuner(
            module_file=module_file,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            schema=schema_gen.outputs['schema'],
            train_args=tfx.proto.TrainArgs(num_steps=500),
            eval_args=tfx.proto.EvalArgs(num_steps=100))
        hparams = tuner.outputs['best_hyperparameters']
        components_list.append(tuner)
    elif hparams_path is not None:
        # Check Tuner directory exists in output folder
        if not os.path.isfile(hparams_path):
            hparams = None
        else:
            # Import parameters
            hparams_importer = Importer(
                source_uri=hparams_path,
                artifact_type=HyperParameters
            ).with_id('import_hparams')
            hparams = hparams_importer.outputs['result']
            components_list.append(hparams_importer)
    else:
        hparams = None

    # +-------------+
    # |   Trainer   |
    # +-------------+

    # Uses user-provided Python function that implements a model.
    trainer_args = {
        'module_file': module_file,
        'examples': transform.outputs['transformed_examples'],
        'schema': schema_gen.outputs['schema'],
        'transform_graph': transform.outputs['transform_graph'],
        'train_args': train_args,
        'eval_args': eval_args,
        'hyperparameters': hparams,
    }
    if ai_platform_training_args is not None:
        trainer_args['custom_config'] = {
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                ai_platform_training_args,
        }
        trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
            **trainer_args)
    else:
        trainer = tfx.components.Trainer(
            **trainer_args
            # If Tuner is in the pipeline, Trainer can take Tuner's output
            # best_hyperparameters artifact as input and utilize it in the
            # user module code.
            #
            # If there isn't Tuner in the pipeline, either use Importer to
            # import a previous Tuner's output to feed to Trainer, or directly
            # use the tuned hyperparameters in user module code and set
            # hyperparameters to None here.
            )
    # NOTE: Uncomment here to add Trainer to the pipeline.
    # components_list.append(trainer)

    # +--------------+
    # |   Resolver   |
    # +--------------+

    # Get the latest blessed model for model validation.
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
                'latest_blessed_model_resolver')
    # NOTE: Uncomment here to add Resolver to the pipeline.
    # components_list.append(model_resolver)

    # +---------------+
    # |   Evaluator   |
    # +---------------+

    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    if eval_config_path:
        with open(eval_config_path, 'r') as eval_file:
            eval_str = eval_file.read()
        eval_config = text_format.Parse(eval_str, tfma.EvalConfig())
    else:
        eval_config = tfma.EvalConfig(
            model_specs=[
                tfma.ModelSpec(
                    signature_name='serving_default',
                    # label_key=features.LABEL_KEY,
                    # Use transformed label key if Transform is used.
                    label_key=features.transformed_name(features.LABEL_KEY),
                    preprocessing_function_names=['transform_features'])
            ],
            slicing_specs=[tfma.SlicingSpec(),
                           tfma.SlicingSpec(feature_keys=['has_paid']),
                           tfma.SlicingSpec(feature_keys=['merchant_group']),
                           ],
            metrics_specs=[
                tfma.MetricsSpec(metrics=[
                    tfma.MetricConfig(
                        class_name='BinaryAccuracy',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': eval_accuracy_threshold}),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={'value': -1e-10})))
                ])
            ])
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        # Change threshold will be ignored if there is no baseline (first run).
        eval_config=eval_config)
    # NOTE: Uncomment here to add Evaluator to the pipeline.
    # components_list.append(evaluator)

    # +--------------------+
    # |   InfraValidator   |
    # +--------------------+

    infra_validator = tfx.components.InfraValidator(
        model=trainer.outputs['model'],
        # This is the source for the data that will be used to build a request
        examples=example_gen.outputs['examples'],
        # Defines what type of model server and where to run it
        serving_spec=tfx.proto.ServingSpec(
            tensorflow_serving=tfx.proto.TensorFlowServing(tags=['latest']),
            local_docker=tfx.proto.LocalDockerConfig()),
        # Optional config to adjust criteria or workflow
        validation_spec=tfx.proto.ValidationSpec(
            # How much time to wait for model to load before automatically
            # making validation fail
            max_loading_time_seconds=60,
            # How many times to retry if infra validation fails
            num_tries=3),
        # Optional config to specify how to build sample requests
        request_spec=tfx.proto.RequestSpec(
            # tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(
            #     signature_names=['serving_default', 'transform_features']),
            tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(),
            num_examples=10
        )
    )
    # NOTE: Uncomment here to add InfraValidator to the pipeline.
    # components_list.append(infra_validator)

    # +------------+
    # |   Pusher   |
    # +------------+

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher_args = {
        'model':
            trainer.outputs['model'],
        'model_blessing':
            evaluator.outputs['blessing'],
        'infra_blessing':
            infra_validator.outputs['blessing'],
    }
    if ai_platform_serving_args is not None:
        pusher_args['custom_config'] = {
            tfx.extensions.google_cloud_ai_platform.experimental
            .PUSHER_SERVING_ARGS_KEY:
                ai_platform_serving_args
        }
        pusher = tfx.extensions.google_cloud_ai_platform.Pusher(**pusher_args)  # noqa: F841 E501
    else:
        pusher_args['push_destination'] = tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir))
        pusher = tfx.components.Pusher(**pusher_args)  # noqa: F841
    # NOTE: Uncomment here to add Pusher to the pipeline.
    # components_list.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components_list,
        # NOTE: Change this value to control caching of execution results.
        # Default value is `False`.
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )
