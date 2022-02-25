"""Define LocalDagRunner to run the pipeline locally."""

import argparse
import os
import absl
from absl import logging

from tfx import v1 as tfx
from pipeline import configs
from pipeline import pipeline


def _parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group('required arguments')

    required.add_argument("--output_dir",
                          help="Metadata output directory.",
                          dest='output_dir',
                          default='./outputs')

    required.add_argument("--data_dir",
                          help="Source data directory.",
                          dest='data_dir',
                          default='./data/klarna/small_train')

    required.add_argument("--log_dir",
                          help="Logs directory.",
                          dest='log_dir',
                          default='./logs')

    required.add_argument("--schema_path",
                          help="Path to custom schema.pbtxt.",
                          dest='schema_path',
                          default='./schema/schema.pbtxt')

    required.add_argument("--hparams_path",
                          help="Path to best_hyperparameters.txt.",
                          dest='hparams_path',
                          default='./config/best_hyperparameters.txt')

    required.add_argument("--eval_config_path",
                          help="Path to custom eval_config.pbtxt.",
                          dest='eval_config_path',
                          default='./config/eval_config.pbtxt')

    # Drop unknown arguments
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    # Fetch function args
    PARSER = _parse_args()

    # Set logging
    logging.use_absl_handler()
    logging.get_absl_handler().use_absl_log_file('absl_logging', PARSER.log_dir)  # noqa: E501
    absl.flags.FLAGS.mark_as_parsed()
    logging.set_verbosity(logging.INFO)

    # TFX pipeline produces many output files and metadata. All output data
    # will be stored under this OUTPUT_DIR.
    # NOTE: It is recommended to have a separated OUTPUT_DIR which is
    #       *outside* of the source code structure. Please change OUTPUT_DIR
    #       to other location where we can store outputs of the pipeline.
    OUTPUT_DIR = os.path.abspath(PARSER.output_dir)

    # TFX produces two types of outputs, files and metadata.
    # - Files will be created under PIPELINE_ROOT directory.
    # - Metadata will be written to SQLite database in METADATA_PATH.
    PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                                 configs.PIPELINE_NAME)
    METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata',
                                 configs.PIPELINE_NAME, 'metadata.db')

    # The last component of the pipeline, "Pusher" will produce serving model
    # under SERVING_MODEL_DIR.
    SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')

    # Specifies data file directory. DATA_DIR should be a directory
    # containing CSV files for CsvExampleGen in this example. By default, data
    # files are in the `data` directory.
    DATA_DIR = PARSER.data_dir

    # Specifices schema file directory. SCHEMA_PATH should point to a .pbtxt
    # file containing infromation about the features in the dataset file.
    SCHEMA_PATH = PARSER.schema_path

    # Specifies which MODULE_FILE (i.e. model.py file) to use when running
    # preprocessing, HPO tuning, and training.
    MODULE_FILE = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), configs.MODULE_FILE)

    # Specifies a path to best_hyperparameters.txt file to be imported in case
    # HPO tuning has already been done and you wish to use already tuned
    # hparams.
    HPARAMS_PATH = PARSER.hparams_path

    # Specifies path to custom eval_config.pbtxt file.
    EVAL_CONFIG_PATH = PARSER.eval_config_path

    # Define local pipeline and run
    tfx.orchestration.LocalDagRunner().run(
        pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=DATA_DIR,
            # NOTE: (Optional) Uncomment here to use BigQueryExampleGen.
            # query=configs.BIG_QUERY_QUERY,
            # NOTE: (Optional) Set the path of the customized schema.
            # schema_path=SCHEMA_PATH,
            module_file=MODULE_FILE,
            train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
            eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
            eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
            serving_model_dir=SERVING_MODEL_DIR,
            # NOTE: (Optional) Uncomment here to perform HPO tuning or load
            # already tuned hparams.
            # enable_tuning=True,
            # hparams_path=HPARAMS_PATH,
            # NOTE: (Optional) Uncomment here to use provide GCP related
            # config for BigQuery with Beam DirectRunner.
            # beam_pipeline_args=configs.
            # BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
            # NOTE: (Optional) Uncomment here to use custom specific
            # eval_config.
            eval_config_path=EVAL_CONFIG_PATH,
            metadata_connection_config=tfx.orchestration.metadata
            .sqlite_metadata_connection_config(METADATA_PATH)))
