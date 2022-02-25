import os
from typing import List
from logging import Logger
import pprint

import tensorflow as tf
import tfx.v1 as tfx
from tfx.orchestration.metadata import Metadata
from tfx.types import standard_component_specs, artifact_utils
from tfx.utils import io_utils
from google.protobuf.json_format import MessageToDict

from utils.mlmd_helpers import get_latest_artifacts

pp = pprint.PrettyPrinter()


def get_records(dataset: tf.data.TFRecordDataset,
                num_records: int) -> List:
    """Extracts records from the given dataset.
    :param TFRecordDataset dataset:
        Dataset saved by ExampleGen.
    :param int num_records:
        Number of records to preview.
    :return:
        List.
    """
    # Initialize an empty list
    records = []

    # Use the `take()` method to specify how many records to get
    for tfrecord in dataset.take(num_records):

        # Get the numpy property of the tensor
        serialized_example = tfrecord.numpy()

        # Initialize a `tf.train.Example()` to read the serialized data
        example = tf.train.Example()

        # Read the example data (output is a protocol buffer message)
        example.ParseFromString(serialized_example)

        # convert the protocol bufffer message to a Python dictionary
        example_dict = (MessageToDict(example))

        # append to the records list
        records.append(example_dict)

    return records


def print_records(metadata_path: str,
                  pipeline_name: str,
                  num_records: int,
                  logger: Logger) -> None:
    """Print records from ExampleGen.

    :param str metadata_path:
        Path to metadata db.
    :param str pipeline_name:
        TFX pipeline name.
    :param str base_path:
        Base path.
    :param int num_records:
        Number of records to print.
    :param Logger logger:
        Logger.
    """
    # Metadata config
    metadata_connection_config = tfx.orchestration.metadata.\
        sqlite_metadata_connection_config(metadata_path)

    # Get the artifact object
    with Metadata(metadata_connection_config) as metadata_handler:
        try:
            example_gen_output = get_latest_artifacts(metadata_handler,
                                                      pipeline_name,
                                                      'CsvExampleGen')
            artifacts = example_gen_output[
                standard_component_specs.EXAMPLES_KEY]
        except AttributeError:
            print('ExampleGen not available')

    if artifacts:
        for artifact in artifacts:
            for split in artifact_utils.decode_split_names(
                    artifact.split_names):

                if split == 'eval':
                    continue

                # Get the URI of the output artifact representing
                # the training examples
                train_uri = io_utils.get_only_uri_in_dir(
                    os.path.abspath(
                        artifact_utils.get_split_uri([artifact], split)))

                # Create a `TFRecordDataset` to read the file
                dataset = tf.data.TFRecordDataset(train_uri,
                                                  compression_type="GZIP")

                # Get records from the dataset
                sample_records = get_records(dataset=dataset,
                                             num_records=num_records)

                # Log the output
                logger.info('Row(s) in training data:')
                logger.info(sample_records)
    else:
        logger.info('No records printed...')
