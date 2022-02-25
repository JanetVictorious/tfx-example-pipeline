import os
from tfx.orchestration.metadata import Metadata
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.orchestration.experimental.interactive import visualizations,\
    standard_visualizations

from tfx.utils import io_utils
from tfx.types import artifact_utils
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html,\
    get_schema_dataframe,\
    get_anomalies_dataframe
from tensorflow_metadata.proto.v0 import anomalies_pb2
from IPython.display import HTML

standard_visualizations.register_standard_visualizations()


def get_latest_artifacts(metadata: Metadata,
                         pipeline_name: str,
                         component_id: str):
    """Output artifacts of the latest run of the component."""
    context = metadata.store.get_context_by_type_and_name(
        'node', f'{pipeline_name}.{component_id}')
    executions = metadata.store.get_executions_by_context(context.id)
    latest_execution = max(executions,
                           key=lambda e: e.last_update_time_since_epoch)
    return execution_lib.get_artifacts_dict(metadata,
                                            latest_execution.id,
                                            [metadata_store_pb2.Event.OUTPUT])


def visualize_artifacts_nb(artifacts):
    """Visualizes artifacts using standard visualization modules."""
    for artifact in artifacts:
        visualization = visualizations.get_registry().get_visualization(
            artifact.type_name)
        if visualization:
            visualization.display(artifact)


def visualize_artifacts(artifacts,
                        output_dir: str) -> None:
    """Visualizes artifacts using standard visualization modules."""
    # Create output directory if not exists
    output_path = os.path.join(output_dir)
    os.makedirs(output_path, exist_ok=True)

    for artifact in artifacts:
        if artifact.type_name == 'ExampleStatistics':
            d = {}
            for split in artifact_utils.decode_split_names(
                    artifact.split_names):
                stats_path = io_utils.get_only_uri_in_dir(
                    os.path.abspath(
                        artifact_utils.get_split_uri([artifact], split)))

                if artifact_utils.is_artifact_version_older_than(
                        artifact,
                        artifact_utils._ARTIFACT_VERSION_FOR_STATS_UPDATE):
                    stats = tfdv.load_statistics(stats_path)
                else:
                    stats = tfdv.load_stats_binary(stats_path)

                html = HTML(get_statistics_html(stats)).data

                # Export as html file
                with open(os.path.join(
                        output_path, f'{split}_statistics.html'), 'w') as file:
                    file.write(html)

                # Save stats to dict
                d[f'{split}'] = stats

            # Generate comparison if more than 1 split
            if len(d.keys()) == 2:
                stats1 = d.popitem()
                stats2 = d.popitem()
                html = HTML(get_statistics_html(lhs_statistics=stats1[1],
                                                rhs_statistics=stats2[1],
                                                lhs_name=stats1[0],
                                                rhs_name=stats2[0])).data

                # Export html file
                with open(os.path.join(
                        output_path, f'{stats1[0]}_vs_{stats2[0]}_statistics.html'), 'w') as file:
                    file.write(html)

        elif artifact.type_name == 'Schema':
            schema_path = os.path.abspath(artifact.uri, 'schema.pbtxt')
            schema = tfdv.load_schema_text(schema_path)
            features_df, domains_df = get_schema_dataframe(schema)
            html1 = features_df.to_html()
            html2 = domains_df.to_html()

            # Export as html file
            with open(os.path.join(output_path, 'features.html'), 'w') as file:
                file.write(html1)
            with open(os.path.join(output_path, 'domains.html'), 'w') as file:
                file.write(html2)
        elif artifact.type_name == 'ExampleAnomalies':
            for split in artifact_utils.decode_split_names(
                    artifact.split_names):
                anomalies_path = io_utils.get_only_uri_in_dir(
                    artifact_utils.get_split_uri([artifact], split))

                if artifact_utils.is_artifact_version_older_than(
                        artifact,
                        artifact_utils._ARTIFACT_VERSION_FOR_ANOMALIES_UPDATE):
                    anomalies = tfdv.load_anomalies_text(anomalies_path)
                else:
                    anomalies = anomalies_pb2.Anomalies()
                    anomalies_bytes = io_utils.read_bytes_file(anomalies_path)
                    anomalies.ParseFromString(anomalies_bytes)
                anomalies_df = get_anomalies_dataframe(anomalies)
                html = anomalies_df.to_html()

                # Export as html file
                with open(os.path.join(
                        output_path, f'{split}_anomalies.html'), 'w') as file:
                    file.write(html)
