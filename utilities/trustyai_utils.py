from typing import Generator, Any

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.resource import ResourceEditor


def patch_dsc_trustyai_lmeval_config(
    dsc: DataScienceCluster,
    trustyai_operator_deployment: Deployment,
    permit_code_execution: bool = False,
    permit_online: bool = False,
) -> Generator[DataScienceCluster, Any, Any]:
    """
    Patch DataScienceCluster object with default deployment mode and wait for it to be set in configmap.

    Args:
        dsc (DataScienceCluster): DataScienceCluster object
        trustyai_operator_deployment: Deployment The trustyai-operator deployment
        permit_code_execution (bool, optional): Allow code execution mode. Defaults to False.
        permit_online (bool, optional): Allow online mode. Defaults to False.
    Yields:
        DataScienceCluster: DataScienceCluster object

    """
    with ResourceEditor(
        patches={
            dsc: {
                "spec": {
                    "components": {
                        "trustyai": {
                            "eval": {
                                "lmeval": {
                                    "permitCodeExecution": "allow" if permit_code_execution else "deny",
                                    "permitOnline": "allow" if permit_online else "deny",
                                }
                            }
                        }
                    }
                }
            }
        }
    ):
        num_replicas: int = trustyai_operator_deployment.replicas
        trustyai_operator_deployment.scale_replicas(replica_count=0)
        trustyai_operator_deployment.scale_replicas(replica_count=num_replicas)
        trustyai_operator_deployment.wait_for_replicas()
        yield dsc
