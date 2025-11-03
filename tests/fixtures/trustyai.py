import json
import subprocess

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment

from typing import Generator

from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import py_config

from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.infra import get_data_science_cluster

DATASET_FILENAME = "example-dk-bench-input-bmo.jsonl"
DATASET_PATH_IN_DRIVER = f"/opt/app-root/src/hf_home/{DATASET_FILENAME}"




@pytest.fixture(scope="class")
def trustyai_operator_deployment(admin_client: DynamicClient) -> Deployment:
    return Deployment(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def patched_dsc_lmeval_allow_all(
    admin_client, trustyai_operator_deployment: Deployment
) -> Generator[DataScienceCluster, None, None]:
    """Enable LMEval PermitOnline and PermitCodeExecution flags in the Datascience cluster."""
    dsc = get_data_science_cluster(client=admin_client)
    with ResourceEditor(
        patches={
            dsc: {
                "spec": {
                    "components": {
                        "trustyai": {
                            "eval": {
                                "lmeval": {
                                    "permitCodeExecution": "allow",
                                    "permitOnline": "allow",
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


@pytest.fixture(scope="class")
def dataset_pvc(admin_client, model_namespace) -> Generator[PersistentVolumeClaim, None, None]:
    """
    Creates a PVC for holding dataset files.
    """
    pvc_kwargs = {
        "name": "my-pvc",
        "namespace": model_namespace.name,
        "client": admin_client,
        "size": "1Gi",
        "accessmodes": "ReadWriteOnce",
        "label": {"app.kubernetes.io/name": "dataset-storage"},
    }

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def dataset_upload(dataset_pvc, admin_client, model_namespace) -> str:
    """
    Uses a temporary BusyBox pod to copy the dataset JSONL into the PVC.
    Returns the full path inside the PVC (driver path).
    """
    remote_dir = "/opt/app-root/src/hf_home"
    remote_path = f"{remote_dir}/{DATASET_FILENAME}"

    # Temporary pod to write dataset into PVC
    with Pod(
        client=admin_client,
        name="dataset-storage-uploader",
        namespace=model_namespace.name,
        containers=[{
            "name": "uploader",
            "image": "quay.io/prometheus/busybox:latest",
            "command": ["/bin/sh", "-c", "sleep 300"],  # short-lived
            "volumeMounts": [{"name": "dataset-storage", "mountPath": remote_dir}],
            "securityContext": {"runAsUser": 0},  # write permissions
        }],
        volumes=[{"name": "dataset-storage", "persistentVolumeClaim": {"claimName": dataset_pvc.name}}],
        restart_policy="Never",
    ) as pod:
        pod.wait_for_status(status=Pod.Status.RUNNING)

        # Dataset content
        dataset_lines = [
            {
                "user_input": "what is the meaning of verifying the identity of a person or an entity",
                "reference": "It means to use methods to ensure that the information in an identification document or from other informational sources matches the information that the person or entity provided.",
                "response": "Verifying identity is the process of obtaining, recording, and maintaining information to confirm a person or entity's identity.",
            },
            {
                "user_input": "Why is it important to verify identity?",
                "reference": "Verifying identity is a foundational element of Canada's anti-money laundering and anti-terrorist financing regime.",
                "response": "Verifying identity is a critical step in maintaining security and preventing fraud.",
            },
        ]

        jsonl_content = "\n".join(json.dumps(line) for line in dataset_lines)

        # Ensure directory exists inside the PVC mount
        subprocess.run(
            ["oc", "exec", "-n", model_namespace.name, pod.name, "--", "mkdir", "-p", remote_dir],
            check=True,
        )

        # Write JSONL dataset
        subprocess.run(
            ["oc", "exec", "-i", "-n", model_namespace.name, pod.name, "--", "sh", "-c", f"cat > {remote_path}"],
            input=jsonl_content.encode("utf-8"),
            check=True,
        )

        # Pod can terminate; data persists in PVC
        pod.delete(wait=True)

    return remote_path
