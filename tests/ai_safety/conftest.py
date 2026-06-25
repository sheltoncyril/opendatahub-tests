from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.service import Service
from pytest_testconfig import config as py_config

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import EMULATOR_NAMESPACE, Labels, Protocols, Timeout, TRUSTYAI_SERVICE_NAME
from utilities.infra import create_ns

VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_PORT: int = 8000
VLLM_EMULATOR_IMAGE = (
    "quay.io/trustyai_testing/vllm_emulator"
    "@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
)


@pytest.fixture(scope="session")
def emulator_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Session-scoped namespace for shared LLM emulators.

    All stateless LLM emulators (LLM-d sim, vLLM emulator) deploy here once
    per session. Test classes reference them via FQDN.
    """
    with create_ns(
        admin_client=admin_client,
        name=EMULATOR_NAMESPACE,
        teardown=True,
    ) as ns:
        yield ns


@pytest.fixture(scope="session")
def session_vllm_emulator_deployment(
    admin_client: DynamicClient, emulator_namespace: Namespace
) -> Generator[Deployment, Any, Any]:
    """Session-scoped vLLM emulator deployed to emulator_namespace."""
    label = {Labels.Openshift.APP: VLLM_EMULATOR}
    with Deployment(
        client=admin_client,
        namespace=emulator_namespace.name,
        name=VLLM_EMULATOR,
        label=label,
        replicas=1,
        selector={"matchLabels": label},
        template={
            "metadata": {"labels": label},
            "spec": {
                "securityContext": {"seccompProfile": {"type": "RuntimeDefault"}},
                "containers": [
                    {
                        "name": "vllm-emulator",
                        "image": VLLM_EMULATOR_IMAGE,
                        "ports": [{"containerPort": VLLM_EMULATOR_PORT, "protocol": "TCP"}],
                        "readinessProbe": {
                            "tcpSocket": {"port": VLLM_EMULATOR_PORT},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5,
                            "failureThreshold": 6,
                        },
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                    }
                ],
            },
        },
    ) as deployment:
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield deployment


@pytest.fixture(scope="session")
def session_vllm_emulator_service(
    admin_client: DynamicClient, emulator_namespace: Namespace, session_vllm_emulator_deployment: Deployment
) -> Generator[Service, Any, Any]:
    """Session-scoped Service fronting the vLLM emulator."""
    with Service(
        client=admin_client,
        namespace=emulator_namespace.name,
        name=f"{VLLM_EMULATOR}-service",
        ports=[
            {
                "name": f"{VLLM_EMULATOR}-endpoint",
                "port": VLLM_EMULATOR_PORT,
                "protocol": Protocols.TCP,
                "targetPort": VLLM_EMULATOR_PORT,
            }
        ],
        selector={Labels.Openshift.APP: VLLM_EMULATOR},
    ) as service:
        yield service


@pytest.fixture(scope="class")
def pvc_minio_namespace(
    admin_client: DynamicClient, minio_namespace: Namespace
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        client=admin_client,
        name="minio-pvc",
        namespace=minio_namespace.name,
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
        size="10Gi",
    ) as pvc:
        yield pvc


@pytest.fixture(scope="session")
def trustyai_operator_configmap(
    admin_client: DynamicClient,
) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        namespace=py_config["applications_namespace"],
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-config",
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def openshift_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """Create CA bundle file for HTTPS verification."""
    return create_ca_bundle_file(client=admin_client)
