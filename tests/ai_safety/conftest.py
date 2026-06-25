from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.service import Service
from pytest_testconfig import config as py_config

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import SHARED_MODELS_NAMESPACE, TRUSTYAI_SERVICE_NAME, Labels, Protocols, Timeout
from utilities.infra import create_ns

LOGGER = structlog.get_logger(name=__name__)

VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_PORT: int = 8000
VLLM_EMULATOR_IMAGE = (
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
)

WORKER_ANNOTATION_PREFIX = "ai-safety.opendatahub.io/worker-"


def _get_worker_id(request: pytest.FixtureRequest) -> str:
    return getattr(request.config, "workerinput", {}).get("workerid", "main")


def _register_worker(ns: Namespace, worker_id: str) -> None:
    ns.update(
        resource_dict={
            "metadata": {
                "name": ns.name,
                "annotations": {f"{WORKER_ANNOTATION_PREFIX}{worker_id}": "active"},
            }
        }
    )
    LOGGER.info(f"Registered worker {worker_id} on namespace {ns.name}")


def _deregister_worker(ns: Namespace, worker_id: str) -> bool:
    """Remove worker annotation. Returns True if this was the last worker."""
    ns.update(
        resource_dict={
            "metadata": {
                "name": ns.name,
                "annotations": {f"{WORKER_ANNOTATION_PREFIX}{worker_id}": None},
            }
        }
    )
    ns.reload()
    annotations = ns.instance.metadata.get("annotations", {}) or {}
    remaining = [k for k in annotations if k.startswith(WORKER_ANNOTATION_PREFIX)]
    LOGGER.info(f"Deregistered worker {worker_id}, {len(remaining)} workers remaining")
    return len(remaining) == 0


@pytest.fixture(scope="session")
def shared_models_namespace(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Session-scoped namespace for shared LLM model servers.

    Workers coordinate via annotations on the namespace. Each worker registers
    on setup and deregisters on teardown. The last worker to deregister deletes
    the namespace and all resources in it.
    """
    worker_id = _get_worker_id(request=request)
    ns = Namespace(client=admin_client, name=SHARED_MODELS_NAMESPACE)

    if ns.exists:
        LOGGER.info(f"Namespace {SHARED_MODELS_NAMESPACE} already exists, reusing")
        _register_worker(ns=ns, worker_id=worker_id)
        yield ns
        if _deregister_worker(ns=ns, worker_id=worker_id):
            LOGGER.info(f"Last worker, deleting namespace {SHARED_MODELS_NAMESPACE}")
            ns.delete(wait=True)
    else:
        with create_ns(admin_client=admin_client, name=SHARED_MODELS_NAMESPACE, teardown=False) as created_ns:
            _register_worker(ns=created_ns, worker_id=worker_id)
            yield created_ns
            if _deregister_worker(ns=created_ns, worker_id=worker_id):
                LOGGER.info(f"Last worker, deleting namespace {SHARED_MODELS_NAMESPACE}")
                created_ns.delete(wait=True)


@pytest.fixture(scope="session")
def session_vllm_emulator_deployment(
    admin_client: DynamicClient,
    shared_models_namespace: Namespace,
) -> Generator[Deployment, Any, Any]:
    """Session-scoped vLLM emulator. Create-if-not-exists, no per-resource teardown.

    Namespace cascade-delete (managed by shared_models_namespace) handles cleanup.
    """
    existing = Deployment(client=admin_client, name=VLLM_EMULATOR, namespace=shared_models_namespace.name)
    if existing.exists:
        LOGGER.info("vLLM emulator deployment already exists, reusing")
        existing.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield existing
        return

    label = {Labels.Openshift.APP: VLLM_EMULATOR}
    deployment = Deployment(
        client=admin_client,
        namespace=shared_models_namespace.name,
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
        teardown=False,
    )
    deployment.deploy()
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    yield deployment


@pytest.fixture(scope="session")
def session_vllm_emulator_service(
    admin_client: DynamicClient,
    shared_models_namespace: Namespace,
    session_vllm_emulator_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Session-scoped Service fronting the vLLM emulator. No per-resource teardown."""
    svc_name = f"{VLLM_EMULATOR}-service"
    existing = Service(client=admin_client, name=svc_name, namespace=shared_models_namespace.name)
    if existing.exists:
        LOGGER.info("vLLM emulator service already exists, reusing")
        yield existing
        return

    svc = Service(
        client=admin_client,
        namespace=shared_models_namespace.name,
        name=svc_name,
        ports=[
            {
                "name": f"{VLLM_EMULATOR}-endpoint",
                "port": VLLM_EMULATOR_PORT,
                "protocol": Protocols.TCP,
                "targetPort": VLLM_EMULATOR_PORT,
            }
        ],
        selector={Labels.Openshift.APP: VLLM_EMULATOR},
        teardown=False,
    )
    svc.deploy()
    yield svc


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
