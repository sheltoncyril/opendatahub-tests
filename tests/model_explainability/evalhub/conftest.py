from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_explainability.evalhub.constants import (
    EVALHUB_JOB_SA_PREFIX,
    EVALHUB_JOB_SA_SUFFIX,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_TENANT_LABEL_VALUE,
)
from tests.model_explainability.evalhub.utils import wait_for_service_account
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import KServeDeploymentType, LLMdInferenceSimConfig, Timeout
from utilities.inference_utils import create_isvc
from utilities.infra import create_inference_token, create_ns
from utilities.resources.evalhub import EvalHub
from utilities.resources.mlflow import MLflow

LOGGER = structlog.get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Shared EvalHub fixtures (used by health tests and garak tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def evalhub_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub custom resource and wait for it to be ready."""
    with EvalHub(
        client=admin_client,
        name="evalhub",
        namespace=model_namespace.name,
        database={"type": "sqlite"},
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub deployment to become available."""
    deployment = Deployment(
        client=admin_client,
        name=evalhub_cr.name,
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="class")
def evalhub_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_deployment: Deployment,
) -> Route:
    """Get the Route created by the operator for the EvalHub service."""
    return Route(
        client=admin_client,
        name=evalhub_deployment.name,
        namespace=model_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def evalhub_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """Create a CA bundle file for verifying the EvalHub route TLS certificate."""
    return create_ca_bundle_file(client=admin_client)


# ---------------------------------------------------------------------------
# MLflow fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def mlflow_instance(
    admin_client: DynamicClient,
) -> Generator[MLflow, Any, Any]:
    """Deploy an MLflow instance in the applications namespace for EvalHub experiment tracking."""
    with MLflow(
        client=admin_client,
        name="mlflow",
        namespace=py_config["applications_namespace"],
        storage={
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": "10Gi"}},
        },
        backend_store_uri="sqlite:////mlflow/mlflow.db",
        artifacts_destination="file:///mlflow/artifacts",
        serve_artifacts=True,
        image={"imagePullPolicy": "Always"},
        wait_for_resource=True,
    ) as mlflow:
        mlflow_deployment = Deployment(
            client=admin_client,
            name="mlflow",
            namespace=py_config["applications_namespace"],
        )
        mlflow_deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield mlflow


# ---------------------------------------------------------------------------
# Garak-specific EvalHub fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def garak_evalhub_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlflow_instance: MLflow,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub CR configured with the garak-kfp provider."""
    mlflow_uri = f"https://mlflow.{py_config['applications_namespace']}.svc.cluster.local:8443"
    with EvalHub(
        client=admin_client,
        name="evalhub",
        namespace=model_namespace.name,
        providers=["garak-kfp"],
        database={"type": "sqlite"},
        env=[{"name": "MLFLOW_TRACKING_URI", "value": mlflow_uri}],
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def garak_evalhub_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    garak_evalhub_cr: EvalHub,
) -> Deployment:
    """Wait for the garak-configured EvalHub deployment to become available."""
    deployment = Deployment(
        client=admin_client,
        name=garak_evalhub_cr.name,
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="class")
def garak_evalhub_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    garak_evalhub_deployment: Deployment,
) -> Route:
    """Get the Route for the garak-configured EvalHub service."""
    return Route(
        client=admin_client,
        name=garak_evalhub_deployment.name,
        namespace=model_namespace.name,
        ensure_exists=True,
    )


# ---------------------------------------------------------------------------
# Tenant namespace and RBAC fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def tenant_namespace(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    garak_evalhub_deployment: Deployment,
) -> Generator[Namespace | Project, Any, Any]:
    """Create a tenant namespace labeled for EvalHub multi-tenancy."""
    with create_ns(
        admin_client=admin_client,
        name=f"{model_namespace.name}-tenant",
        labels={EVALHUB_TENANT_LABEL_KEY: EVALHUB_TENANT_LABEL_VALUE},
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def evalhub_job_service_account(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    tenant_namespace: Namespace | Project,
) -> ServiceAccount:
    """Wait for the auto-created EvalHub job ServiceAccount in the tenant namespace."""
    sa_name = f"{EVALHUB_JOB_SA_PREFIX}{model_namespace.name}{EVALHUB_JOB_SA_SUFFIX}"
    return wait_for_service_account(
        admin_client=admin_client,
        namespace=tenant_namespace.name,
        sa_name=sa_name,
    )


@pytest.fixture(scope="class")
def tenant_user_sa(
    admin_client: DynamicClient,
    tenant_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """Create a ServiceAccount for tenant API access."""
    with ServiceAccount(
        client=admin_client,
        name="evalhub-user",
        namespace=tenant_namespace.name,
        wait_for_resource=True,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def tenant_user_role(
    admin_client: DynamicClient,
    tenant_namespace: Namespace,
    tenant_user_sa: ServiceAccount,
) -> Generator[Role, Any, Any]:
    """Create a Role granting EvalHub evaluation permissions in the tenant namespace."""
    with Role(
        client=admin_client,
        name="evalhub-evaluator",
        namespace=tenant_namespace.name,
        rules=[
            {
                "apiGroups": ["trustyai.opendatahub.io"],
                "resources": ["evaluations", "collections", "providers"],
                "verbs": ["get", "list", "create", "update", "delete"],
            },
            {
                "apiGroups": ["mlflow.kubeflow.org"],
                "resources": ["experiments"],
                "verbs": ["create", "get"],
            },
        ],
        wait_for_resource=True,
    ) as role:
        yield role


@pytest.fixture(scope="class")
def tenant_user_role_binding(
    admin_client: DynamicClient,
    tenant_namespace: Namespace,
    tenant_user_role: Role,
    tenant_user_sa: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    """Bind the evaluator role to the tenant user ServiceAccount."""
    with RoleBinding(
        client=admin_client,
        name="evalhub-evaluator-binding",
        namespace=tenant_namespace.name,
        role_ref_kind=tenant_user_role.kind,
        role_ref_name=tenant_user_role.name,
        subjects_kind="ServiceAccount",
        subjects_name=tenant_user_sa.name,
        subjects_namespace=tenant_namespace.name,
        wait_for_resource=True,
    ) as binding:
        yield binding


@pytest.fixture(scope="class")
def tenant_user_token(
    tenant_user_sa: ServiceAccount,
    tenant_user_role_binding: RoleBinding,
) -> str:
    """Create an API token for the tenant user ServiceAccount."""
    return create_inference_token(model_service_account=tenant_user_sa)


# ---------------------------------------------------------------------------
# DSPA and pipeline access fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def tenant_dspa(
    admin_client: DynamicClient,
    tenant_namespace: Namespace,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """Deploy a DataSciencePipelinesApplication with built-in MinIO in the tenant namespace."""
    with DataSciencePipelinesApplication(
        client=admin_client,
        name="dspa",
        namespace=tenant_namespace.name,
        dsp_version="v2",
        object_storage={
            "disableHealthCheck": False,
            "enableExternalRoute": True,
            "minio": {
                "deploy": True,
                "image": "quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance",
            },
        },
    ) as dspa:
        dspa_deployment = Deployment(
            client=admin_client,
            name="ds-pipeline-dspa",
            namespace=tenant_namespace.name,
        )
        dspa_deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)

        sw_deployment = Deployment(
            client=admin_client,
            name="ds-pipeline-scheduledworkflow-dspa",
            namespace=tenant_namespace.name,
        )
        sw_deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_2MIN)
        yield dspa


@pytest.fixture(scope="class")
def dspa_secret_patch(
    admin_client: DynamicClient,
    tenant_namespace: Namespace,
    tenant_dspa: DataSciencePipelinesApplication,
) -> Secret:
    """Patch the auto-created DSPA S3 secret with additional fields required by garak-kfp."""
    import base64

    secret = Secret(
        client=admin_client,
        name="ds-pipeline-s3-dspa",
        namespace=tenant_namespace.name,
    )
    assert secret.exists, f"Secret 'ds-pipeline-s3-dspa' not found in {tenant_namespace.name}"

    access_key = secret.instance.data.get("accesskey", "")
    secret_key = secret.instance.data.get("secretkey", "")

    access_key_decoded = base64.b64decode(access_key).decode() if access_key else ""
    secret_key_decoded = base64.b64decode(secret_key).decode() if secret_key else ""

    endpoint = f"http://minio-dspa.{tenant_namespace.name}.svc.cluster.local:9000"

    secret_data = {
        "AWS_ACCESS_KEY_ID": access_key_decoded,
        "AWS_SECRET_ACCESS_KEY": secret_key_decoded,
        "AWS_S3_ENDPOINT": endpoint,
        "AWS_S3_BUCKET": "mlpipeline",
        "AWS_DEFAULT_REGION": "us-east-1",
    }

    secret.update(resource_dict={
        "metadata": {"name": secret.name, "namespace": tenant_namespace.name},
        "stringData": secret_data,
    })
    LOGGER.info(f"Patched DSPA S3 secret with additional fields in {tenant_namespace.name}")
    return secret


@pytest.fixture(scope="class")
def dsp_access_for_job_sa(
    admin_client: DynamicClient,
    tenant_namespace: Namespace,
    evalhub_job_service_account: ServiceAccount,
    tenant_dspa: DataSciencePipelinesApplication,
) -> Generator[tuple[Role, RoleBinding, RoleBinding], Any, Any]:
    """Grant the EvalHub job ServiceAccount access to the DSP API and pipeline management."""
    with Role(
        client=admin_client,
        name="evalhub-jobs-dspa-api",
        namespace=tenant_namespace.name,
        rules=[
            {
                "apiGroups": ["datasciencepipelinesapplications.opendatahub.io"],
                "resources": ["datasciencepipelinesapplications/api"],
                "verbs": ["get", "create"],
            },
        ],
        wait_for_resource=True,
    ) as role:
        with RoleBinding(
            client=admin_client,
            name="evalhub-jobs-dspa-api",
            namespace=tenant_namespace.name,
            role_ref_kind=role.kind,
            role_ref_name=role.name,
            subjects_kind="ServiceAccount",
            subjects_name=evalhub_job_service_account.name,
            subjects_namespace=tenant_namespace.name,
            wait_for_resource=True,
        ) as api_binding:
            with RoleBinding(
                client=admin_client,
                name="evalhub-jobs-pipeline-management",
                namespace=tenant_namespace.name,
                role_ref_kind="Role",
                role_ref_name="ds-pipeline-dspa",
                subjects_kind="ServiceAccount",
                subjects_name=evalhub_job_service_account.name,
                subjects_namespace=tenant_namespace.name,
                wait_for_resource=True,
            ) as pipeline_binding:
                yield role, api_binding, pipeline_binding


# ---------------------------------------------------------------------------
# LLM-d Inference Simulator model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def garak_sim_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    """Create an LLM-d inference simulator serving runtime for garak testing."""
    with ServingRuntime(
        client=admin_client,
        name=LLMdInferenceSimConfig.serving_runtime_name,
        namespace=model_namespace.name,
        annotations={
            "description": "LLM-d Simulator KServe",
            "serving.kserve.io/enable-agent": "false",
        },
        containers=[
            {
                "name": "kserve-container",
                "image": "quay.io/trustyai_testing/llm-d-inference-sim-dataset-builtin"
                "@sha256:79e525cfd57a0d72b7e71d5f1e2dd398eca9315cfbd061d9d3e535b1ae736239",
                "imagePullPolicy": "Always",
                "args": [
                    "--model",
                    LLMdInferenceSimConfig.model_name,
                    "--port",
                    str(LLMdInferenceSimConfig.port),
                ],
                "ports": [{"containerPort": LLMdInferenceSimConfig.port, "protocol": "TCP"}],
                "securityContext": {"allowPrivilegeEscalation": False},
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": LLMdInferenceSimConfig.port, "scheme": "HTTP"},
                    "initialDelaySeconds": 5,
                    "periodSeconds": 10,
                },
            }
        ],
        multi_model=False,
        supported_model_formats=[{"autoSelect": True, "name": LLMdInferenceSimConfig.name}],
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def garak_sim_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    garak_sim_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    """Deploy the LLM-d inference simulator for garak testing."""
    with create_isvc(
        client=admin_client,
        name=LLMdInferenceSimConfig.isvc_name,
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format=LLMdInferenceSimConfig.name,
        runtime=garak_sim_serving_runtime.name,
        wait_for_predictor_pods=False,
        min_replicas=1,
        max_replicas=1,
        resources={
            "requests": {"cpu": "1", "memory": "1Gi"},
            "limits": {"cpu": "1", "memory": "1Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def garak_sim_isvc_url(garak_sim_isvc: InferenceService) -> str:
    """Get the internal service URL for the LLM-d inference simulator."""
    return (
        f"http://{garak_sim_isvc.name}-predictor.{garak_sim_isvc.namespace}"
        f".svc.cluster.local:{LLMdInferenceSimConfig.port}/v1"
    )
