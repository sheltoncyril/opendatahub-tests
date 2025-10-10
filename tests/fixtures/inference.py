from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import (
    RuntimeTemplates,
    KServeDeploymentType,
    QWEN_MODEL_NAME,
    LLMdInferenceSimConfig,
    Protocols,
)
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def vllm_cpu_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-runtime-cpu-fp16",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.VLLM_CUDA,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        runtime_image="quay.io/rh-aiservices-bu/vllm-cpu-openai-ubi9"
        "@sha256:ada6b3ba98829eb81ae4f89364d9b431c0222671eafb9a04aa16f31628536af2",
        containers={
            "kserve-container": {
                "args": ["--port=8032", "--model=/mnt/models", "--served-model-name={{.Name}}"],
                "ports": [{"containerPort": 8032, "protocol": "TCP"}],
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "shm"}],
            }
        },
        volumes=[{"emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}, "name": "shm"}],
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def qwen_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
    vllm_cpu_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=QWEN_MODEL_NAME,
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="vLLM",
        runtime=vllm_cpu_runtime.name,
        storage_key=minio_data_connection.name,
        storage_path="Qwen2.5-0.5B-Instruct",
        wait_for_predictor_pods=False,
        resources={
            "requests": {"cpu": "2", "memory": "10Gi"},
            "limits": {"cpu": "2", "memory": "12Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def qwen_isvc_url(qwen_isvc: InferenceService) -> str:
    return f"http://{qwen_isvc.name}-predictor.{qwen_isvc.namespace}.svc.cluster.local:8032/v1"


@pytest.fixture(scope="function")
def llm_d_inference_sim_deployment(admin_client, model_namespace: Namespace) -> Generator[Deployment, Any, Any]:
    with Deployment(
        client=admin_client,
        namespace=model_namespace.name,
        name=LLMdInferenceSimConfig.name,
        label=LLMdInferenceSimConfig.label,
        selector={"matchLabels": LLMdInferenceSimConfig.label},
        template={
            "metadata": {
                "labels": LLMdInferenceSimConfig.label,
                "name": LLMdInferenceSimConfig.name,
            },
            "spec": {
                "containers": [
                    {
                        "image": "quay.io/trustyai_testing/llmd-inference-sim-dataset-builtin"
                        "@sha256:1c8891b3bdf7dbe657d8b3945297b550921083bd3df72f9b8d202ffd99beb341",
                        "name": LLMdInferenceSimConfig.container_name,
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "args": [
                            "--model",
                            LLMdInferenceSimConfig.model_name,
                            "--port",
                            str(LLMdInferenceSimConfig.port),
                        ],
                    }
                ]
            },
        },
        replicas=1,
    ) as deployment:
        yield deployment


@pytest.fixture(scope="function")
def llm_d_inference_sim_service(
    admin_client: DynamicClient, model_namespace: Namespace, llm_d_inference_sim_deployment: Deployment
) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        namespace=llm_d_inference_sim_deployment.namespace,
        name=LLMdInferenceSimConfig.service_name,
        ports=[
            {
                "name": LLMdInferenceSimConfig.endpoint_name,
                "port": LLMdInferenceSimConfig.port,
                "protocol": Protocols.TCP,
                "targetPort": LLMdInferenceSimConfig.port,
            }
        ],
        selector=LLMdInferenceSimConfig.label,
    ) as service:
        yield service


@pytest.fixture(scope="function")
def llm_d_inference_sim_route(
    admin_client: DynamicClient, model_namespace: Namespace, llm_d_inference_sim_service: Service
) -> Generator[Route, Any, Any]:
    with Route(
        client=admin_client,
        namespace=llm_d_inference_sim_service.namespace,
        name=LLMdInferenceSimConfig.route_name,
        service=llm_d_inference_sim_service.name,
    ) as route:
        yield route
