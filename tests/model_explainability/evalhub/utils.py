import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.service_account import ServiceAccount
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_explainability.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
    EVALHUB_JOBS_PATH,
    EVALHUB_PROVIDERS_PATH,
    GARAK_JOB_POLL_INTERVAL,
    GARAK_JOB_TIMEOUT,
)
from utilities.guardrails import get_auth_headers

LOGGER = get_logger(name=__name__)


def validate_evalhub_health(
    host: str,
    token: str,
    ca_bundle_file: str,
) -> None:
    """Validate that the EvalHub service health endpoint returns healthy status.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.

    Raises:
        AssertionError: If the health check fails.
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_HEALTH_PATH}"
    LOGGER.info(f"Checking EvalHub health at {url}")

    response = requests.get(
        url=url,
        headers=get_auth_headers(token=token),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    LOGGER.info(f"EvalHub health response: {data}")

    assert "status" in data, "Health response missing 'status' field"
    assert data["status"] == EVALHUB_HEALTH_STATUS_HEALTHY, (
        f"Expected status '{EVALHUB_HEALTH_STATUS_HEALTHY}', got '{data['status']}'"
    )
    assert "timestamp" in data, "Health response missing 'timestamp' field"


def validate_evalhub_providers(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant_namespace: str,
    expected_providers: list[str],
) -> None:
    """Validate that the EvalHub providers endpoint returns the expected providers."""
    url = f"https://{host}{EVALHUB_PROVIDERS_PATH}"
    LOGGER.info(f"Checking EvalHub providers at {url}")

    response = requests.get(
        url=url,
        headers={**get_auth_headers(token=token), "X-Tenant": tenant_namespace},
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    LOGGER.info(f"EvalHub providers response: {data}")

    provider_ids = [item["resource"]["id"] for item in data.get("items", [])]
    for expected in expected_providers:
        assert expected in provider_ids, f"Expected provider '{expected}' not found in {provider_ids}"


def submit_garak_job(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant_namespace: str,
    payload: dict,
) -> str:
    """Submit a garak evaluation job and return the job ID."""
    url = f"https://{host}{EVALHUB_JOBS_PATH}"
    LOGGER.info(f"Submitting garak job to {url}")

    response = requests.post(
        url=url,
        headers={
            **get_auth_headers(token=token),
            "Content-Type": "application/json",
            "X-Tenant": tenant_namespace,
        },
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )
    if not response.ok:
        LOGGER.error(f"Job submission failed ({response.status_code}): {response.text}")
    response.raise_for_status()

    data = response.json()
    LOGGER.info(f"Garak job submission response: {data}")

    job_id = data.get("id") or data.get("job_id") or (data.get("resource", {}).get("id"))
    assert job_id, f"No job ID in response: {data}"
    return job_id


def get_job_status(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant_namespace: str,
    job_id: str,
) -> dict:
    """Get the status of an evaluation job."""
    url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
    LOGGER.info(f"Checking job status at {url}")

    response = requests.get(
        url=url,
        headers={**get_auth_headers(token=token), "X-Tenant": tenant_namespace},
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    return response.json()


def wait_for_job_completion(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant_namespace: str,
    job_id: str,
    timeout: int = GARAK_JOB_TIMEOUT,
    poll_interval: int = GARAK_JOB_POLL_INTERVAL,
) -> dict:
    """Poll for job completion, returning the final job status."""
    terminal_states = {"completed", "failed", "error"}

    def _check_job_status() -> dict | None:
        data = get_job_status(
            host=host,
            token=token,
            ca_bundle_file=ca_bundle_file,
            tenant_namespace=tenant_namespace,
            job_id=job_id,
        )
        status_field = data.get("status", {})
        status = (status_field.get("state", "") if isinstance(status_field, dict) else str(status_field)).lower()
        LOGGER.info(f"Job {job_id} status: {status}")
        if status in terminal_states:
            return data
        return None

    for result in TimeoutSampler(
        wait_timeout=timeout,
        sleep=poll_interval,
        func=_check_job_status,
    ):
        if result is not None:
            status_field = result.get("status", {})
            status = (status_field.get("state", "") if isinstance(status_field, dict) else str(status_field)).lower()
            assert status == "completed", f"Job {job_id} ended with status '{status}': {result}"
            return result

    return {}


def wait_for_service_account(
    admin_client: DynamicClient,
    namespace: str,
    sa_name: str,
    timeout: int = 360,
) -> ServiceAccount:
    """Wait for a ServiceAccount to be created in the given namespace."""
    LOGGER.info(f"Waiting for ServiceAccount '{sa_name}' in namespace '{namespace}'")

    def _sa_exists() -> ServiceAccount | None:
        try:
            sa = ServiceAccount(client=admin_client, name=sa_name, namespace=namespace)
            if sa.exists:
                return sa
        except Exception:
            pass
        return None

    for sa in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=_sa_exists,
    ):
        if sa is not None:
            LOGGER.info(f"ServiceAccount '{sa_name}' found in namespace '{namespace}'")
            return sa

    raise TimeoutError(f"ServiceAccount '{sa_name}' not found in namespace '{namespace}' within {timeout}s")
