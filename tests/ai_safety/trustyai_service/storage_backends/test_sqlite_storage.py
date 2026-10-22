"""SQLite storage backend.

SQLite is the dev/test/single-instance backend: it shares the SQLAlchemy Core
base with PostgreSQL, so these tests check that the same API surface behaves
identically, that `STORAGE_DATABASE_PATH` is honoured, and that the in-memory
flavour keeps nothing on disk.
"""

from http import HTTPStatus

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from tests.ai_safety.trustyai_service.storage_backends.constants import (
    DATABASE_STORAGE_FORMAT,
    ENV_DATABASE_KIND,
    ENV_SQLITE_PATH,
    ENV_STORAGE_FORMAT,
    HEALTH_STATUS_OK,
    SQLITE_DATA_FILE,
    STORAGE_CHECK_NAME,
)
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    build_upload_payload,
    deployment_pod_selector,
    wait_for_model_observations,
)

LOGGER = structlog.get_logger(name=__name__)


def _service_pod(client: DynamicClient, namespace: str, deployment: Deployment) -> Pod:
    """Return the pod controlled by the operator-created service Deployment."""
    selector = deployment_pod_selector(deployment=deployment)
    pods = list(Pod.get(client=client, namespace=namespace, label_selector=selector))
    assert len(pods) == 1, f"Expected exactly 1 pod for {deployment.name}, found {len(pods)}"
    return pods[0]


@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-sqlite-storage"})],
    indirect=True,
)
class TestSqliteStorageBackend:
    """The SQLite backend serves the same API as PostgreSQL, from a local file."""

    def test_service_reports_sqlite_backend(
        self,
        sqlite_backed_service: Deployment,
        sqlite_storage_client: StorageBackendClient,
    ) -> None:
        """The CR selects DATABASE and its Secret selects SQLite."""
        container = sqlite_backed_service.instance.spec.template.spec.containers[0]
        env = {var["name"]: var.get("value") for var in container.env}
        assert env.get(ENV_STORAGE_FORMAT) == DATABASE_STORAGE_FORMAT
        assert env.get(ENV_SQLITE_PATH) == SQLITE_DATA_FILE
        kind_entry = next(var for var in container.env if var["name"] == ENV_DATABASE_KIND)
        assert kind_entry["valueFrom"]["secretKeyRef"]["key"] == "databaseKind"

        response = sqlite_storage_client.readiness()
        assert response.status_code == HTTPStatus.OK, f"Readiness probe failed: {response.text}"
        checks = {check["name"]: check["status"] for check in response.json()["checks"]}
        assert checks.get(STORAGE_CHECK_NAME) == HEALTH_STATUS_OK, (
            f"Storage readiness is not ok against SQLite: {response.json()}"
        )

    def test_upload_round_trip(self, sqlite_storage_client: StorageBackendClient) -> None:
        """Uploaded rows are stored in and read back from SQLite."""
        model_name = "sqlite-upload-model"
        n_rows = 5

        response = sqlite_storage_client.upload(payload=build_upload_payload(model_name=model_name, n_rows=n_rows))
        assert response.status_code == HTTPStatus.OK, f"Upload failed: {response.text}"

        model_info = wait_for_model_observations(
            storage_client=sqlite_storage_client, model_name=model_name, expected=n_rows
        )
        assert model_info["data"]["observations"] == n_rows
        assert len(model_info["data"]["inputSchema"]["items"]) == 2

    def test_database_written_to_configured_path(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        sqlite_storage_client: StorageBackendClient,
        sqlite_backed_service: Deployment,
    ) -> None:
        """STORAGE_DATABASE_PATH is where the database file actually lands."""
        model_name = "sqlite-path-model"
        assert (
            sqlite_storage_client.upload(payload=build_upload_payload(model_name=model_name, n_rows=2)).status_code
            == HTTPStatus.OK
        )
        wait_for_model_observations(storage_client=sqlite_storage_client, model_name=model_name, expected=2)

        pod = _service_pod(client=admin_client, namespace=model_namespace.name, deployment=sqlite_backed_service)
        listing = pod.execute(
            command=["/bin/sh", "-c", f"ls -l {SQLITE_DATA_FILE}"],
            container="trustyai-service",
            ignore_rc=True,
        )
        assert SQLITE_DATA_FILE in listing, f"No SQLite database at {SQLITE_DATA_FILE}; got: {listing}"

    def test_memory_backend_loses_data_after_service_restart(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        sqlite_memory_storage_client: StorageBackendClient,
        sqlite_memory_backed_service: Deployment,
    ) -> None:
        """`:memory:` round-trips data in the process and writes no database file."""
        model_name = "sqlite-memory-model"
        n_rows = 3

        response = sqlite_memory_storage_client.upload(
            payload=build_upload_payload(model_name=model_name, n_rows=n_rows)
        )
        assert response.status_code == HTTPStatus.OK, f"Upload failed: {response.text}"

        model_info = wait_for_model_observations(
            storage_client=sqlite_memory_storage_client, model_name=model_name, expected=n_rows
        )
        assert model_info["data"]["observations"] == n_rows

        pod = _service_pod(client=admin_client, namespace=model_namespace.name, deployment=sqlite_memory_backed_service)
        pod.delete(wait=True)
        sqlite_memory_backed_service.wait_for_replicas(deployed=True, timeout=600)

        response = sqlite_memory_storage_client.info()
        assert response.status_code == HTTPStatus.OK, f"Info request failed after restart: {response.text}"
        assert model_name not in response.json(), "In-memory SQLite data survived replacement of its process"
