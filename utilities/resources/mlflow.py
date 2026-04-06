from typing import Any

from ocp_resources.resource import NamespacedResource


class MLflow(NamespacedResource):
    """MLflow is the Schema for the mlflows API."""

    api_group: str = "mlflow.opendatahub.io"

    def __init__(
        self,
        storage: dict[str, Any] | None = None,
        backend_store_uri: str | None = None,
        artifacts_destination: str | None = None,
        serve_artifacts: bool | None = None,
        image: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.storage = storage
        self.backend_store_uri = backend_store_uri
        self.artifacts_destination = artifacts_destination
        self.serve_artifacts = serve_artifacts
        self.image = image

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.storage is not None:
                _spec["storage"] = self.storage

            if self.backend_store_uri is not None:
                _spec["backendStoreUri"] = self.backend_store_uri

            if self.artifacts_destination is not None:
                _spec["artifactsDestination"] = self.artifacts_destination

            if self.serve_artifacts is not None:
                _spec["serveArtifacts"] = self.serve_artifacts

            if self.image is not None:
                _spec["image"] = self.image
