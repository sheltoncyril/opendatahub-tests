# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.resource import NamespacedResource


class EvalHub(NamespacedResource):
    """
    EvalHub is the Schema for the evalhubs API
    """

    api_group: str = NamespacedResource.ApiGroup.TRUSTYAI_OPENDATAHUB_IO

    def __init__(
        self,
        database: dict[str, Any] | None = None,
        env: list[Any] | None = None,
        providers: list[str] | None = None,
        replicas: int | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            database (dict[str, Any]): Database configuration for the eval-hub service

            env (list[Any]): Environment variables for the eval-hub container

            providers (list[str]): List of evaluation providers to enable

            replicas (int): Number of replicas for the eval-hub deployment

        """
        super().__init__(**kwargs)

        self.database = database
        self.env = env
        self.providers = providers
        self.replicas = replicas

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.database is not None:
                _spec["database"] = self.database

            if self.env is not None:
                _spec["env"] = self.env

            if self.providers is not None:
                _spec["providers"] = self.providers

            if self.replicas is not None:
                _spec["replicas"] = self.replicas

    # End of generated code
