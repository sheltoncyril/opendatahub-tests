EVALHUB_SERVICE_NAME: str = "evalhub"
EVALHUB_SERVICE_PORT: int = 8443
EVALHUB_CONTAINER_PORT: int = 8080
EVALHUB_HEALTH_PATH: str = "/api/v1/health"
EVALHUB_PROVIDERS_PATH: str = "/api/v1/evaluations/providers"
EVALHUB_HEALTH_STATUS_HEALTHY: str = "healthy"

EVALHUB_APP_LABEL: str = "eval-hub"

# CRD details
EVALHUB_API_GROUP: str = "trustyai.opendatahub.io"
EVALHUB_API_VERSION: str = "v1alpha1"
EVALHUB_KIND: str = "EvalHub"
EVALHUB_PLURAL: str = "evalhubs"

# Evaluation job API paths
EVALHUB_JOBS_PATH: str = "/api/v1/evaluations/jobs"

# Garak provider
GARAK_PROVIDER_ID: str = "garak-kfp"
GARAK_BENCHMARK_ID: str = "quick"
GARAK_JOB_TIMEOUT: int = 1800  # 30 minutes
GARAK_JOB_POLL_INTERVAL: int = 30  # seconds

# Tenant namespace
EVALHUB_TENANT_LABEL_KEY: str = "evalhub.trustyai.opendatahub.io/tenant"
EVALHUB_TENANT_LABEL_VALUE: str = "true"

# Job service account naming
EVALHUB_JOB_SA_PREFIX: str = "evalhub-"
EVALHUB_JOB_SA_SUFFIX: str = "-job"

