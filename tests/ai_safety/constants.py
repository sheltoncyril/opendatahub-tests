# Shared by every ai_safety component's session-scoped model resources — must stay a single
# global name. See the shared_models_namespace fixture docstring in tests/ai_safety/conftest.py.
AI_SAFETY_SHARED_MODELS_NAMESPACE: str = "ai-safety-models"

VLLM_EMULATOR: str = "vllm-emulator"
VLLM_EMULATOR_PORT: int = 8000
VLLM_EMULATOR_IMAGE: str = (
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
)
