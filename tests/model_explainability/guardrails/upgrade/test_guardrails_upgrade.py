import pytest
import yaml
from tests.model_explainability.guardrails.constants import (
    PII_INPUT_DETECTION_PROMPT,
    PII_OUTPUT_DETECTION_PROMPT,
    HARMLESS_PROMPT,
    PII_ENDPOINT,
)
from tests.model_explainability.guardrails.utils import (
    verify_health_info_response,
    send_and_verify_unsuitable_input_detection,
    send_and_verify_unsuitable_output_detection,
    send_and_verify_negative_detection,
)
from utilities.constants import LLM_D_CHAT_GENERATION_CONFIG, BUILTIN_DETECTOR_CONFIG, LLMdInferenceSimConfig
from utilities.plugins.constant import OpenAIEnpoints


@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_gateway_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-builtin-upgrade"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": LLM_D_CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {
                "guardrails_gateway_config_data": {
                    "config.yaml": yaml.dump({
                        "orchestrator": {
                            "host": "localhost",
                            "port": 8032,
                        },
                        "detectors": [
                            {
                                "name": "regex",
                                "input": True,
                                "output": True,
                                "detector_params": {"regex": ["email", "ssn"]},
                            },
                        ],
                        "routes": [
                            {"name": "pii", "detectors": ["regex"]},
                            {"name": "passthrough", "detectors": []},
                        ],
                    })
                },
            },
            {
                "orchestrator_config": True,
                "enable_built_in_detectors": True,
                "enable_guardrails_gateway": True,
                "guardrails_gateway_config": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "guardrails_gateway_config")
class TestGuardrailsOrchestratorWithBuiltInDetectorsPreUpgrade:
    """
    Tests if basic functions of the GuardrailsOrchestrator are working properly with the built-in (regex) detectors.
        1. Deploy an LLM using vLLM as a SR.
        2. Deploy the Guardrails Orchestrator.
        3. Check that the Orchestrator is healthy by querying the health and info endpoints of its /health route.
        4. Check that the built-in regex detectors work as expected:
         4.1. Unsuitable input detection.
         4.2. Unsuitable output detection.
         4.3. No detection.
        5. Check that the /passthrough endpoint forwards the
         query directly to the model without performing any detection.
    """

    @pytest.mark.pre_upgrade
    def test_guardrails_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
    ):
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    @pytest.mark.pre_upgrade
    def test_guardrails_builtin_detectors_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_INPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    @pytest.mark.pre_upgrade
    def test_guardrails_builtin_detectors_unsuitable_output(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_output_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_OUTPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    @pytest.mark.parametrize(
        "message, url_path",
        [
            pytest.param(
                HARMLESS_PROMPT,
                PII_ENDPOINT,
                id="harmless_input",
            ),
            pytest.param(PII_INPUT_DETECTION_PROMPT.content, "/passthrough", id="passthrough_endpoint"),
        ],
    )
    @pytest.mark.pre_upgrade
    def test_guardrails_builtin_detectors_negative_detection(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        message,
        url_path,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(message),
            model=LLMdInferenceSimConfig.model_name,
        )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-guardrails-builtin-upgrade"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("guardrails_gateway_config")
class TestGuardrailsOrchestratorWithBuiltInDetectorsPostUpgrade:
    @pytest.mark.post_upgrade
    def test_guardrails_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
    ):
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    @pytest.mark.post_upgrade
    def test_guardrails_builtin_detectors_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_INPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    @pytest.mark.post_upgrade
    def test_guardrails_builtin_detectors_unsuitable_output(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_output_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_OUTPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    @pytest.mark.parametrize(
        "message, url_path",
        [
            pytest.param(
                HARMLESS_PROMPT,
                PII_ENDPOINT,
                id="harmless_input",
            ),
            pytest.param(PII_INPUT_DETECTION_PROMPT.content, "/passthrough", id="passthrough_endpoint"),
        ],
    )
    @pytest.mark.post_upgrade
    def test_guardrails_builtin_detectors_negative_detection(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        message,
        url_path,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(message),
            model=LLMdInferenceSimConfig.model_name,
        )
