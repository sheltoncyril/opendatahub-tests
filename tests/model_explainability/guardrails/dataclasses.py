from dataclasses import dataclass


@dataclass
class GuardrailsDetectionPrompt:
    prompt: str
    detection: str | None = None
    detector_id: str | None = None
    detection_type: str | None = None
