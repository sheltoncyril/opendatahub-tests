# Canary file — tests stray image detection across components
# This file should be flagged by check_stray_images.py

STRAY_MINIO = "quay.io/fake-org/minio-stray:v2024.1"
STRAY_VLLM = "ghcr.io/fake-org/vllm-stray:0.4.0"
SUPPRESSED = "quay.io/fake-org/should-not-flag:v1"  # noqa: IMG001
