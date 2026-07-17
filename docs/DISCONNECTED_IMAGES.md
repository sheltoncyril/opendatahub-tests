# Consuming Disconnected Images

The `odh-tests` container image embeds a manifest of all container images required by tests
as OCI labels. Disconnected or air-gapped environments can extract this manifest to discover
which images need to be mirrored.

## Labels

| Label                                          | Content                                    |
| ---------------------------------------------- | ------------------------------------------ |
| `io.opendatahub.tests.required-images`         | Compact JSON manifest grouped by component |
| `io.opendatahub.tests.required-images.sha256`  | SHA-256 hex digest of the manifest JSON    |

## Prerequisites

- `skopeo`
- `jq`
- `sha256sum` (coreutils)

## Extracting the image list

Set the image reference:

```sh
IMAGE=quay.io/opendatahub/opendatahub-tests:latest
LABEL=io.opendatahub.tests.required-images
INSPECT=$(skopeo inspect "docker://$IMAGE")
MANIFEST=$(printf '%s' "$INSPECT" | jq -r ".Labels[\"$LABEL\"]")
```

### Option 1: JSON (grouped by component)

```sh
printf '%s' "$MANIFEST" | jq . > required-images.json
```

Output:

```json
{
  "ai_safety": [
    "quay.io/minio/minio@sha256:14cea4...",
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5..."
  ],
  "shared": [
    "ghcr.io/project-zot/zot:v2.1.8",
    "quay.io/opendatahub/openvino_model_server@sha256:564664..."
  ]
}
```

### Option 2: Plain text (one image per line)

```sh
printf '%s' "$MANIFEST" | jq -r '.[][]' > required-images.txt
```

Output:

```text
quay.io/minio/minio@sha256:14cea4...
quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5...
ghcr.io/project-zot/zot:v2.1.8
quay.io/opendatahub/openvino_model_server@sha256:564664...
```

## Verifying the checksum

After extracting the manifest, verify it was not corrupted or tampered with:

```sh
EXPECTED=$(printf '%s' "$INSPECT" | jq -r ".Labels[\"$LABEL.sha256\"]")
ACTUAL=$(printf '%s' "$MANIFEST" | sha256sum | cut -d' ' -f1)

if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "OK: checksum verified ($ACTUAL)"
else
    echo "FAIL: checksum mismatch (expected=$EXPECTED, actual=$ACTUAL)"
fi
```

## Full example

```sh
IMAGE=quay.io/opendatahub/opendatahub-tests:latest
LABEL=io.opendatahub.tests.required-images

# Extract
INSPECT=$(skopeo inspect "docker://$IMAGE")
MANIFEST=$(printf '%s' "$INSPECT" | jq -r ".Labels[\"$LABEL\"]")
EXPECTED=$(printf '%s' "$INSPECT" | jq -r ".Labels[\"$LABEL.sha256\"]")

# Verify
ACTUAL=$(printf '%s' "$MANIFEST" | sha256sum | cut -d' ' -f1)
if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "OK: checksum verified ($ACTUAL)"
else
    echo "FAIL: checksum mismatch (expected=$EXPECTED, actual=$ACTUAL)"
    exit 1
fi

# Save both formats
printf '%s' "$MANIFEST" | jq .    > required-images.json
printf '%s' "$MANIFEST" | jq -r '.[][]' > required-images.txt

echo "Wrote required-images.json and required-images.txt"
```
