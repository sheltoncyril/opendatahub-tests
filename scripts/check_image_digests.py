#!/usr/bin/env python3
"""Check container image references for digest pinning and registry policies.

Scans IMAGE_SOURCES classes and reports:
  - ERROR: images using mutable tag references instead of @sha256: digests
  - WARNING: images pulled from DockerHub (docker.io) which has strict rate limits

Exit codes:
  0  -- no errors (warnings may still be present)
  1  -- one or more images use tag references

Suppress codes:
  # noqa: IMG002  -- suppress tag-without-digest error
  # noqa: IMG003  -- suppress DockerHub warning
"""

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUPPRESS_DIGEST: str = "noqa: IMG002"
SUPPRESS_DOCKERHUB: str = "noqa: IMG003"
DIGEST_PATTERN: re.Pattern[str] = re.compile(r"@sha256:[0-9a-f]{64}$")
DOCKERHUB_PATTERN: re.Pattern[str] = re.compile(r"^(docker\.io)/")


def _find_image_line(source_lines: list[str], start_line: int, image: str) -> int | None:
    normalized = image.removeprefix("oci://")
    for i, line in enumerate(source_lines):
        if normalized in line or any(part in line for part in normalized.split("@sha256:")):
            return start_line + i
    return None


def _is_suppressed(source_lines: list[str], line_no: int | None, marker: str) -> bool:
    return bool(line_no and marker in source_lines[line_no - 1])


def _check_images() -> tuple[list[dict], list[dict]]:
    from scripts.generate_image_manifest import IMAGE_SOURCES, REGISTRY_PATTERN, _normalize_image

    errors = []
    warnings = []

    for component, class_path in sorted(IMAGE_SOURCES.items()):
        module_path, class_name = class_path.rsplit(".", 1)
        source_file = ROOT / module_path.replace(".", "/")
        source_file = source_file.with_suffix(suffix=".py")

        try:
            source_text = source_file.read_text()
        except OSError:
            continue

        source_lines = source_text.splitlines()
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)

        for name, value in inspect.getmembers(cls):
            if name.startswith("_"):
                continue
            if inspect.isclass(value):
                for sub_name, sub_value in inspect.getmembers(value):
                    if sub_name.startswith("_"):
                        continue
                    if isinstance(sub_value, str) and REGISTRY_PATTERN.match(sub_value):
                        _check_single(
                            image=_normalize_image(value=sub_value),
                            attr=f"{name}.{sub_name}",
                            component=component,
                            source_file=source_file,
                            source_lines=source_lines,
                            errors=errors,
                            warnings=warnings,
                        )
            elif isinstance(value, str) and REGISTRY_PATTERN.match(value):
                _check_single(
                    image=_normalize_image(value=value),
                    attr=name,
                    component=component,
                    source_file=source_file,
                    source_lines=source_lines,
                    errors=errors,
                    warnings=warnings,
                )

    return errors, warnings


def _check_single(
    image: str,
    attr: str,
    component: str,
    source_file: Path,
    source_lines: list[str],
    errors: list[dict],
    warnings: list[dict],
) -> None:
    line_no = _find_image_line(source_lines=source_lines, start_line=1, image=image)
    rel_path = str(source_file.relative_to(ROOT))

    if not DIGEST_PATTERN.search(image) and not _is_suppressed(
        source_lines=source_lines, line_no=line_no, marker=SUPPRESS_DIGEST
    ):
        errors.append({
            "file": rel_path,
            "line": line_no or 0,
            "attribute": attr,
            "image": image,
            "component": component,
            "severity": "error",
            "rule": "IMG002",
            "message": "image uses tag reference instead of digest",
        })

    if DOCKERHUB_PATTERN.match(image) and not _is_suppressed(
        source_lines=source_lines, line_no=line_no, marker=SUPPRESS_DOCKERHUB
    ):
        warnings.append({
            "file": rel_path,
            "line": line_no or 0,
            "attribute": attr,
            "image": image,
            "component": component,
            "severity": "warning",
            "rule": "IMG003",
            "message": "image sourced from DockerHub (strict pull rate limits)",
        })


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output all findings as JSON array",
    )
    args = parser.parse_args()

    errors, warnings = _check_images()

    if args.json:
        json.dump(errors + warnings, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if errors else 0

    if errors:
        print(f"Found {len(errors)} image(s) using tag references instead of digests:\n")
        for f in errors:
            loc = f"{f['file']}:{f['line']}" if f["line"] else f["file"]
            print(f"  {loc}: {f['attribute']}")
            print(f"    {f['image']}")
            print()
        print(f"Pin these images with @sha256:<digest> or suppress with '# {SUPPRESS_DIGEST}'")
        print()

    if warnings:
        print(f"Found {len(warnings)} image(s) sourced from DockerHub (rate-limited):\n")
        for f in warnings:
            loc = f"{f['file']}:{f['line']}" if f["line"] else f["file"]
            print(f"  {loc}: {f['attribute']}")
            print(f"    {f['image']}")
            print()
        print(f"Consider mirroring to quay.io or suppress with '# {SUPPRESS_DOCKERHUB}'")
        print()

    if not errors and not warnings:
        print("All registered images use digest references and approved registries.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
