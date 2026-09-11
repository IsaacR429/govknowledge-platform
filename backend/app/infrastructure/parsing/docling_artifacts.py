from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path


BUNDLE_ID = "govknowledge-docling-p0"
ENVIRONMENT_VARIABLE = (
    "GOVKNOWLEDGE_DOCLING_ARTIFACTS_PATH"
)
MANIFEST_NAME = "manifest.json"

LAYOUT_REPOSITORY = (
    "docling-project/docling-layout-heron"
)
LAYOUT_REVISION = (
    "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
)
LAYOUT_FOLDER = (
    "docling-project--docling-layout-heron"
)

TABLE_REPOSITORY = (
    "docling-project/docling-models"
)
TABLE_REVISION = (
    "fc0f2d45e2218ea24bce5045f58a389aed16dc23"
)
TABLE_FOLDER = (
    "docling-project--docling-models"
)
TABLE_MODEL_PATH = (
    Path("model_artifacts")
    / "tableformer"
)

OCR_ENGINE = "rapidocr"
OCR_BACKEND = "onnxruntime"
OCR_MODEL_FAMILY = "PP-OCRv6"

OCR_FOLDER = "rapidocr"

OCR_MODEL_FOLDER = Path(
    "ppocr-v6"
)

OCR_ASSET_PATHS = {
    "det": (
        OCR_MODEL_FOLDER
        / "PP-OCRv6_det_small.onnx"
    ),
    "cls": (
        OCR_MODEL_FOLDER
        / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    ),
    "rec": (
        OCR_MODEL_FOLDER
        / "PP-OCRv6_rec_small.onnx"
    ),
}

OCR_ASSET_SHA256 = {
    "det": (
        "090f04abcd9d9a7498bc4ebf677e4cb9"
        "bdce1fe4197ddb7e529f1ef44e1ff94f"
    ),
    "cls": (
        "e47acedf663230f8863ff1ab0e64dd2d"
        "82b838fceb5957146dab185a89d6215c"
    ),
    "rec": (
        "6f327246b50388f3c176ae304bd95767"
        "ea6dc0c9ae92153ef8cbe210b3c14884"
    ),
}

CHUNKING_STRATEGY = (
    "paragraph_greedy_char_limit"
)

P0_MAX_CHUNK_CHARS = 1500

EXPECTED_PACKAGE_VERSIONS = {
    "docling": "2.65.0",
    "docling-core": "2.96.0",
    "docling-ibm-models": "3.15.0",
    "docling-parse": "4.7.3",
    "rapidocr": "3.9.2",
    "onnxruntime": "1.30.0",
    "torch": "2.14.0",
}

EXPECTED_BUNDLE_REVISION = (
    "64253811295ecd8b2b20ed3d3b9f58a0"
    "ebefe90b1882df56bf23d5632b6f6d4d"
)


class DoclingArtifactBundleError(
    RuntimeError
):
    """Docling P0 artifact bundle is invalid."""


@dataclass(frozen=True)
class DoclingArtifactBundle:
    root: Path
    bundle_id: str
    bundle_revision: str

    ocr_det_model_path: Path
    ocr_cls_model_path: Path
    ocr_rec_model_path: Path

    ocr_model_family: str

    chunking_strategy: str
    max_chunk_chars: int


def validate_runtime_package_versions() -> None:
    for package, expected in (
        EXPECTED_PACKAGE_VERSIONS.items()
    ):
        actual = version(package)

        if actual != expected:
            raise DoclingArtifactBundleError(
                "Docling runtime package version "
                f"mismatch for {package}: "
                f"expected {expected}, "
                f"found {actual}."
            )


def compute_file_sha256(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            1024 * 1024
        ):
            digest.update(block)

    return digest.hexdigest()


def compute_bundle_sha256(
    root: Path,
) -> str:
    digest = sha256()

    files: list[Path] = []

    for folder_name in (
        LAYOUT_FOLDER,
        TABLE_FOLDER,
        OCR_FOLDER,
    ):
        folder = root / folder_name

        if not folder.is_dir():
            raise DoclingArtifactBundleError(
                "Required Docling artifact folder "
                f"is missing: {folder_name}."
            )

        for path in folder.rglob("*"):
            if path.is_symlink():
                raise DoclingArtifactBundleError(
                    "Docling artifact bundle "
                    "must not contain symlinks."
                )

            if path.is_file():
                files.append(path)

    for path in sorted(
        files,
        key=lambda value: (
            value.relative_to(root).as_posix()
        ),
    ):
        relative = (
            path.relative_to(root)
            .as_posix()
        )

        digest.update(
            relative.encode("utf-8")
        )
        digest.update(b"\0")

        with path.open("rb") as handle:
            while block := handle.read(
                1024 * 1024
            ):
                digest.update(block)

        digest.update(b"\0")

    return digest.hexdigest()


def load_docling_artifact_bundle(
    root: Path | None = None,
) -> DoclingArtifactBundle:
    validate_runtime_package_versions()

    if root is None:
        configured = os.getenv(
            ENVIRONMENT_VARIABLE
        )

        if not configured:
            raise DoclingArtifactBundleError(
                f"{ENVIRONMENT_VARIABLE} "
                "must point to the frozen "
                "Docling artifact bundle."
            )

        root = Path(configured)

    root = root.expanduser().resolve()

    manifest_path = (
        root / MANIFEST_NAME
    )

    if not manifest_path.is_file():
        raise DoclingArtifactBundleError(
            "Docling artifact manifest "
            "is missing."
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise DoclingArtifactBundleError(
            "Docling artifact manifest "
            "could not be read."
        ) from exc

    if (
        manifest.get("bundle_id")
        != BUNDLE_ID
    ):
        raise DoclingArtifactBundleError(
            "Unexpected Docling bundle id."
        )

    models = manifest.get("models", {})

    expected_models = {
        "layout": {
            "repository": (
                LAYOUT_REPOSITORY
            ),
            "revision": LAYOUT_REVISION,
        },
        "table": {
            "repository": (
                TABLE_REPOSITORY
            ),
            "revision": TABLE_REVISION,
        },
    }

    for name, expected in (
        expected_models.items()
    ):
        actual = models.get(name)

        if actual != expected:
            raise DoclingArtifactBundleError(
                "Docling artifact model "
                f"identity mismatch: {name}."
            )

    expected_ocr = {
        "engine": OCR_ENGINE,
        "backend": OCR_BACKEND,
        "model_family": OCR_MODEL_FAMILY,
        "rec_keys_path": None,
        "assets": {
            name: {
                "path": (
                    relative_path.as_posix()
                ),
                "sha256": (
                    OCR_ASSET_SHA256[name]
                ),
            }
            for name, relative_path in (
                OCR_ASSET_PATHS.items()
            )
        },
    }

    if manifest.get("ocr") != expected_ocr:
        raise DoclingArtifactBundleError(
            "Docling OCR configuration "
            "does not match P0."
        )

    expected_chunking = {
        "strategy": CHUNKING_STRATEGY,
        "max_chunk_chars": (
            P0_MAX_CHUNK_CHARS
        ),
    }

    if (
        manifest.get("chunking")
        != expected_chunking
    ):
        raise DoclingArtifactBundleError(
            "Docling chunking configuration "
            "does not match P0."
        )

    ocr_root = root / OCR_FOLDER

    for name, relative_path in (
        OCR_ASSET_PATHS.items()
    ):
        artifact = (
            ocr_root
            / relative_path
        )

        if not artifact.is_file():
            raise DoclingArtifactBundleError(
                "Required RapidOCR artifact "
                f"is missing: {relative_path}."
            )

        actual_hash = compute_file_sha256(
            artifact
        )

        expected_hash = (
            OCR_ASSET_SHA256[name]
        )

        if actual_hash != expected_hash:
            raise DoclingArtifactBundleError(
                "RapidOCR artifact integrity "
                f"verification failed: {name}."
            )

    if (
        manifest.get("packages")
        != EXPECTED_PACKAGE_VERSIONS
    ):
        raise DoclingArtifactBundleError(
            "Docling artifact package "
            "manifest does not match P0."
        )

    table_path = (
        root
        / TABLE_FOLDER
        / TABLE_MODEL_PATH
    )

    if not table_path.is_dir():
        raise DoclingArtifactBundleError(
            "Frozen TableFormer artifacts "
            "are missing."
        )

    expected_hash = manifest.get(
        "content_sha256"
    )

    if (
        not isinstance(
            expected_hash,
            str,
        )
        or len(expected_hash) != 64
    ):
        raise DoclingArtifactBundleError(
            "Docling bundle hash is invalid."
        )

    actual_hash = compute_bundle_sha256(
        root
    )

    if actual_hash != expected_hash:
        raise DoclingArtifactBundleError(
            "Docling artifact bundle "
            "integrity verification failed."
        )

    if actual_hash != EXPECTED_BUNDLE_REVISION:
        raise DoclingArtifactBundleError(
            "Docling artifact bundle revision "
            "does not match the accepted P0 "
            "revision."
        )

    return DoclingArtifactBundle(
        root=root,
        bundle_id=BUNDLE_ID,
        bundle_revision=actual_hash,

        ocr_det_model_path=(
            ocr_root
            / OCR_ASSET_PATHS["det"]
        ),
        ocr_cls_model_path=(
            ocr_root
            / OCR_ASSET_PATHS["cls"]
        ),
        ocr_rec_model_path=(
            ocr_root
            / OCR_ASSET_PATHS["rec"]
        ),

        ocr_model_family=(
            OCR_MODEL_FAMILY
        ),

        chunking_strategy=(
            CHUNKING_STRATEGY
        ),
        max_chunk_chars=(
            P0_MAX_CHUNK_CHARS
        ),
    )
