"""Error definitions aligned with SPEC.md."""

ERROR_DEFINITIONS = {
    "INSUFFICIENT_IMAGES": {
        "message": "Not enough images",
        "suggestion": "Add at least 3 images",
    },
    "TOO_MANY_IMAGES": {
        "message": "Too many images for MVP",
        "suggestion": "Use 3-5 images",
    },
    "INSUFFICIENT_FEATURES": {
        "message": "Images too similar or low quality",
        "suggestion": "Try different angles or better lighting",
    },
    "RECONSTRUCTION_FAILED": {
        "message": "3D reconstruction failed",
        "suggestion": "Try different images or cloud mode",
    },
    "PYTHON_DEPENDENCIES_MISSING": {
        "message": "Required Python libraries are missing",
        "suggestion": "Install required local-mode dependencies and retry",
    },
    "MODEL_WEIGHTS_UNAVAILABLE": {
        "message": "Could not load TripoSR model weights",
        "suggestion": "Ensure internet access and enough disk space for first-run model download",
    },
    "API_ERROR": {
        "message": "Cloud service unavailable",
        "suggestion": "Try local mode or retry later",
    },
    "MESH_REPAIR_FAILED": {
        "message": "Could not create printable mesh",
        "suggestion": "Try different source images",
    },
    "OPERATION_CANCELLED": {
        "message": "Operation was cancelled",
        "suggestion": "Retry when ready",
    },
    "FILE_IO_ERROR": {
        "message": "Could not save/load file",
        "suggestion": "Check disk space and permissions",
    },
    "UNSUPPORTED_FILE_FORMAT": {
        "message": "Unsupported image format",
        "suggestion": "Use JPG, PNG, HEIC, HEIF, WebP, or AVIF images",
    },
    "IMAGE_QUALITY_WARNING": {
        "message": "One or more images may have quality issues",
        "suggestion": "Use well-lit, sharp images from varied angles at 512x512 or higher",
    },
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".avif"}


def make_error(command: str, code: str) -> dict:
    details = ERROR_DEFINITIONS[code]
    return {
        "type": "error",
        "command": command,
        "errorCode": code,
        "message": details["message"],
        "suggestion": details["suggestion"],
    }
