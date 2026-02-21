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
    "API_ERROR": {
        "message": "Cloud service unavailable",
        "suggestion": "Try local mode or retry later",
    },
    "MESH_REPAIR_FAILED": {
        "message": "Could not create printable mesh",
        "suggestion": "Try different source images",
    },
    "FILE_IO_ERROR": {
        "message": "Could not save/load file",
        "suggestion": "Check disk space and permissions",
    },
    "UNSUPPORTED_FILE_FORMAT": {
        "message": "Unsupported image format",
        "suggestion": "Use JPG, PNG, or HEIC images",
    },
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}


def make_error(command: str, code: str) -> dict:
    details = ERROR_DEFINITIONS[code]
    return {
        "type": "error",
        "command": command,
        "errorCode": code,
        "message": details["message"],
        "suggestion": details["suggestion"],
    }
