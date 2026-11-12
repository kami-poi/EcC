"""Public error categories for input, data, configuration and run failures."""


class InputError(ValueError):
    """A source-code pair cannot be used as model input."""


class DataError(ValueError):
    """Supervision or dataset metadata violates the data contract."""


class ConfigError(ValueError):
    """A run configuration is malformed or inconsistent."""


class ArtifactError(RuntimeError):
    """A saved model or run artifact cannot be safely used."""


class ResourceError(RuntimeError):
    """A run cannot preserve its input contract with available resources."""
