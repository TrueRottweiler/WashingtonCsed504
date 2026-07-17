"""Framework-specific exceptions with actionable failure messages."""


class CVSearchError(RuntimeError):
    """Base exception for search-framework errors."""


class ConfigurationError(CVSearchError, ValueError):
    """Raised when configuration values are missing, inconsistent, or unsafe."""


class TrialRejected(CVSearchError):
    """Raised when a candidate violates a pre-allocation or hard study constraint."""


class CheckpointError(CVSearchError):
    """Raised when a checkpoint is absent, corrupt, or incompatible."""


class InsufficientResources(CVSearchError):
    """Raised when disk, memory, or configured compute budgets cannot support execution."""
