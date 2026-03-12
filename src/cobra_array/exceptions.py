# src/cobra_array/exceptions.py
"""
Exceptions for :pkg:`cobra_array`.
"""


# === WARNING ===
class CobraArrayWarning(Warning):
    """Base warning class for :pkg:`cobra_array` package."""


class ParameterIgnoredWarning(CobraArrayWarning):
    """Warning raised when a parameter is ignored."""


# === ERROR ===
class CobraArrayError(Exception):
    """Base error class for :pkg:`cobra_array` package."""


class MissingDependencyError(CobraArrayError, ImportError):
    """Raised when a required dependency is missing."""


class ConvertNoneTypeError(CobraArrayError, TypeError):
    """Raised when conversion of `NoneType` is attempted."""


class UnsupportedNameSpaceError(CobraArrayError):
    """Raised when an unsupported `array namespace` is specified."""


class ArrayConversionError(CobraArrayError):
    """Raised when an error occurs during array conversion in an `array namespace`."""


class NotArrayAPIObjectError(CobraArrayError, ValueError):
    """Raised when an array is not an array API compatible array object."""


class NoArrayInputsError(CobraArrayError, ValueError):
    """Raised when no array inputs are provided."""


class GetArrayNamespaceError(CobraArrayError):
    """Raised when an error occurs while determining the `array namespace`."""
