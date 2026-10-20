class Book2AudioException(Exception):
    """Raised for any user-facing error; main() prints it and exits with 1."""


class MissingDependency(Book2AudioException):
    pass


class UnsupportedFormat(Book2AudioException):
    pass


class InputError(Book2AudioException):
    pass


class VoiceError(Book2AudioException):
    pass


class ModelError(Book2AudioException):
    pass
