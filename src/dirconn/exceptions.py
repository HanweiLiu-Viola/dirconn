"""Warning categories raised by dirconn."""


class DirconnWarning(UserWarning):
    """A diagnostic about the input or the estimate that does not stop the computation.

    Filter it like any warning, for example
    ``warnings.simplefilter("ignore", DirconnWarning)``. The same diagnostic is also recorded in
    the assumption checklist of the result.
    """
