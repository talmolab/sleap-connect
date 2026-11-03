"""Exceptions for sleap-connect client-worker communication."""


class NoWorkersAvailableError(Exception):
    """No workers available matching requirements."""

    pass


class NoWorkersAcceptedError(Exception):
    """No workers accepted the job request."""

    pass


class JobFailedError(Exception):
    """Job execution failed on worker."""

    pass


class WorkerDiscoveryError(Exception):
    """Error during worker discovery process."""

    pass


class JobTimeoutError(Exception):
    """Job execution timed out."""

    pass
