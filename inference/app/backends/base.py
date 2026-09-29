"""Backend interface and shared backend errors (Task 5).

Task 6 (MCP backend) plugs into the same ``Backend`` protocol.
"""

from typing import Protocol

from common.contracts import PredictRequest, PredictResponse


class BackendError(RuntimeError):
    """A backend failed at predict time. Mapped to HTTP 502."""


class WindowTooSmallError(ValueError):
    """Fewer candles than the model window. Mapped to HTTP 422."""


class Backend(Protocol):
    async def predict(self, req: PredictRequest) -> PredictResponse:
        """Run inference for ``req`` and return a contract response."""
        ...
