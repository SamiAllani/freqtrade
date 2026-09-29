"""Freqtrade exporter package (Task 9)."""

from typing import TYPE_CHECKING, Any

__all__ = ["FreqtradeExporter"]

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from monitoring.freqtrade_exporter import FreqtradeExporter


def __getattr__(name: str) -> Any:
    """Lazily re-export ``FreqtradeExporter`` without side effects at import.

    The exporter module registers Prometheus metrics at import time. Eagerly
    importing it here means ``python -m monitoring.freqtrade_exporter`` first
    imports this package (registering metrics once) and then executes the
    module as ``__main__`` (registering them a second time), which crashes
    with ``DuplicateTimeseries: {'freqtrade_up'}``. A PEP 562 lazy export
    keeps ``from monitoring import FreqtradeExporter`` working while letting
    ``-m`` execution register metrics exactly once.
    """
    if name == "FreqtradeExporter":
        from monitoring.freqtrade_exporter import FreqtradeExporter

        return FreqtradeExporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
