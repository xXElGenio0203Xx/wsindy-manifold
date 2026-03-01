"""
Grid specification for WSINDy.

Stores the uniform grid spacings (dt, dx, dy) and boundary conditions
used throughout the weak formulation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    """Uniform spatiotemporal grid metadata.

    Parameters
    ----------
    dt : float
        Temporal spacing.
    dx : float
        Spatial spacing in x.
    dy : float
        Spatial spacing in y.
    periodic_space : bool
        If True the spatial domain is periodic (enables FFT convolutions
        later).  Default ``True``.
    periodic_time : bool
        If True the temporal domain is periodic.  Default ``False``.
    """

    dt: float
    dx: float
    dy: float
    periodic_space: bool = True
    periodic_time: bool = False

    # ── convenience ──────────────────────────────────────────────────────
    @property
    def steps(self) -> tuple[float, float, float]:
        """Return (dt, dx, dy) as a tuple."""
        return (self.dt, self.dx, self.dy)

    def support_half_widths(
        self,
        ellt: int,
        ellx: int,
        elly: int,
    ) -> tuple[float, float, float]:
        """Physical half-widths of the compact support.

        Returns ``(ellt * dt, ellx * dx, elly * dy)``.
        """
        return (ellt * self.dt, ellx * self.dx, elly * self.dy)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GridSpec(dt={self.dt}, dx={self.dx}, dy={self.dy}, "
            f"periodic_space={self.periodic_space}, "
            f"periodic_time={self.periodic_time})"
        )
