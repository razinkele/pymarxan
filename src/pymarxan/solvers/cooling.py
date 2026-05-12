"""Cooling schedule variants for simulated annealing."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CoolingSchedule:
    """Cooling schedule for simulated annealing."""

    name: str
    initial_temp: float
    final_temp: float = 0.001
    num_steps: int = 10_000
    _alpha: float = 1.0
    _beta: float = 0.0
    # Lazily-built lookup table for Lundy-Mees, where each step depends on the
    # previous one; without this every call would recompute from step 0.
    _lundy_table: list[float] = field(default_factory=list)

    def temperature(self, step: int) -> float:
        """Return temperature at given step index."""
        if self.name == "geometric":
            return self.initial_temp * (self._alpha ** step)
        if self.name == "exponential":
            ratio = math.log(self.initial_temp / self.final_temp) / max(1, self.num_steps)
            return self.initial_temp * math.exp(-step * ratio)
        if self.name == "linear":
            t = self.initial_temp - step * (self.initial_temp - self.final_temp) / max(
                1, self.num_steps
            )
            return max(t, self.final_temp)
        if self.name == "lundy_mees":
            return self._lundy_mees_temperature(step)
        msg = f"Unknown cooling schedule: {self.name}"
        raise ValueError(msg)

    def _lundy_mees_temperature(self, step: int) -> float:
        """Look up Lundy-Mees temperature in O(1) via a precomputed table."""
        if not self._lundy_table:
            t = self.initial_temp
            table = [max(t, self.final_temp)]
            for _ in range(self.num_steps):
                t = t / (1.0 + self._beta * t)
                table.append(max(t, self.final_temp))
            self._lundy_table = table
        idx = min(step, len(self._lundy_table) - 1)
        return self._lundy_table[idx]

    @staticmethod
    def geometric(
        initial_temp: float,
        final_temp: float = 0.001,
        num_steps: int = 10_000,
    ) -> CoolingSchedule:
        """Geometric cooling: T(k) = T0 * alpha^k."""
        alpha = (final_temp / initial_temp) ** (1.0 / max(1, num_steps))
        return CoolingSchedule(
            name="geometric",
            initial_temp=initial_temp,
            final_temp=final_temp,
            num_steps=num_steps,
            _alpha=alpha,
        )

    @staticmethod
    def exponential(
        initial_temp: float,
        final_temp: float = 0.001,
        num_steps: int = 10_000,
    ) -> CoolingSchedule:
        """Exponential cooling: T(k) = T0 * exp(-k * ln(T0/Tf) / n)."""
        return CoolingSchedule(
            name="exponential",
            initial_temp=initial_temp,
            final_temp=final_temp,
            num_steps=num_steps,
        )

    @staticmethod
    def linear(
        initial_temp: float,
        final_temp: float = 0.001,
        num_steps: int = 10_000,
    ) -> CoolingSchedule:
        """Linear cooling: T(k) = T0 - k * (T0 - Tf) / n."""
        return CoolingSchedule(
            name="linear",
            initial_temp=initial_temp,
            final_temp=final_temp,
            num_steps=num_steps,
        )

    @staticmethod
    def lundy_mees(
        initial_temp: float,
        final_temp: float = 0.001,
        num_steps: int = 10_000,
        beta: float | None = None,
    ) -> CoolingSchedule:
        """Lundy-Mees cooling: T(k+1) = T(k) / (1 + beta * T(k)).

        If beta is not provided, it is estimated so that T(num_steps) ≈ final_temp.
        """
        if beta is None:
            # Derive beta so that after num_steps iterations we reach final_temp.
            # 1/T(k) = 1/T0 + k*beta  =>  beta = (1/Tf - 1/T0) / n
            beta = (1.0 / final_temp - 1.0 / initial_temp) / max(1, num_steps)
        return CoolingSchedule(
            name="lundy_mees",
            initial_temp=initial_temp,
            final_temp=final_temp,
            num_steps=num_steps,
            _beta=beta,
        )
