"""Explicit interpolation of the official eight-step distilled sigma schedule."""


def distilled_sigmas(steps: int = 8) -> list[float]:
    base = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    if steps < 1:
        raise ValueError("steps must be positive")
    result = []
    for i in range(steps + 1):
        position = i * 8 / steps
        low = int(position)
        fraction = position - low
        result.append(base[low] * (1 - fraction) + base[min(low + 1, 8)] * fraction)
    return result
