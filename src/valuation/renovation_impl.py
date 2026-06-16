"""Compatibility import for the renovation cost implementation."""

try:
    from .calculate_renovation_cost import calculate_renovation_cost
except ImportError:
    from calculate_renovation_cost import calculate_renovation_cost


__all__ = ["calculate_renovation_cost"]
