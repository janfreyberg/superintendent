"""
Functions to prioritise labelling data points (to drive active learning).
"""

from typing import Dict, Callable, Union, List
import numpy as np
from .functions import entropy, margin, certainty, bald, random

__all__ = ["entropy", "margin", "certainty", "bald", "functions", "random"]

AcquisitionFunction = Callable[
    [Union[np.ndarray, List[np.ndarray]]], np.ndarray
]

functions: Dict[str, AcquisitionFunction] = {
    "entropy": entropy,
    "margin": margin,
    "certainty": certainty,
    "bald": bald,
    "random": random,
}
"""A dictionary of functions to prioritise data."""
