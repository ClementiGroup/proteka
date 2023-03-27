from enum import Enum

__all__ = [
    "PerFrameQuantity",
    "PRESET_BUILTIN_QUANTITIES",
]


class PerFrameQuantity(str, Enum):
    """Enum class for some known per-frame quantities. Carrying the shape as a string."""

    SCALAR = "[n_frames]"
    ATOMIC_VECTOR = "[n_frames, n_atoms, 3]"
    BOX_VECTOR = "[n_frames, 3, 3]"
    BOX_QUANTITIES = "[n_frames, 3]"


PFQ = PerFrameQuantity
PRESET_BUILTIN_QUANTITIES = {
    "coords": (PFQ.ATOMIC_VECTOR, "[L]"),
    "time": (PFQ.SCALAR, "[T]"),
    "forces": (PFQ.ATOMIC_VECTOR, "[E]/[L]"),
    "velocities": (PFQ.ATOMIC_VECTOR, "[L]/[T]"),
    "cell_lengths": (PFQ.BOX_QUANTITIES, "[L]"),
    "cell_angles": (PFQ.BOX_QUANTITIES, "degree"),
    "weights": (PFQ.SCALAR, "dimensionless"),
}
