"""Pure joint-space mappings shared by reset coordination tests and nodes."""

import numpy as np


def factr_target_from_xarm(
    factr_origin, xarm_position, xarm_initial, xarm_to_factr_signs=None
):
    """Map absolute xArm radians to the corresponding FACTR absolute target.

    Args:
        factr_origin: FACTR teleoperation origin, float array shape ``(7,)``.
        xarm_position: Current absolute xArm angles, shape ``(7,)`` in rad.
        xarm_initial: xArm teleoperation initial angles, shape ``(7,)`` in rad.
        xarm_to_factr_signs: Optional joint-direction map, shape ``(7,)`` with
            values ``-1`` or ``1``. ``None`` preserves the legacy identity map.

    Returns:
        Float array shape ``(7,)`` equal to
        ``origin + signs * (xarm - initial)``.
    """

    signs = (
        np.ones(7, dtype=float)
        if xarm_to_factr_signs is None
        else np.asarray(xarm_to_factr_signs, dtype=float)
    )
    return np.asarray(factr_origin, dtype=float) + signs * (
        np.asarray(xarm_position, dtype=float) - np.asarray(xarm_initial, dtype=float)
    )


def rebase_factr_origin(
    factr_position, xarm_position, xarm_initial, xarm_to_factr_signs=None
):
    """Compute a FACTR origin that makes the next xArm command continuous.

    Args:
        factr_position: Measured FACTR angles, float array shape ``(7,)`` in rad.
        xarm_position: Measured xArm angles, float array shape ``(7,)`` in rad.
        xarm_initial: Fixed xArm initial angles, float array shape ``(7,)`` in rad.
        xarm_to_factr_signs: Optional joint-direction map, shape ``(7,)`` with
            values ``-1`` or ``1``. ``None`` preserves the legacy identity map.

    Returns:
        Float FACTR origin array shape ``(7,)`` satisfying the forward delta map.
    """

    signs = (
        np.ones(7, dtype=float)
        if xarm_to_factr_signs is None
        else np.asarray(xarm_to_factr_signs, dtype=float)
    )
    return np.asarray(factr_position, dtype=float) - signs * (
        np.asarray(xarm_position, dtype=float) - np.asarray(xarm_initial, dtype=float)
    )
