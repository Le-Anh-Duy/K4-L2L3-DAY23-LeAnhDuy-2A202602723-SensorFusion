"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

from fusion_lab.workspace_support import get_tracking_params
import numpy as np


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    transform = np.asarray(meas.sensor.sens_to_veh)
    pos_sensor = np.asarray(meas.z, dtype=float).reshape(-1)[:3]
    pos_veh = transform[:3, :3] @ pos_sensor + transform[:3, 3]

    x = np.matrix(np.r_[pos_veh, [0.0, 0.0, 0.0]]).T

    R_sensor = np.asarray(meas.R, dtype=float)
    R_veh = transform[:3, :3] @ R_sensor @ transform[:3, :3].T

    P = np.zeros((6, 6), dtype=float)
    P[:3, :3] = R_veh
    P[3, 3] = params.sigma_p44**2
    P[4, 4] = params.sigma_p55**2
    P[5, 5] = params.sigma_p66**2

    score = 1.0 / params.window
    state = "initialized"

    return {"x": x, "P": np.matrix(P), "state": state, "score": score}


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    score = track["score"]
    state = track["state"]

    if associated:
        score = min(1.0, score + 1.0 / params.window)
        if state == "confirmed":
            new_state = "confirmed"
        elif score > params.confirmed_threshold:
            new_state = "confirmed"
        else:
            new_state = "tentative"
    else:
        score = score - 1.0 / params.window
        if state == "confirmed":
            new_state = "confirmed"
        else:
            new_state = state

    track["score"] = score
    track["state"] = new_state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    if "P" in track and track["P"] is not None:
        P = np.asarray(track["P"])
        if P[0, 0] > params.max_P or P[1, 1] > params.max_P:
            return True

    if track["state"] == "confirmed":
        return bool(track["score"] < params.delete_threshold)
    else:
        return bool(track["score"] <= 0.0)
