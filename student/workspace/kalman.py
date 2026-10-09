"""Extended Kalman filter helpers for 6D constant-velocity motion.

Part E supplies prediction and correction for docs/HUONG_DAN_KY_THUAT.md §2.
Read the shared time step and process-noise settings with get_tracking_params().
"""

from __future__ import annotations

from fusion_lab.workspace_support import get_tracking_params
from typing import Any, Optional

import numpy as np

Matrix = np.matrix | np.ndarray
# vi: Gợi ý module — params = get_tracking_params() sau khi import ở trên.


def build_F(dt: Optional[float] = None) -> Matrix:
    """Build the constant-velocity state transition matrix F.

    Args:
        dt: Time step in seconds; default from tracking params.

    Returns:
        6x6 state transition matrix as ``np.matrix``.
    """
    # vi: TODO Part E — Nếu dt is None, lấy params.dt từ get_tracking_params().
    # vi: F = I_6; gán F[0,3]=F[1,4]=F[2,5]=dt (vị trí += v * dt).
    if dt is None:
        dt = get_tracking_params().dt

    f = np.eye(6)
    f[0, 3] = dt
    f[1, 4] = dt
    f[2, 5] = dt

    return np.matrix(f)


def build_Q(dt: Optional[float] = None, q: Optional[float] = None) -> Matrix:
    """Build the process noise covariance matrix Q.

    Args:
        dt: Time step; default from tracking params.
        q: Process noise scale; default from tracking params.

    Returns:
        6x6 process noise matrix.
    """
    # vi: TODO Part E — Q đường chéo: q_diag = dt * q trên 6 trục (mô hình lab).
    
    if dt is None:
        dt = get_tracking_params().dt

    if q is None:
        q = get_tracking_params().q

    Q = np.eye(6) * (dt * q)
    return np.matrix(Q)


def ekf_predict(
    x: Matrix,
    P: Matrix,
    F: Optional[Matrix] = None,
    Q: Optional[Matrix] = None,
) -> tuple[Matrix, Matrix]:
    """Predict state and covariance one time step forward.

    Args:
        x: State vector (6x1).
        P: State covariance (6x6).
        F: Optional transition matrix; build via ``build_F`` if None.
        Q: Optional process noise; build via ``build_Q`` if None.

    Returns:
        Tuple ``(x_pred, P_pred)``.
    """
    if F is None:
        F = build_F()
    if Q is None:
        Q = build_Q()

    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred

def innovation(x: Matrix, meas: Any) -> Matrix:
    # x: trạng thái dự đoán, gồm vị trí và vận tốc.
    # h(x): chuyển trạng thái thành kết quả đo dự kiến của cảm biến.
    # Ví dụ: camera chuyển vị trí 3D thành tọa độ pixel trên ảnh.
    z_pred = meas.sensor.get_hx(x)

    # meas.z: kết quả cảm biến thực sự đo được.
    # Độ chênh = đo thực tế - đo dự kiến.
    gamma = meas.z - z_pred

    return gamma


def innovation_covariance(P: Matrix, meas: Any, H: Matrix) -> Matrix:
    # P: độ bất định của trạng thái dự đoán.
    # H: Jacobian của h(x), mô tả kết quả đo thay đổi
    # thế nào khi trạng thái thay đổi.
    # H @ P @ H.T: chuyển độ bất định sang không gian đo.
    P_meas = H @ P @ H.T

    # meas.R: độ bất định của phép đo từ cảm biến.
    # Cộng hai nguồn bất định để có độ bất định của innovation.
    S = P_meas + meas.R

    return S
def ekf_update(x: Matrix, P: Matrix, meas: Any) -> tuple[Matrix, Matrix]:
    # Jacobian: mô tả kết quả đo thay đổi theo trạng thái.
    H = meas.sensor.get_H(x)

    # Độ chênh giữa phép đo thực tế và phép đo dự kiến.
    gamma = innovation(x, meas)

    # Độ bất định của độ chênh đó.
    S = innovation_covariance(P, meas, H)

    # Kalman gain: quyết định mức điều chỉnh dựa trên phép đo.
    # np.linalg.inv(S) tính ma trận nghịch đảo của S.
    K = P @ H.T @ np.linalg.inv(S)

    # Điều chỉnh trạng thái bằng một phần của độ chênh.
    x_upd = x + K @ gamma

    # Cập nhật độ bất định sau khi nhận thêm thông tin đo.
    I = np.eye(P.shape[0])
    P_upd = (I - K @ H) @ P

    return x_upd, P_upd