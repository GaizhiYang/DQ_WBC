import numpy as np
import copy


def _hat_omega(w):
    wx, wy, wz = w
    return np.array([[0,-wz, wy],
                     [wz, 0,-wx],
                     [-wy,wx, 0]], dtype=float)

def _so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    W = _hat_omega(w)
    A = np.sin(theta)/theta
    B = (1-np.cos(theta))/(theta**2)
    return np.eye(3) + A*W + B*(W@W)

def _so3_log(R):
    c = (np.trace(R)-1)/2
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-9:
        return np.zeros(3)
    W = (R - R.T) / (2*np.sin(theta))
    return np.array([W[2,1], W[0,2], W[1,0]]) * theta

def _left_jac_SO3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        # 1st order
        return np.eye(3) + 0.5*_hat_omega(w)
    W = _hat_omega(w)
    A = (1 - np.cos(theta))/(theta**2)
    B = (theta - np.sin(theta))/(theta**3)
    return np.eye(3) + A*W + B*(W@W)

def se3_exp(xi):
    v = xi[:3]; w = xi[3:]
    R = _so3_exp(w)
    J = _left_jac_SO3(w)
    t = J @ v
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def se3_log(T):
    R = T[:3,:3]
    t = T[:3, 3]
    w = _so3_log(R)
    J = _left_jac_SO3(w)
    # evita instabilità nella J^-1
    if np.linalg.cond(J) > 1e8:
        v = t
    else:
        v = np.linalg.solve(J, t)
    return np.hstack([v, w])

def adjoint(T):
    R = T[:3,:3]
    t = T[:3, 3]
    t_hat = _hat_omega(t)
    Adj = np.zeros((6,6))
    Adj[:3,:3] = R
    Adj[3:,3:] = R
    Adj[:3,3:] = t_hat @ R
    return Adj


# def ekf_predict_horizon(kf, N=20, dt=1.0,
#                         max_trace=None,
#                         max_pos_std=None,
#                         max_rot_std_deg=None
#                         ):
#     """
#     Predice le prossime N pose SE(3) usando SOLO predict() su una copia del filtro.
#     Non modifica lo stato del kf originale.
    
#     Gate di confidenza (opzionali):
#       - max_trace:     soglia sul trace(P) complessivo
#       - max_pos_std:   soglia su std posizione (m) per ciascun asse (x,y,z)
#       - max_rot_std_deg: soglia su std rotazione (deg) per ciascun asse rotazionale (rx,ry,rz)
      
#     Ritorna:
#       - T_preds:  lista di (4,4) np.ndarray lunghezza N  (oppure [])
#       - ok:       bool (True se passa il gating di incertezza)
#       - info:     dict con metriche utili
#     """
#     # --- 0) Gating sull'incertezza attuale ---
#     P = kf.P
#     pose_cov = P[:6, :6]
#     tr = float(np.trace(P))
#     pos_std = np.sqrt(np.maximum(np.diag(pose_cov)[:3], 0.0))  # [σx, σy, σz]
#     rot_std = np.sqrt(np.maximum(np.diag(pose_cov)[3:6], 0.0)) # [σrx,σry,σrz] in rad
#     rot_std_deg = np.rad2deg(rot_std)

#     ok = True
#     if max_trace is not None and tr > max_trace:
#         ok = False
#     if max_pos_std is not None and np.any(pos_std > max_pos_std):
#         ok = False
#     if max_rot_std_deg is not None and np.any(rot_std_deg > max_rot_std_deg):
#         ok = False

#     info = {
#         "trace_P": tr,
#         "pos_std_xyz": pos_std,
#         "rot_std_deg_rpy": rot_std_deg
#     }
#     if not ok:
#         return [], False, info

#     # --- 1) Clona il filtro e fai solo predict ---
#     kf_tmp = copy.deepcopy(kf)

#     T_preds = []
#     for _ in range(N):
#         kf_tmp.predict(dt=dt)
#         T_preds.append(kf_tmp.T.copy())

#     return T_preds, ok, info

def ekf_predict_horizon(kf, N=20, dt=1.0):
    """
    Predice le prossime N pose SE(3) usando SOLO predict() su una copia del filtro.
    Non modifica lo stato del kf originale.

    Ritorna:
      - T_preds: lista di pose future (4x4)
    """
    kf_tmp = copy.deepcopy(kf)

    T_preds = []
    for _ in range(N):
        kf_tmp.predict(dt=dt)
        T_preds.append(kf_tmp.T.copy())

    return T_preds

def ekf_health_check(obj_kf, prev_can_plan, stable_steps, good_count, bad_count, uncertainty_threshold=0.0, nis_threshold=0.0, trans_innovation_threshold=0.0, rot_innovation_threshold=0.0, verbose=True):
    """
    Single gate to check if the EKF is healthy enough for planning.
    """
    if obj_kf is None:
        status = {
            "trace_P": float("inf"),
            "trans_std_max": None,
            "rot_std_max_deg": None,
            "uncertainty_bad": True,
            "NIS": None,
            "trans_innov": None,
            "rot_innov_deg": None,
            "had_measurement_update": False,
            "innovation_bad": False,
            "planning_inhibit": True,
            "can_plan": False,
        }
        if verbose:
            print("[EKF] No filter initialized -> planning inhibited")
        return status

    # -------------------------
    # 1) Covariance / uncertainty
    # -------------------------
    P = obj_kf.P
    trace_P = float(np.trace(P))

    pose_cov = P[:6, :6]
    pos_std = np.sqrt(np.maximum(np.diag(pose_cov)[:3], 0.0))
    rot_std = np.sqrt(np.maximum(np.diag(pose_cov)[3:], 0.0))

    trans_std_max = float(np.max(pos_std))
    rot_std_max_deg = float(np.max(np.rad2deg(rot_std)))

    uncertainty_bad = (
        trace_P > uncertainty_threshold
    )

    # -------------------------
    # 2) Innovation / consistency with measurement
    # -------------------------
    innov = getattr(obj_kf, "last_innovation", None)

    if innov is None:
        nis = None
        tnorm = None
        rnorm = None
        had_measurement_update = False
        innovation_bad = False
    else:
        nis = float(innov["NIS"])
        tnorm = float(innov["trans_norm"])
        rnorm = float(innov["rot_norm_deg"])
        had_measurement_update = True

        innovation_bad = (
            (nis > nis_threshold)
            or (tnorm > trans_innovation_threshold)
            or (rnorm > rot_innovation_threshold)
        )

    # -------------------------
    # 3) Instantaneous gate
    # -------------------------
    instant_good = not (uncertainty_bad or innovation_bad)

    if instant_good:
        good_count += 1
        bad_count = 0
    else:
        bad_count += 1
        good_count = 0

    # -------------------------
    # 4) Stable gate with hysteresis
    # -------------------------
    can_plan = prev_can_plan

    if not prev_can_plan:
        if good_count >= stable_steps:
            can_plan = True
    else:
        if bad_count >= 1:
            can_plan = False

    planning_inhibit = not can_plan

    abort_current_plan = had_measurement_update and innovation_bad
    
    status = {
        "trace_P": trace_P,
        "trans_std_max": trans_std_max,
        "rot_std_max_deg": rot_std_max_deg,
        "uncertainty_bad": uncertainty_bad,
        "NIS": nis,
        "trans_innov": tnorm,
        "rot_innov_deg": rnorm,
        "had_measurement_update": had_measurement_update,
        "innovation_bad": innovation_bad,
        "instant_good": instant_good,
        "planning_inhibit": planning_inhibit,
        "can_plan": can_plan,
        "good_count": good_count,
        "bad_count": bad_count,
        "abort_current_plan": abort_current_plan
    }

    if verbose:
        print(
            f"[EKF] traceP={trace_P:.3f}, "
            f"NIS={nis}, ‖y_t‖={tnorm}, ‖y_r‖={rnorm}"
        )

    return status

class SE3KF:
    """
    Stato: T (SE3 pose assoluta nel mondo), w (twist 6D).
    Filtro EKF su stato esteso [delta_pose(6), delta_w(6)].
    """
    def __init__(self, T0=None, w0=None, P0=None, Qw=np.diag([1e-4, 1e-4, 1e-4, 5e-4, 5e-4, 5e-4]), Qt=None, Rm=None):
        self.T = np.eye(4) if T0 is None else T0.copy()
        self.w = np.zeros(6) if w0 is None else w0.copy()
        self.P = np.eye(12) if P0 is None else P0.copy()
        # process noise: sulla velocità e una piccola diffusione sulla posa
        self.Qw = (0.02*np.eye(6)) if Qw is None else Qw
        self.Qt = (1e-4*np.eye(6)) if Qt is None else Qt
        # measurement noise (sulla misura di posa in se(3))
        self.R = (1e-3*np.eye(6)) if Rm is None else Rm

    def predict(self, dt):
        # Propagazione stato
        Xi = self.w * dt
        self.T = self.T @ se3_exp(Xi)   # T_k|k-1

        # Linearizzazione (error-state x=[dpose(6), dw(6)])
        F = np.eye(12)
        A = adjoint(se3_exp(self.w*dt))  # rotazione dell'errore posa
        F[:6,:6] = A
        F[:6,6:] = dt*np.eye(6)          # coupling twist->pose

        Q = np.zeros((12,12))
        Q[:6,:6] = self.Qt
        Q[6:,6:] = self.Qw

        self.P = F @ self.P @ F.T + Q

    def update_with_absolute_pose(self, T_meas):
        T_err = np.linalg.inv(self.T) @ T_meas
        y = se3_log(T_err)                 # 6 = [dv, dw]
        H = np.zeros((6,12)); H[:,:6] = np.eye(6)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self.P = (np.eye(12) - K@H) @ self.P

        # Retract
        self.T = self.T @ se3_exp(dx[:6])
        self.w = self.w + dx[6:]

        # ---- NUOVO: salva/ritorna metriche di innovazione ----
        # NIS = yᵀ S⁻¹ y
        try:
            S_inv = np.linalg.inv(S)
            nis = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            nis = float("inf")

        trans_norm = float(np.linalg.norm(y[:3]))                 # [m]
        rot_norm_deg = float(np.linalg.norm(y[3:])) * 180.0/np.pi # [deg]

        self.last_innovation = {
            "y": y.copy(),
            "NIS": nis,
            "trans_norm": trans_norm,
            "rot_norm_deg": rot_norm_deg
        }
        return self.last_innovation
