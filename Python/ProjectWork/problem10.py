"""Problem 10: symbolic dynamics and torques along the Problem 6 trajectory.

All dynamic quantities use SI units: metres, kilograms, seconds and radians.
The trajectory coefficients from problem6.py are stored in degrees and are
converted to radians before evaluation.
"""

from __future__ import annotations

import os
import sys
import contextlib
import io
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    from problem6 import T as SEGMENT_TIME
    from problem6 import coeffs as trajectory_coeffs_deg


# Link data from the assignment.
LINK_MASSES = np.array([0.06, 0.08, 0.08, 0.04], dtype=float)
L1, L2, L3, L4 = 0.050, 0.093, 0.093, 0.050

# Rough rectangular-servo estimate (link 1 as a 50 x 40 mm box, m = 60 g):
# I0 = m/12 * (0.050^2 + 0.040^2) = 2.05e-5 kg m^2.
I0 = 0.06 / 12.0 * (0.050**2 + 0.040**2)

# COM coordinates are expressed in the corresponding link frame. (Figure 2)
COM_LOCAL = (
    sp.Matrix([0.0, -0.020, 0.0]),
    sp.Matrix([-0.030, 0.0, 0.0]),
    sp.Matrix([-0.030, 0.0, 0.0]),
    sp.Matrix([-0.025, 0.015, 0.0]),
)

INERTIAS_LOCAL = (
    sp.diag(I0, 0.4 * I0, 0.9 * I0),
    sp.diag(0.45 * I0, 1.4 * I0, 1.2 * I0),
    sp.diag(0.45 * I0, 1.4 * I0, 1.2 * I0),  # link 3 = link 2 (assignment)
    sp.diag(0.5 * I0, 0.5 * I0, 0.5 * I0),
)

GRAVITY = 9.81


def dh(theta: sp.Expr, d: float, a: float, alpha: sp.Expr) -> sp.Matrix:
    """Return the standard DH transform A_i(theta_i, d_i, a_i, alpha_i)."""
    ct = cast(sp.Expr, sp.cos(theta))
    st = cast(sp.Expr, sp.sin(theta))
    ca = cast(sp.Expr, sp.cos(alpha))
    sa = cast(sp.Expr, sp.sin(alpha))
    return sp.Matrix(
        [
            [ct, sp.Mul(-1, st, ca), sp.Mul(st, sa), sp.Mul(a, ct)],
            [st, sp.Mul(ct, ca), sp.Mul(-1, ct, sa), sp.Mul(a, st)],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )


def build_dynamics() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, tuple[sp.Expr, ...]]:
    """Construct D(q), C(q, qdot) and g(q) symbolically."""
    q: tuple[sp.Expr, ...] = tuple(sp.Symbol(f"q{i}", real=True) for i in range(1, 5))
    qdot: tuple[sp.Expr, ...] = tuple(
        sp.Symbol(f"qd{i}", real=True) for i in range(1, 5)
    )
    alpha = (sp.pi / 2, sp.Integer(0), sp.Integer(0), sp.Integer(0))
    transforms = (
        dh(q[0], L1, 0, alpha[0]),
        dh(q[1] + sp.pi / 2, 0, L2, alpha[1]),
        dh(q[2], 0, L3, alpha[2]),
        dh(q[3], 0, L4, alpha[3]),
    )

    # T_0^i is the pose of frame i relative to the base frame.
    frames = [sp.eye(4)]
    for transform in transforms:
        frames.append(frames[-1] * transform)

    origins = [frame[:3, 3] for frame in frames]
    axes = [frame[:3, 2] for frame in frames]
    # The joint-space model is
    #     D(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau.
    d_matrix = sp.zeros(4, 4)
    potential = 0

    for link in range(4):
        frame = frames[link + 1]
        rotation = frame[:3, :3]
        # COM position: o_ci = o_i + R_i r_ci.
        com = origins[link + 1] + rotation * COM_LOCAL[link]
        jv = sp.zeros(3, 4)
        jw = sp.zeros(3, 4)
        for joint in range(link + 1):
            # For a revolute joint:
            # J_vci[:, j] = z_(j-1) x (o_ci - o_(j-1))
            # J_wi[:, j] = z_(j-1).
            jv[:, joint] = axes[joint].cross(com - origins[joint])
            jw[:, joint] = axes[joint]

        # Rotate the link inertia tensor from its COM frame to the base frame.
        inertia_world = rotation * INERTIAS_LOCAL[link] * rotation.T
        # D(q) = sum_i [m_i J_vci^T J_vci
        #                + J_wi^T R_i I_i R_i^T J_wi].
        d_matrix += LINK_MASSES[link] * jv.T * jv + jw.T * inertia_world * jw
        # P(q) = sum_i m_i g z_ci, so g(q) = dP/dq.
        potential += LINK_MASSES[link] * GRAVITY * com[2]

    # D(q) is left unsimplified: trigsimp takes ~4 min and the numbers are identical.
    gravity = sp.Matrix(sp.trigsimp(sp.Matrix([sp.diff(potential, qi) for qi in q])))
    coriolis = sp.zeros(4, 4)
    for row in range(4):
        for column in range(4):
            terms = []
            for index in range(4):
                # Christoffel contribution (k = row, j = column, i = index):
                # c_ijk = 1/2 (d(d_kj)/dq_i + d(d_ki)/dq_j - d(d_ij)/dq_k).
                # C_kj = sum_i c_ijk q_dot_i.
                christoffel = sp.Add(
                    sp.diff(d_matrix[row, column], q[index]),
                    sp.diff(d_matrix[row, index], q[column]),
                    sp.Mul(-1, sp.diff(d_matrix[index, column], q[row])),
                )
                terms.append(sp.Mul(sp.Rational(1, 2), christoffel, qdot[index]))
            coriolis[row, column] = sp.Add(*terms)
    return d_matrix, coriolis, gravity, q


def evaluate_trajectory(d_fn, c_fn, g_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the four quintic segments; return time, total torque and gravity torque."""
    coeffs_rad = np.deg2rad(trajectory_coeffs_deg)
    times = np.linspace(0.0, 4.0 * SEGMENT_TIME, 801)
    torques = np.zeros((times.size, 4))
    gravity_torques = np.zeros((times.size, 4))
    for sample, time in enumerate(times):
        segment = min(int(time // SEGMENT_TIME), 3)
        local_time = time - segment * SEGMENT_TIME
        if sample == times.size - 1:
            segment, local_time = 3, SEGMENT_TIME
        coefficients = coeffs_rad[segment]
        q = np.array([np.polyval(row[::-1], local_time) for row in coefficients])
        qdot = np.array(
            [np.polyval(np.polyder(row[::-1]), local_time) for row in coefficients]
        )
        qddot = np.array(
            [np.polyval(np.polyder(row[::-1], 2), local_time) for row in coefficients]
        )
        d_value = np.asarray(d_fn(*q), dtype=float)
        c_value = np.asarray(c_fn(*q, *qdot), dtype=float)
        g_value = np.asarray(g_fn(*q), dtype=float).reshape(4)
        # tau = D(q) q_ddot + C(q, q_dot) q_dot + g(q).
        # The three terms are inertia, Coriolis/centrifugal, and gravity torques.
        torques[sample] = d_value @ qddot + c_value @ qdot + g_value
        gravity_torques[sample] = g_value
    return times, torques, gravity_torques


def main() -> None:
    d_matrix, coriolis, gravity, q_symbols = build_dynamics()
    print(f"I0 = {I0:.6e} kg m^2")
    print(f"Link masses: {LINK_MASSES} kg")
    print("D(q), C(q, qdot) and g(q) derived symbolically.")
    print("g(q) =")
    sp.print_latex(gravity)

    qdot_symbols = sp.symbols("qd1:5", real=True)
    d_fn = sp.lambdify(q_symbols, d_matrix, "numpy", cse=True)
    c_fn = sp.lambdify((*q_symbols, *qdot_symbols), coriolis, "numpy", cse=True)
    g_fn = sp.lambdify(q_symbols, gravity, "numpy", cse=True)

    times, torques, gravity_torques = evaluate_trajectory(d_fn, c_fn, g_fn)
    torque_data = np.column_stack((times, torques))
    torque_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "problem10_torques.csv")
    np.savetxt(
        torque_file,
        torque_data,
        delimiter=",",
        header="time_s,tau1_Nm,tau2_Nm,tau3_Nm,tau4_Nm",
        comments="",
    )
    print(f"Saved all Problem 6/7 trajectory torques to: {torque_file}")
    d_dot = sp.zeros(4, 4)
    for q_symbol, qdot_symbol in zip(q_symbols, qdot_symbols):
        d_dot += (  # pyright: ignore[reportOperatorIssue]
            sp.Matrix(sp.diff(d_matrix, q_symbol)) * qdot_symbol
        )
    d_dot_fn = sp.lambdify((*q_symbols, *qdot_symbols), d_dot, "numpy", cse=True)
    q_test = np.array([0.4, -0.7, 1.1, 0.3])
    qdot_test = np.array([0.3, -0.2, 0.4, -0.1])
    d_test = np.asarray(d_fn(*q_test), dtype=float)
    c_test = np.asarray(c_fn(*q_test, *qdot_test), dtype=float)
    d_dot_test = np.asarray(d_dot_fn(*q_test, *qdot_test), dtype=float)
    print(f"\nChecks at test configuration q = {q_test}:")
    print(f"D(q) symmetric: {np.allclose(d_test, d_test.T)}")
    print(f"minimum eigenvalue of D(q): {np.linalg.eigvalsh(d_test).min():.3e}")
    print(
        "Ddot - 2C skew-symmetric: "
        f"{np.allclose(d_dot_test - 2.0 * c_test, -(d_dot_test - 2.0 * c_test).T)}"
    )
    # Independent check: C qdot = Ddot qdot - grad_q (1/2 qdot^T D qdot), finite differences in q
    h = 1e-6
    basis = np.eye(4)
    kin = lambda qv: 0.5 * qdot_test @ np.asarray(d_fn(*qv), dtype=float) @ qdot_test
    grad = np.array([(kin(q_test + h * e) - kin(q_test - h * e)) / (2 * h) for e in basis])
    print(f"C qdot matches finite-difference reference: {np.allclose(c_test @ qdot_test, d_dot_test @ qdot_test - grad, atol=1e-9)}")
    print(f"g1(q) = {gravity[0]}")
    print(f"maximum |torque|: {np.abs(torques).max():.3e} N m")
    dynamic_part = torques - gravity_torques
    print(f"maximum |D qddot + C qdot| (inertia + Coriolis): {np.abs(dynamic_part).max():.3e} N m")
    print("\nNumerical result of")
    print("D(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau:")
    print("time [s]       tau_1        tau_2        tau_3        tau_4 [N m]")
    for knot in (0.0, 2.0, 4.0, 6.0, 8.0):
        index = int(round(knot / (times[-1] / (times.size - 1))))
        values = " ".join(f"{value: .6e}" for value in torques[index])
        print(f"{knot:7.1f}       {values}")

    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8))
    for joint, axis in enumerate(axes):
        axis.plot(times, torques[:, joint], label="total $\\tau$")
        axis.plot(times, gravity_torques[:, joint], "--", label="gravity $g(q)$")
        axis.set_ylabel(f"$\\tau_{joint + 1}$ [N m]")
        axis.grid(True)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("Problem 10: joint torques along the Problem 6 trajectory")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
