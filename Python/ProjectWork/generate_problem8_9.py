"""Reproduce the numerical results and vector figures for Problems 8 and 9.

The kinematic convention and trajectory construction are copied from Problems
3, 4, and 6 of the group report.  Running this file creates two vector PDFs in
the repository root and prints the values used in the report tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent

L2 = 93.0  # mm
L3 = 93.0  # mm
L4 = 50.0  # mm
D1 = 50.0  # mm
ELL = L2 + L3 + L4  # characteristic length used in Problem 8
RADIUS = 32.0  # mm
CIRCLE_CENTER = np.array([150.0, 0.0, 120.0])  # mm


def circle_position(phi: np.ndarray | float) -> np.ndarray:
    """Desired stylus-tip position from Problem 3."""
    phi = np.asarray(phi, dtype=float)
    return np.stack(
        [
            np.full_like(phi, CIRCLE_CENTER[0]),
            RADIUS * np.cos(phi),
            CIRCLE_CENTER[2] + RADIUS * np.sin(phi),
        ],
        axis=-1,
    )


def inverse_kinematics(phi: np.ndarray | float) -> np.ndarray:
    """Continuous elbow-up branch used in Problems 3--7."""
    p = circle_position(phi)
    x, y, z = np.moveaxis(p, -1, 0)
    q1 = np.arctan2(y, x)
    rw = np.hypot(x, y) - L4
    zw = z - D1
    d = (rw**2 + zw**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    d = np.clip(d, -1.0, 1.0)
    q3 = -np.arccos(d)
    q2 = (
        np.arctan2(zw, rw)
        - np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3))
        - np.pi / 2.0
    )
    q4 = -np.pi / 2.0 - q2 - q3
    return np.stack([q1, q2, q3, q4], axis=-1)


def jacobian(q: np.ndarray) -> np.ndarray:
    """Full 6-by-4 stylus geometric Jacobian from Problem 4, in mm/rad."""
    q1, q2, q3, q4 = np.asarray(q, dtype=float)
    c1, s1 = np.cos(q1), np.sin(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)
    beta = q2 + q3 + q4
    sb, cb = np.sin(beta), np.cos(beta)

    rho4 = -L2 * s2 - L3 * s23 - L4 * sb
    h4 = L2 * c2 + L3 * c23 + L4 * cb
    h43 = L3 * c23 + L4 * cb
    p43 = -L3 * s23 - L4 * sb

    return np.array(
        [
            [-s1 * rho4, -c1 * h4, -c1 * h43, -L4 * c1 * cb],
            [c1 * rho4, -s1 * h4, -s1 * h43, -L4 * s1 * cb],
            [0.0, rho4, p43, -L4 * sb],
            [0.0, s1, s1, s1],
            [0.0, -c1, -c1, -c1],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def forward_position(q: np.ndarray) -> np.ndarray:
    """Stylus-tip position from Problem 4."""
    q1, q2, q3, q4 = np.asarray(q, dtype=float)
    beta = q2 + q3 + q4
    rho4 = -L2 * np.sin(q2) - L3 * np.sin(q2 + q3) - L4 * np.sin(beta)
    h4 = L2 * np.cos(q2) + L3 * np.cos(q2 + q3) + L4 * np.cos(beta)
    return np.array([np.cos(q1) * rho4, np.sin(q1) * rho4, D1 + h4])


def task_jacobian(q: np.ndarray) -> np.ndarray:
    """Three translational rows plus the horizontal-stylus constraint."""
    beta = float(np.sum(q[1:]))
    horizontal_row = np.array([0.0, -np.sin(beta), -np.sin(beta), -np.sin(beta)])
    return np.vstack([jacobian(q)[:3, :], horizontal_row])


def quintic_coefficients(
    q0: np.ndarray,
    qf: np.ndarray,
    qd0: np.ndarray,
    qdf: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Return coefficients [a0,...,a5] for all four joints."""
    delta = qf - q0
    t = duration
    a0 = q0
    a1 = qd0
    a2 = np.zeros_like(q0)
    a3 = 10.0 * delta / t**3 - (6.0 * qd0 + 4.0 * qdf) / t**2
    a4 = -15.0 * delta / t**4 + (8.0 * qd0 + 7.0 * qdf) / t**3
    a5 = 6.0 * delta / t**5 - 3.0 * (qd0 + qdf) / t**4
    return np.stack([a0, a1, a2, a3, a4, a5], axis=-1)


def problem6_trajectory(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the four Problem 6 quintic segments."""
    knot_phi = np.linspace(0.0, 2.0 * np.pi, 5)
    knot_q = inverse_kinematics(knot_phi)
    knot_v = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -27.0, 0.0],
            [0.0, 0.0, -27.0],
            [0.0, 27.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    knot_qd = np.empty_like(knot_q)
    for i, (q, v) in enumerate(zip(knot_q, knot_v)):
        knot_qd[i] = np.linalg.solve(task_jacobian(q), np.r_[v, 0.0])

    duration = 2.0
    coeffs = np.stack(
        [
            quintic_coefficients(knot_q[i], knot_q[i + 1], knot_qd[i], knot_qd[i + 1], duration)
            for i in range(4)
        ]
    )

    q_times = np.empty((len(times), 4))
    for k, time in enumerate(times):
        segment = min(int(time // duration), 3)
        tau = time - segment * duration
        powers = tau ** np.arange(6)
        q_times[k] = coeffs[segment] @ powers
    return q_times, knot_qd, coeffs


def condition_data(q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.empty(len(q_values))
    scaled = np.empty(len(q_values))
    sigma_min = np.empty(len(q_values))
    for i, q in enumerate(q_values):
        j = jacobian(q)
        j_scaled = j.copy()
        j_scaled[:3, :] /= ELL
        raw[i] = np.linalg.cond(j)
        singular_values = np.linalg.svd(j_scaled, compute_uv=False)
        scaled[i] = singular_values[0] / singular_values[-1]
        sigma_min[i] = singular_values[-1]
    return raw, scaled, sigma_min


def load_torque(q: np.ndarray) -> np.ndarray:
    """Generalized torque due to a 1 N load in the negative z0 direction."""
    force = np.array([0.0, 0.0, -1.0])
    return jacobian(q)[:3, :].T @ force  # N mm


def _nice_ticks(vmin: float, vmax: float, count: int = 5) -> np.ndarray:
    if np.isclose(vmin, vmax):
        span = max(1.0, abs(vmin) * 0.1)
        vmin, vmax = vmin - span, vmax + span
    raw_step = (vmax - vmin) / max(count - 1, 1)
    power = 10.0 ** np.floor(np.log10(raw_step))
    fraction = raw_step / power
    nice_fraction = 1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 2.5 if fraction <= 2.5 else 5.0 if fraction <= 5.0 else 10.0
    step = nice_fraction * power
    start = np.floor(vmin / step) * step
    stop = np.ceil(vmax / step) * step
    return np.arange(start, stop + 0.5 * step, step)


def _draw_panel(
    c: canvas.Canvas,
    box: tuple[float, float, float, float],
    x: np.ndarray,
    curves: list[tuple[np.ndarray, colors.Color, float]],
    title: str,
    xlabel: str,
    ylabel: str,
    xticks: np.ndarray,
    xticklabels: list[str],
    yticks: np.ndarray | None = None,
    points: list[tuple[np.ndarray, np.ndarray, colors.Color]] | None = None,
) -> None:
    x0, y0, width, height = box
    left, right, bottom, top = 48.0, 14.0, 34.0, 28.0
    px0, py0 = x0 + left, y0 + bottom
    pw, ph = width - left - right, height - bottom - top

    all_y = np.concatenate([values for values, _, _ in curves])
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    if np.isclose(ymin, ymax):
        pad = max(1.0, abs(ymin) * 0.08)
    else:
        pad = 0.08 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad
    if yticks is None:
        yticks = _nice_ticks(ymin, ymax)
    ymin = min(ymin, float(yticks[0]))
    ymax = max(ymax, float(yticks[-1]))

    xmin, xmax = float(np.min(x)), float(np.max(x))

    def sx(value: np.ndarray | float) -> np.ndarray | float:
        return px0 + (np.asarray(value) - xmin) / (xmax - xmin) * pw

    def sy(value: np.ndarray | float) -> np.ndarray | float:
        return py0 + (np.asarray(value) - ymin) / (ymax - ymin) * ph

    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#17233B"))
    c.drawString(x0 + 3, y0 + height - 12, title)

    c.setLineWidth(0.35)
    c.setStrokeColor(colors.HexColor("#D7DCE5"))
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#3F4756"))
    for tick, label in zip(xticks, xticklabels):
        xpos = float(sx(tick))
        c.line(xpos, py0, xpos, py0 + ph)
        c.drawCentredString(xpos, py0 - 12, label)
    for tick in yticks:
        ypos = float(sy(tick))
        c.line(px0, ypos, px0 + pw, ypos)
        label = f"{tick:.0f}" if abs(tick) >= 10 else f"{tick:.1f}"
        c.drawRightString(px0 - 5, ypos - 2.5, label)

    c.setStrokeColor(colors.HexColor("#3F4756"))
    c.setLineWidth(0.75)
    c.rect(px0, py0, pw, ph, stroke=1, fill=0)

    for values, colour, line_width in curves:
        xs = np.asarray(sx(x), dtype=float)
        ys = np.asarray(sy(values), dtype=float)
        path = c.beginPath()
        path.moveTo(xs[0], ys[0])
        for xv, yv in zip(xs[1:], ys[1:]):
            path.lineTo(xv, yv)
        c.setStrokeColor(colour)
        c.setLineWidth(line_width)
        c.drawPath(path, stroke=1, fill=0)

    if points:
        for xp, yp, colour in points:
            c.setFillColor(colour)
            for xv, yv in zip(np.asarray(sx(xp)), np.asarray(sy(yp))):
                c.circle(float(xv), float(yv), 1.45, stroke=0, fill=1)

    c.setFillColor(colors.HexColor("#202634"))
    c.setFont("Helvetica", 8)
    c.drawCentredString(px0 + pw / 2.0, y0 + 5, xlabel)
    c.saveState()
    c.translate(x0 + 9, py0 + ph / 2.0)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()


def make_condition_figure(
    phi: np.ndarray,
    exact_raw: np.ndarray,
    exact_scaled: np.ndarray,
    time: np.ndarray,
    trajectory_raw: np.ndarray,
    trajectory_scaled: np.ndarray,
    sampled_phi: np.ndarray,
    sampled_raw: np.ndarray,
    sampled_scaled: np.ndarray,
) -> None:
    output = ROOT / "p8_condition_numbers.pdf"
    page_size = landscape((720.0, 460.0))
    c = canvas.Canvas(str(output), pagesize=page_size)
    width, height = page_size
    c.setTitle("Condition numbers along the exact and interpolated paths")
    c.setAuthor("Robotics 34753 Group 15")

    navy = colors.HexColor("#245C8A")
    orange = colors.HexColor("#E07A2D")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#17233B"))
    c.drawString(34, height - 24, "Conditioning of the stylus geometric Jacobian")
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#4E596B"))
    c.drawRightString(width - 28, height - 22, "Solid line: dense evaluation   Orange dots: 37 Problem 3 configurations")

    gap = 18.0
    margin_x = 25.0
    margin_bottom = 22.0
    top_space = 44.0
    panel_w = (width - 2 * margin_x - gap) / 2.0
    panel_h = (height - margin_bottom - top_space - gap) / 2.0
    left_x = margin_x
    right_x = margin_x + panel_w + gap
    bottom_y = margin_bottom
    top_y = margin_bottom + panel_h + gap

    phi_ticks = np.linspace(0.0, 2.0 * np.pi, 5)
    phi_labels = ["0", "pi/2", "pi", "3pi/2", "2pi"]
    time_ticks = np.linspace(0.0, 8.0, 5)
    time_labels = ["0", "2", "4", "6", "8"]

    _draw_panel(
        c,
        (left_x, top_y, panel_w, panel_h),
        phi,
        [(exact_raw, navy, 1.5)],
        "Exact circular path",
        "phi [rad]",
        "kappa2(J4), mm convention",
        phi_ticks,
        phi_labels,
        points=[(sampled_phi, sampled_raw, orange)],
    )
    _draw_panel(
        c,
        (right_x, top_y, panel_w, panel_h),
        time,
        [(trajectory_raw, navy, 1.5)],
        "Problem 6 quintic trajectory",
        "time [s]",
        "kappa2(J4), mm convention",
        time_ticks,
        time_labels,
    )
    _draw_panel(
        c,
        (left_x, bottom_y, panel_w, panel_h),
        phi,
        [(exact_scaled, navy, 1.5)],
        "Exact circular path",
        "phi [rad]",
        "kappa2(J4-hat), ell = 236 mm",
        phi_ticks,
        phi_labels,
        points=[(sampled_phi, sampled_scaled, orange)],
    )
    _draw_panel(
        c,
        (right_x, bottom_y, panel_w, panel_h),
        time,
        [(trajectory_scaled, navy, 1.5)],
        "Problem 6 quintic trajectory",
        "time [s]",
        "kappa2(J4-hat), ell = 236 mm",
        time_ticks,
        time_labels,
    )
    c.showPage()
    c.save()


def make_torque_figure(phi: np.ndarray, torques: np.ndarray, sampled_phi: np.ndarray, sampled_torque: np.ndarray) -> None:
    output = ROOT / "p9_joint_torques.pdf"
    page_size = landscape((720.0, 390.0))
    c = canvas.Canvas(str(output), pagesize=page_size)
    width, height = page_size
    c.setTitle("Joint torques caused by a 1 N vertical load")
    c.setAuthor("Robotics 34753 Group 15")

    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#17233B"))
    c.drawString(34, height - 26, "Joint torques caused by a 1 N load along -z0")
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#4E596B"))
    c.drawRightString(width - 28, height - 24, "Lines: dense evaluation   Markers: 37 Problem 3 configurations")

    palette = [
        colors.HexColor("#245C8A"),
        colors.HexColor("#C94C4C"),
        colors.HexColor("#3B8D6E"),
        colors.HexColor("#8B63A8"),
    ]
    curves = [(torques[:, i], palette[i], 1.7) for i in range(4)]
    point_sets = [(sampled_phi, sampled_torque[:, i], palette[i]) for i in range(4)]
    phi_ticks = np.linspace(0.0, 2.0 * np.pi, 5)
    phi_labels = ["0", "pi/2", "pi", "3pi/2", "2pi"]

    _draw_panel(
        c,
        (28.0, 28.0, width - 56.0, height - 76.0),
        phi,
        curves,
        "Load-induced generalized torque, tau_load = Jv,4^T F_load",
        "phi [rad]",
        "joint torque [N mm]",
        phi_ticks,
        phi_labels,
        yticks=np.arange(-175.0, 26.0, 25.0),
        points=point_sets,
    )

    legend_y = height - 47.0
    legend_x = 155.0
    c.setFont("Helvetica", 8.5)
    for i, colour in enumerate(palette):
        c.setStrokeColor(colour)
        c.setLineWidth(2.2)
        c.line(legend_x, legend_y, legend_x + 22, legend_y)
        c.setFillColor(colors.HexColor("#202634"))
        label = f"tau{i + 1}"
        c.drawString(legend_x + 27, legend_y - 3, label)
        legend_x += 78
    c.showPage()
    c.save()


def main() -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 7201)
    q_exact = inverse_kinematics(phi)
    exact_raw, exact_scaled, exact_sigma_min = condition_data(q_exact)

    sampled_phi = np.linspace(0.0, 2.0 * np.pi, 37)
    q_sampled = inverse_kinematics(sampled_phi)
    sampled_raw, sampled_scaled, sampled_sigma_min = condition_data(q_sampled)

    time = np.linspace(0.0, 8.0, 8001)
    q_trajectory, knot_qd, coeffs = problem6_trajectory(time)
    trajectory_raw, trajectory_scaled, trajectory_sigma_min = condition_data(q_trajectory)

    # Directly verify that the reconstructed trajectory matches Problem 7.
    p_trajectory = np.array([forward_position(q) for q in q_trajectory])
    distance = np.sqrt(
        (p_trajectory[:, 0] - 150.0) ** 2
        + (np.hypot(p_trajectory[:, 1], p_trajectory[:, 2] - 120.0) - 32.0) ** 2
    )

    torques = np.array([load_torque(q) for q in q_exact])
    sampled_torques = np.array([load_torque(q) for q in q_sampled])

    make_condition_figure(
        phi,
        exact_raw,
        exact_scaled,
        time,
        trajectory_raw,
        trajectory_scaled,
        sampled_phi,
        sampled_raw,
        sampled_scaled,
    )
    make_torque_figure(phi, torques, sampled_phi, sampled_torques)

    print("Problem 6 knot joint velocities [rad/s]:")
    print(np.array2string(knot_qd, precision=9, suppress_small=True))
    print(f"Problem 7 reconstructed maximum deviation [mm]: {distance.max():.9f}")
    print()
    print("Problem 8 ranges:")
    print(f"  exact raw condition number:       {exact_raw.min():.9f} .. {exact_raw.max():.9f}")
    print(f"  exact scaled condition number:    {exact_scaled.min():.9f} .. {exact_scaled.max():.9f}")
    print(f"  exact min sigma_4 (scaled):       {exact_sigma_min.min():.9f}")
    print(f"  37-point raw condition number:    {sampled_raw.min():.9f} .. {sampled_raw.max():.9f}")
    print(f"  37-point scaled condition number: {sampled_scaled.min():.9f} .. {sampled_scaled.max():.9f}")
    print(f"  37-point min sigma_4 (scaled):    {sampled_sigma_min.min():.9f}")
    print(f"  trajectory raw condition number:  {trajectory_raw.min():.9f} .. {trajectory_raw.max():.9f}")
    print(f"  trajectory scaled condition no.:  {trajectory_scaled.min():.9f} .. {trajectory_scaled.max():.9f}")
    print(f"  trajectory min sigma_4 (scaled):  {trajectory_sigma_min.min():.9f}")
    print(f"  exact q3 range [deg]:              {np.rad2deg(q_exact[:, 2].min()):.9f} .. {np.rad2deg(q_exact[:, 2].max()):.9f}")
    print(f"  trajectory q3 range [deg]:         {np.rad2deg(q_trajectory[:, 2].min()):.9f} .. {np.rad2deg(q_trajectory[:, 2].max()):.9f}")
    print()
    print("Problem 9 cardinal-point torques [N mm]:")
    cardinal_phi = np.linspace(0.0, 2.0 * np.pi, 5)
    cardinal_torques = np.array([load_torque(q) for q in inverse_kinematics(cardinal_phi)])
    for angle, values in zip(["0", "pi/2", "pi", "3pi/2", "2pi"], cardinal_torques):
        print(f"  {angle:>5s}: " + " ".join(f"{v: .9f}" for v in values))
    print("Problem 9 torque ranges [N mm]:")
    for i in range(4):
        print(f"  tau_{i + 1}: {torques[:, i].min():.9f} .. {torques[:, i].max():.9f}")
    print()
    print("Generated:")
    print(ROOT / "p8_condition_numbers.pdf")
    print(ROOT / "p9_joint_torques.pdf")


if __name__ == "__main__":
    main()
