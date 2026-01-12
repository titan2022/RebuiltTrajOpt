"""
FRC 2022 shooter trajectory optimization.

This program uses the Sleipnir NLP solver to find the initial pitch and yaw for a game
piece to hit the 2026 FRC game's target given an initial velocity.

This optimization problem formulation uses direct transcription of the flight dynamics, including
air resistance.

Based on the 2022 trajectory optimization example code by Tyler Veness.
https://github.com/SleipnirGroup/Sleipnir/blob/main/examples/frc_2022_shooter/main.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sleipnir.autodiff import VariableMatrix, hypot
from sleipnir.optimization import ExitStatus, Problem

# Field dimensions
field_width = 8.043  # m
field_length = 16.518  # m
target_wrt_field = np.array(
    [
        [0],
        [0],
        [72 * 0.0254],
        [0.0],
        [0.0],
        [0.0],
    ]
)
# Physical characteristics
shooter_wrt_robot = np.array([[0.0], [0.0], [20 * 0.0254], [0.0], [0.0], [0.0]])
g = 9.81  # m/s²
max_shooter_velocity = 15  # m/s
ball_mass = 0.5 / 2.205  # kg
ball_diameter = 5.91 * 0.0254  # m


def lerp(a, b, t):
    return a + t * (b - a)


def f(x):
    """
    Apply the drag equation to a velocity.
    """
    # x' = x'
    # y' = y'
    # z' = z'
    # x" = −a_D(v_x)
    # y" = −a_D(v_y)
    # z" = −g − a_D(v_z)
    #
    # where a_D(v) = ½ρv² C_D A / m
    # (see https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation)
    rho = 1.204  # kg/m³
    C_D = 0.5
    m = ball_mass
    A = math.pi * ((ball_diameter / 2) ** 2)
    a_D = lambda v: 0.5 * rho * v**2 * C_D * A / m

    v_x = x[3, 0]
    v_y = x[4, 0]
    v_z = x[5, 0]
    return VariableMatrix(
        [[v_x], [v_y], [v_z], [-a_D(v_x)], [-a_D(v_y)], [-g - a_D(v_z)]]
    )


N = 20


def setup_problem(distance, robot_vx, robot_vy):
    """
    Set up the problem for the solver.
    """
    # Robot initial state
    robot_wrt_field = np.array(
        [[-distance], [0], [0.0], [robot_vx], [robot_vy], [0.0]]
    )

    shooter_wrt_field = robot_wrt_field + shooter_wrt_robot

    problem = Problem()

    # Set up duration decision variables
    T = problem.decision_variable()
    problem.subject_to(T >= 0)
    T.set_value(1)
    dt = T / N

    # Ball state in field frame
    #
    #     [x position]
    #     [y position]
    #     [z position]
    # x = [x velocity]
    #     [y velocity]
    #     [z velocity]
    X = problem.decision_variable(6, N)

    p = X[:3, :]

    v_x = X[3, :]
    v_y = X[4, :]
    v_z = X[5, :]

    v0_wrt_shooter = X[3:, :1] - shooter_wrt_field[3:, :]

    # Shooter initial position
    problem.subject_to(p[:, :1] == shooter_wrt_field[:3, :])

    # Dynamics constraints - RK4 integration
    h = dt
    for k in range(N - 1):
        x_k = X[:, k]
        x_k1 = X[:, k + 1]

        k1 = f(x_k)
        k2 = f(x_k + h / 2 * k1)
        k3 = f(x_k + h / 2 * k2)
        k4 = f(x_k + h * k3)
        problem.subject_to(x_k1 == x_k + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    # Require final position is in center of target circle
    problem.subject_to(p[:, -1] == target_wrt_field[:3, :])

    # Require the final velocity is at least somewhat downwards by limiting horizontal velocity
    # and requiring negative vertical velocity
    problem.subject_to(hypot(v_x[-1], v_y[-1]) < 4)
    problem.subject_to(v_z[-1] < 0)

    return problem, shooter_wrt_field, v0_wrt_shooter, X


def min_velocity(distance, robot_vx, robot_vy):
    """
    Solve for minimum velocity.
    :returns: A tuple of [True, velocity, pitch, yaw, X] if it succeeds at a solve, and a tuple of[False, 0] if it fails.
    """
    setup = setup_problem(distance, robot_vx, robot_vy)
    problem = setup[0]
    shooter_wrt_field = setup[1]
    v0_wrt_shooter = setup[2]
    X = setup[3]

    p_x = X[0, :]
    p_y = X[1, :]
    p_z = X[2, :]

    v = X[3:, :]
    v_x = X[3, :]
    v_y = X[4, :]
    v_z = X[5, :]

    # Position initial guess is linear interpolation between start and end position
    for k in range(N):
        p_x[k].set_value(lerp(shooter_wrt_field[0, 0], target_wrt_field[0, 0], k / N))
        p_y[k].set_value(lerp(shooter_wrt_field[1, 0], target_wrt_field[1, 0], k / N))
        p_z[k].set_value(lerp(shooter_wrt_field[2, 0], target_wrt_field[2, 0], k / N))

    # Velocity initial guess is max initial velocity toward target
    uvec_shooter_to_target = target_wrt_field[:3, :] - shooter_wrt_field[:3, :]
    uvec_shooter_to_target /= norm(uvec_shooter_to_target)
    for k in range(N):
        v[:, k].set_value(
            shooter_wrt_field[3:, :] + max_shooter_velocity * uvec_shooter_to_target
        )

    # Require initial velocity is less than max shooter velocity
    #
    #   √(v_x² + v_y² + v_z²) = v
    #   v_x² + v_y² + v_z² = v²
    problem.subject_to(
        (v_x[0] - shooter_wrt_field[3, 0]) ** 2
        + (v_y[0] - shooter_wrt_field[4, 0]) ** 2
        + (v_z[0] - shooter_wrt_field[5, 0]) ** 2
        <= max_shooter_velocity**2
    )

    # Minimize initial velocity
    problem.minimize(v0_wrt_shooter.T @ v0_wrt_shooter)

    if problem.solve(tolerance=0.01) == ExitStatus.SUCCESS:
        print("Minimum velocity solve:")
        # Initial velocity vector with respect to shooter
        v0 = v0_wrt_shooter.value()

        velocity = norm(v0)
        print(f"Velocity = {velocity:.03f} m/s")

        pitch = math.atan2(v0[2, 0], math.hypot(v0[0, 0], v0[1, 0]))
        print(f"Pitch = {np.rad2deg(pitch):.03f}°")

        yaw = math.atan2(v0[1, 0], v0[0, 0])
        print(f"Yaw = {np.rad2deg(yaw):.03f}°")

        return True, velocity, pitch, yaw, X
    print("Infeasible at this location")
    return False, 0


def fixed_velocity(distance, robot_vx, robot_vy, target_vel, prev_X):
    """
    Solve for a fixed velocity.
    :param prev_X: The previous solve's state vectors, to act as an initial guess.
    :returns: A tuple of [True, velocity, pitch, yaw, X] if it succeeds at a solve, and a tuple of[False, 0] if it fails.
    """
    setup = setup_problem(distance, robot_vx, robot_vy)
    problem = setup[0]
    shooter_wrt_field = setup[1]
    v0_wrt_shooter = setup[2]
    X = setup[3]

    prev_p_x = prev_X[0, :]
    prev_p_y = prev_X[1, :]
    prev_p_z = prev_X[2, :]

    prev_v = prev_X[3:, :]

    p_x = X[0, :]
    p_y = X[1, :]
    p_z = X[2, :]

    v = X[3:, :]
    v_x = X[3, :]
    v_y = X[4, :]
    v_z = X[5, :]

    # Position initial guess is last solve's position
    for k in range(N):
        p_x[k].set_value(prev_p_x[k].value())
        p_y[k].set_value(prev_p_y[k].value())
        p_z[k].set_value(prev_p_z[k].value())

    # Velocity initial guess is last solve's velocity
    for k in range(N):
        v[:, k].set_value(prev_v[:, k].value())

    # Require initial velocity is equal to target
    #
    #   √(v_x² + v_y² + v_z²) = v
    #   v_x² + v_y² + v_z² = v²
    problem.subject_to(
        (v_x[0] - shooter_wrt_field[3, 0]) ** 2
        + (v_y[0] - shooter_wrt_field[4, 0]) ** 2
        + (v_z[0] - shooter_wrt_field[5, 0]) ** 2
        == target_vel**2
    )

    if problem.solve(tolerance=0.01) == ExitStatus.SUCCESS:
        print("Fixed velocity solve:")
        # Initial velocity vector with respect to shooter
        v0 = v0_wrt_shooter.value()

        velocity = norm(v0)
        print(f"Velocity = {velocity:.03f} m/s")

        pitch = math.atan2(v0[2, 0], math.hypot(v0[0, 0], v0[1, 0]))
        print(f"Pitch = {np.rad2deg(pitch):.03f}°")

        yaw = math.atan2(v0[1, 0], v0[0, 0])
        print(f"Yaw = {np.rad2deg(yaw):.03f}°")

        return True, velocity, pitch, yaw, X
    print(f"Infeasible at this location with velocity {target_vel:.03f} m/s")
    return False, 0


if __name__ == "__main__":
    start_distance = 1
    end_distance = 5

    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xlim(start_distance, end_distance)
    ax.set_xlabel("Distance (m)")
    ax.set_ylim(5,max_shooter_velocity)
    ax.set_ylabel("Shooter velocity (m/s)")
    ax.set_zlim(45, 90)
    ax.set_zlabel("Hood angle (deg)")

    samples = 20
    for i in range(0, samples + 1):
        distance = lerp(start_distance, end_distance, i / samples)
        vx = 0
        vy = 1

        velocities = []
        angles = []

        # Solve for minimum velocity
        min_vel_solve = min_velocity(distance, vx, vy)
        # If the position is possible,
        if min_vel_solve[0]:
            velocities.append(min_vel_solve[1])
            angles.append(np.rad2deg(min_vel_solve[2]))
            d_V = 0.5
            vel = min_vel_solve[1] + d_V
            prev_solve = min_vel_solve
            while vel < max_shooter_velocity:
                # Feed previous solve into new solve
                prev_solve = fixed_velocity(distance, vx, vy, vel, prev_solve[4])
                if not prev_solve[0]:
                    break
                velocities.append(vel)
                angles.append(np.rad2deg(prev_solve[2]))
                vel += d_V
            if prev_solve[0]:
                prev_solve = fixed_velocity(
                    distance, vx, vy, max_shooter_velocity, prev_solve[4]
                )
                velocities.append(max_shooter_velocity)
                angles.append(np.rad2deg(prev_solve[2]))
        ax.scatter(distance, velocities, angles)
    plt.show()
