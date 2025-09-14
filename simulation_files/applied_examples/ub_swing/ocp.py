from __future__ import annotations
from typing import Tuple
import numpy as np
from bioptim import (
    Node, ConstraintList, InterpolationType, ConstraintFcn, ObjectiveList,
    OptimalControlProgram, BoundsList, InitialGuessList, ObjectiveFcn,
    OdeSolver, OdeSolverBase, ControlType, PhaseDynamics, Axis,
    DynamicsOptionsList, DynamicsOptions, ObjectiveWeight, DefectType,
)
from .config import PHASE_COUNT, IDX, S_JOINTS, ROTATIONS, ROT_END, BarParams
from .dynamics import DynamicModel
from .helpers import add_common_objectives, add_symmetry_constraints, make_x_bounds

def prepare_ocp(
    biorbd_model_path: str,
    final_time: Tuple[float, ...],
    n_shooting: Tuple[int, ...],
    min_time: float,
    max_time: float,
    total_mass: float,
    init_sol: bool,
    final_state_bound: bool,
    coef_fig: float,
    weight_control: float,
    weight_time: float = 1.0,
    mode: str = "",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
    x_init: InitialGuessList | None = None,
    u_init: InitialGuessList | None = None,
) -> OptimalControlProgram:
    params = BarParams()
    bio_model = tuple(DynamicModel(biorbd_model_path, params) for _ in range(PHASE_COUNT))

    n_tau = bio_model[0].nb_tau
    n_q   = bio_model[0].nb_q

    # Objectives
    objectives = ObjectiveList()
    add_common_objectives(objectives, weight_control, weight_time, min_time, max_time, coef_fig, init_sol)

    if not init_sol:
        leg_weight = np.hstack((3*np.ones(26), np.zeros(25)))
        objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=IDX["RxThighR"], node=Node.ALL,
                       weight=ObjectiveWeight(leg_weight, interpolation=InterpolationType.EACH_FRAME))
        objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=IDX["RxThighR"],
                       target=0, weight=3*coef_fig, phase=2)

    # Dynamics
    dynamics = DynamicsOptionsList()
    for _ in range(PHASE_COUNT):
        dynamics.add(DynamicsOptions(
            expand_dynamics=expand_dynamics,
            phase_dynamics=phase_dynamics,
            ode_solver=OdeSolver.COLLOCATION(method="radau", polynomial_degree=5,
                                             defects_type=DefectType.TAU_EQUALS_INVERSE_DYNAMICS),
        ))

    # Constraints
    constraints = ConstraintList()
    if not init_sol:
        # avoid lower bar
        constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL,
                        first_marker="LowerBarMarker", second_marker="MarkerR",
                        min_bound=0.02, max_bound=np.inf, axes=Axis.X, phase=1)
        # pelvic orientation during descent
        if mode == "anteversion":
            constraints.add(ConstraintFcn.TRACK_MARKERS, phase=0, node=Node.ALL,
                            reference_jcs=IDX["Back"], marker_index=3, axes=Axis.X,
                            min_bound=-np.inf, max_bound=0)
        elif mode == "retroversion":
            constraints.add(ConstraintFcn.TRACK_MARKERS, phase=0, node=Node.ALL,
                            reference_jcs=IDX["Back"], marker_index=3, axes=Axis.X,
                            min_bound=0, max_bound=np.inf)

    # end phase 0: feet reach lower bar height
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END,
                    first_marker="LowerBarMarker", second_marker="MarkerR",
                    axes=Axis.Z, phase=0, min_bound=0.02, max_bound=0.02)

    # symmetry
    add_symmetry_constraints(constraints)

    # Bounds
    x_bounds = make_x_bounds(bio_model, total_mass, stiffness=params.stiffness)

    if final_state_bound:
        x_bounds[2]["q"][S_JOINTS, -1] = 0
        x_bounds[2]["q"][IDX["RyHands"], -1]   = ROT_END
        x_bounds[2]["qdot"][IDX["RyHands"], -1] = -np.pi
    else:
        constraints.add(ConstraintFcn.BOUND_STATE, key="q", phase=2, node=Node.END,
                        index=IDX["RyHands"], min_bound=ROT_END, max_bound=ROT_END)
        constraints.add(ConstraintFcn.BOUND_STATE, key="q", phase=2, node=Node.END,
                        index=S_JOINTS, min_bound=0, max_bound=0)
        constraints.add(ConstraintFcn.BOUND_STATE, key="qdot", phase=2, node=Node.END,
                        index=IDX["RyHands"], min_bound=-np.pi, max_bound=-np.pi)

    # Control bounds
    tau_min, tau_max = (-200, 200)
    u_min = [0] * 3 + [tau_min] * (n_tau - 3)
    u_max = [0] * 3 + [tau_max] * (n_tau - 3)

    if not init_sol:
        u_min[IDX["Shoulder"]] = -3.11 * total_mass
        u_max[IDX["Shoulder"]] =  2.15 * total_mass
        u_min[IDX["RyThighR"]] = -4.20 * total_mass
        u_max[IDX["RyThighR"]] =  9.36 * total_mass
        u_min[IDX["RyThighL"]] = -4.20 * total_mass
        u_max[IDX["RyThighL"]] =  9.36 * total_mass

    u_bounds = BoundsList()
    for p in range(PHASE_COUNT):
        u_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=p)

    # Initial guesses (if not provided)
    if x_init is None:
        x_init = InitialGuessList()
        for p in range(PHASE_COUNT):
            init_q = np.zeros((n_q, 2))
            init_q[IDX["RyHands"], :] = [ROTATIONS[p], ROTATIONS[p + 1]]
            x_init.add("q", init_q, phase=p, interpolation=InterpolationType.LINEAR)
            x_init.add("qdot", [0] * n_q, phase=p)

    if u_init is None:
        u_init = InitialGuessList()
        for p in range(PHASE_COUNT):
            u_init.add("tau", [0] * n_tau, phase=p)

    return OptimalControlProgram(
        bio_model, n_shooting, final_time, dynamics=dynamics,
        x_init=x_init, u_init=u_init, x_bounds=x_bounds, u_bounds=u_bounds,
        objective_functions=objectives, use_sx=use_sx, n_threads=n_threads,
        control_type=control_type, constraints=constraints,
    )
