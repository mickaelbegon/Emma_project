from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from casadi import MX, vertcat

from bioptim import (
    Node,
    ConstraintList,
    InterpolationType,
    ConstraintFcn,
    ObjectiveList,
    DynamicsFunctions,
    OptimalControlProgram,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    NonLinearProgram,
    ControlType,
    PhaseDynamics,
    DynamicsEvaluation,
    SolutionMerge,
    Axis,
    DynamicsOptionsList,
    DynamicsOptions,
    TorqueBiorbdModel,
    ObjectiveWeight,
)


# ----------------------------
# Configuration & constants
# ----------------------------
PHASE_COUNT = 3

NAMES = [
    "TxHands",
    "TzHands",
    "RyHands",
    "Elbow",
    "Shoulder",
    "Back",
    "LowBack",
    "RxThighR",
    "RyThighR",
    "KneeR",
    "FootR",
    "RxThighL",
    "RyThighL",
    "KneeL",
    "FootL",
]
IDX = {name: int(i) for i, name in enumerate(NAMES)}
S_JOINTS = slice(IDX["RyHands"] + 1, IDX["FootL"] + 1)  # used for final state zeroing

# Hand rotation waypoints
ROT_START = -2 * np.pi / 45  # ~ -8°
ROT_MID_1 = -np.pi / 3
ROT_MID_2 = -np.pi
ROT_END = -2 * np.pi
ROTATIONS = [ROT_START, ROT_MID_1, ROT_MID_2, ROT_END]


@dataclass(frozen=True)
class BarParams:
    stiffness: float = 14160.0
    damping: float = 91.0


# ----------------------------
# Dynamic model
# ----------------------------
class DynamicModel(TorqueBiorbdModel):
    def __init__(self, biorbd_model_path: str, params: BarParams):
        super().__init__(biorbd_model_path)
        self.params = params

    def dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """Spring-damper bar on TxHands (x) and TzHands (z)."""
        k, c = self.params.stiffness, self.params.damping

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        # Replace the first two joint torques with spring-damper forces
        tau[0] = -k * q[0] + c * qdot[0]  # x
        tau[1] = -k * q[1] + c * qdot[1]  # z

        qddot = nlp.model.forward_dynamics(with_contact=False)(q, qdot, tau, [], [])

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            defects = vertcat(slope_q, slope_qdot) * nlp.dt - vertcat(qdot, qddot) * nlp.dt

        return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=defects)


# ----------------------------
# Helpers
# ----------------------------

def split_indices(n_shooting: Iterable[int]):
    ns = np.array(tuple(n_shooting), dtype=int)
    s_ctrl = np.concatenate(([0], np.cumsum(ns)))  # ni
    s_state = np.concatenate(([0], np.cumsum(ns + 1)))  # ni+1
    return s_state, s_ctrl


def build_initial_guesses(qs: np.ndarray, qdots: np.ndarray, taus: np.ndarray, n_shooting: Iterable[int]):
    s_state, s_ctrl = split_indices(n_shooting)
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    for p, (sa, sb, ua, ub) in enumerate(zip(s_state[:-1], s_state[1:], s_ctrl[:-1], s_ctrl[1:])):
        x_init.add("q", qs[:, sa:sb], InterpolationType.EACH_FRAME, phase=p)
        x_init.add("qdot", qdots[:, sa:sb], InterpolationType.EACH_FRAME, phase=p)
        u_init.add("tau", taus[:, ua:ub], InterpolationType.EACH_FRAME, phase=p)
    return x_init, u_init


def add_common_objectives(
    objectives: ObjectiveList,
    weight_control: float,
    weight_time: float,
    min_time: float,
    max_time: float,
    coef_fig: float,
    init_sol: bool,
    phase_count: int = PHASE_COUNT,
):
    for p in range(phase_count):
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            quadratic=True,
            weight=weight_control,
            phase=p,
        )
        objectives.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            weight=weight_time,
            min_bound=min_time,
            max_bound=max_time,
            phase=p,
        )
    if not init_sol:
        for p in range(phase_count):
            objectives.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                key="qdot",
                weight=1.0,
                derivative=True,
                phase=p,
            )
            for name, w in {"Elbow": 5, "KneeR": 10, "FootR": 2}.items():
                objectives.add(
                    ObjectiveFcn.Lagrange.TRACK_STATE,
                    key="q",
                    index=IDX[name],
                    target=0,
                    weight=w * coef_fig,
                    phase=p,
                )


def add_symmetry_constraints(constraints: ConstraintList, phase_count: int = PHASE_COUNT):
    pairs = [
        ("RxThighR", "RxThighL", -1),
        ("RyThighR", "RyThighL", 1),
        ("KneeR", "KneeL", 1),
        ("FootR", "FootL", 1),
    ]
    for p in range(phase_count):
        for a, b, c in pairs:
            constraints.add(
                ConstraintFcn.PROPORTIONAL_STATE,
                key="q",
                phase=p,
                node=Node.ALL,
                first_dof=IDX[a],
                second_dof=IDX[b],
                coef=c,
            )


def make_x_bounds(models, total_mass: float) -> BoundsList:
    xb = BoundsList()
    for p in range(PHASE_COUNT):
        xb.add("q", bounds=models[0].bounds_from_ranges("q"), phase=p)
        xb.add("qdot", bounds=models[0].bounds_from_ranges("qdot"), phase=p)

    # Initial node conditions
    xb[0]["q"][:, 0] = 0
    xb[0]["q"][IDX["RyHands"], 0] = ROT_START
    xb[0]["q"][1, 0] = -total_mass * 9.81 / BarParams().stiffness  # equilibrium position on z (index 1)
    xb[0]["qdot"][:, 0] = 0

    # End of phase 1: hands under the upper bar
    xb[1]["q"][IDX["RyHands"], -1] = -np.pi
    return xb


def stack_states(sol) -> Tuple[np.ndarray, np.ndarray]:
    parts = sol.decision_states(to_merge=[SolutionMerge.NODES])
    q = np.hstack([p["q"] for p in parts])
    qd = np.hstack([p["qdot"] for p in parts])
    return q, qd


def stack_controls(sol) -> np.ndarray:
    parts = sol.decision_controls(to_merge=[SolutionMerge.NODES])
    return np.hstack([p["tau"] for p in parts])


def step_time(sol) -> np.ndarray:
    return sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]).T[0]


def save_arrays(path: Path, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, time: np.ndarray):
    with open(path, "wb") as f:
        pickle.dump({"q": q, "qdot": qdot, "tau": tau, "time": time}, f)


# ----------------------------
# OCP builder
# ----------------------------

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
    n_q = bio_model[0].nb_q

    # Objectives
    objectives = ObjectiveList()
    add_common_objectives(objectives, weight_control, weight_time, min_time, max_time, coef_fig, init_sol)

    if not init_sol:
        leg_weight = np.hstack((3 * np.ones(26), np.zeros(25)))
        objectives.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="q",
            index=IDX["RxThighR"],
            node=Node.ALL,
            weight=ObjectiveWeight(leg_weight, interpolation=InterpolationType.EACH_FRAME),
        )
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            key="q",
            index=IDX["RxThighR"],
            target=0,
            weight=3 * coef_fig,
            phase=2,
        )

    # Dynamics
    dynamics = DynamicsOptionsList()
    for _ in range(PHASE_COUNT):
        dynamics.add(
            DynamicsOptions(
                expand_dynamics=expand_dynamics,
                phase_dynamics=phase_dynamics,
                ode_solver=OdeSolver.COLLOCATION(method="radau", polynomial_degree=5),
            )
        )

    # Constraints
    constraints = ConstraintList()
    if not init_sol:
        # Avoid the lower bar
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            node=Node.ALL,
            first_marker="LowerBarMarker",
            second_marker="MarkerR",
            min_bound=0.02,
            max_bound=np.inf,
            axes=Axis.X,
            phase=1,
        )

        # Pelvic orientation during descent (phase 0)
        if mode == "anteversion":
            constraints.add(
                ConstraintFcn.TRACK_MARKERS,
                phase=0,
                node=Node.ALL,
                reference_jcs=IDX["Back"],
                marker_index=3,
                axes=Axis.X,
                min_bound=-np.inf,
                max_bound=0,
            )
        elif mode == "retroversion":
            constraints.add(
                ConstraintFcn.TRACK_MARKERS,
                phase=0,
                node=Node.ALL,
                reference_jcs=IDX["Back"],
                marker_index=3,
                axes=Axis.X,
                min_bound=0,
                max_bound=np.inf,
            )

    # End phase 0 condition: feet reach lower bar height
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="LowerBarMarker",
        second_marker="MarkerR",
        axes=Axis.Z,
        phase=0,
        min_bound=0.02,
        max_bound=0.02,
    )

    # Symmetry
    add_symmetry_constraints(constraints)

    # Bounds
    x_bounds = make_x_bounds(bio_model, total_mass)

    if final_state_bound:
        x_bounds[2]["q"][S_JOINTS, -1] = 0
        x_bounds[2]["q"][IDX["RyHands"], -1] = ROT_END
        x_bounds[2]["qdot"][IDX["RyHands"], -1] = -np.pi
    else:
        constraints.add(
            ConstraintFcn.BOUND_STATE,
            key="q",
            phase=2,
            node=Node.END,
            index=IDX["RyHands"],
            min_bound=ROT_END,
            max_bound=ROT_END,
        )
        constraints.add(
            ConstraintFcn.BOUND_STATE,
            key="q",
            phase=2,
            node=Node.END,
            index=S_JOINTS,
            min_bound=0,
            max_bound=0,
        )
        constraints.add(
            ConstraintFcn.BOUND_STATE,
            key="qdot",
            phase=2,
            node=Node.END,
            index=IDX["RyHands"],
            min_bound=-np.pi,
            max_bound=-np.pi,
        )

    # Control bounds
    tau_min, tau_max = (-200, 200)
    u_min = [0] * 3 + [tau_min] * (n_tau - 3)
    u_max = [0] * 3 + [tau_max] * (n_tau - 3)

    if not init_sol:
        # Scale some DOFs with athlete mass
        u_min[IDX["Shoulder"]] = -3.11 * total_mass
        u_max[IDX["Shoulder"]] = 2.15 * total_mass
        u_min[IDX["RyThighR"]] = -4.20 * total_mass
        u_max[IDX["RyThighR"]] = 9.36 * total_mass
        u_min[IDX["RyThighL"]] = -4.20 * total_mass
        u_max[IDX["RyThighL"]] = 9.36 * total_mass

    u_bounds = BoundsList()
    for p in range(PHASE_COUNT):
        u_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=p)

    # Initial guesses (if not provided)
    if x_init is None:
        x_init = InitialGuessList()
        for p in range(PHASE_COUNT):
            init_q = np.zeros((n_q, 2))
            init_q[IDX["RyHands"], :] = [ROTATIONS[p], ROTATIONS[p + 1]]  # shape (1, 2)
            x_init.add("q", init_q, phase=p, interpolation=InterpolationType.LINEAR)
            x_init.add("qdot", [0] * n_q, phase=p)

    if u_init is None:
        u_init = InitialGuessList()
        for p in range(PHASE_COUNT):
            u_init.add("tau", [0] * n_tau, phase=p)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
        constraints=constraints,
    )


# ----------------------------
# Entry point
# ----------------------------

def main():
    ROOT = Path(__file__).resolve().parent
    RESULTS_DIR = ROOT / "applied_examples" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    use_pkl = True
    n_shooting = (50, 50, 50)

    for num in [501]:  # or range(576)
        model_path = (ROOT / f"applied_examples/athlete_{num}_deleva.bioMod").as_posix()
        print("model:", filename)

        masses = pd.read_csv(ROOT / "applied_examples" / "masses.csv")
        total_mass = float(masses["total_mass"][num - 1])

        base_pkl = RESULTS_DIR / f"athlete{num}_base.pkl"

        # ------------- Initial solution -------------
        if (not use_pkl) or (not base_pkl.exists()):
            ocp = prepare_ocp(
                biorbd_model_path=model_path,
                final_time=(1, 0.5, 1),
                n_shooting=n_shooting,
                min_time=0.2,
                max_time=2.0,
                coef_fig=1.0,
                total_mass=total_mass,
                init_sol=True,
                weight_control=1.0,
                weight_time=0.1,
                final_state_bound=True,
                n_threads=16,
                use_sx=False,
            )

            ocp.add_plot_penalty(CostType.ALL)
            ocp.print(to_console=False, to_graph=False)

            solver = Solver.IPOPT(show_online_optim=False)
            solver.set_linear_solver("ma57")
            solver.set_bound_frac(1e-8)
            solver.set_bound_push(1e-8)
            solver.set_maximum_iterations(20000)

            print("start solving (initial)")
            sol = ocp.solve(solver)
            print("solving finished (initial)")

            q, qd = stack_states(sol)
            tau = stack_controls(sol)
            t = step_time(sol)
            save_arrays(base_pkl, q, qd, tau, t)
            print("initial solution saved")

        # ------------- Load initial solution -------------
        print("load initial solution")
        with open(base_pkl, "rb") as f:
            prev = pickle.load(f)
        qs, qdots, taus = prev["q"], prev["qdot"], prev["tau"]

        # Warm start builders (kept in case you want to pass x_init/u_init later)
        _x_init, _u_init = build_initial_guesses(qs, qdots, taus, n_shooting)

        # ------------- Full solutions -------------
        for mode in ["anteversion", "retroversion"]:
            full_pkl = RESULTS_DIR / f"athlete{num}_complet_{mode}.pkl"
            sol_pkl = RESULTS_DIR / f"Sol_athlete{num}_complet_{mode}.pkl"

            if (not use_pkl) or (not full_pkl.exists()):
                ocp = prepare_ocp(
                    biorbd_model_path=model_path,
                    final_time=(1, 0.5, 1),
                    n_shooting=n_shooting,
                    min_time=0.01,
                    max_time=2.0,
                    total_mass=total_mass,
                    init_sol=False,
                    weight_control=0.0001,
                    weight_time=1.0,
                    coef_fig=1.0,
                    final_state_bound=True,
                    mode=mode,
                    n_threads=16,
                    use_sx=False,
                )

                ocp.add_plot_penalty(CostType.ALL)
                ocp.print(to_console=False, to_graph=False)

                solver = Solver.IPOPT()
                solver.set_linear_solver("ma57")
                solver.set_maximum_iterations(20000)

                print(f"start solving (full, {mode})")
                sol = ocp.solve(solver)
                print(f"solving finished (full, {mode})")

                q, qd = stack_states(sol)
                tau = stack_controls(sol)
                t = step_time(sol)
                save_arrays(full_pkl, q, qd, tau, t)
                print("data of full solution saved")

                sol.print_cost()
                sol.graphs(show_bounds=True, show_now=True)

                # save solution object (lighter: remove ocp ref)
                with open(sol_pkl, "wb") as f:
                    del sol.ocp
                    pickle.dump(sol, f)
                print("object solution of full solution saved")

            # Optional animation example
            if num == 502:
                viewer = "pyorerun"
                sol.animate(n_frames=0, viewer=viewer, show_now=True)


if __name__ == "__main__":
    main()
