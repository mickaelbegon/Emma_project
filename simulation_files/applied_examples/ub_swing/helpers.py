from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import pickle

from bioptim import (
    Node, InterpolationType, ObjectiveList, ObjectiveFcn, ConstraintList,
    ObjectiveWeight, SolutionMerge, BoundsList,
)
from .config import PHASE_COUNT, IDX, S_JOINTS, ROT_START

def split_indices(n_shooting: Iterable[int]):
    ns = np.array(tuple(n_shooting), dtype=int)
    s_ctrl  = np.concatenate(([0], np.cumsum(ns)))      # ni
    s_state = np.concatenate(([0], np.cumsum(ns + 1)))  # ni+1
    return s_state, s_ctrl

def build_initial_guesses(qs: np.ndarray, qdots: np.ndarray, taus: np.ndarray, n_shooting: Iterable[int]):
    from bioptim import InitialGuessList
    s_state, s_ctrl = split_indices(n_shooting)
    x_init = InitialGuessList(); u_init = InitialGuessList()
    for p, (sa, sb, ua, ub) in enumerate(zip(s_state[:-1], s_state[1:], s_ctrl[:-1], s_ctrl[1:])):
        x_init.add("q",    qs[:,   sa:sb], InterpolationType.EACH_FRAME, phase=p)
        x_init.add("qdot", qdots[:, sa:sb], InterpolationType.EACH_FRAME, phase=p)
        u_init.add("tau",  taus[:,  ua:ub], InterpolationType.EACH_FRAME, phase=p)
    return x_init, u_init

def add_common_objectives(objectives: ObjectiveList, weight_control: float, weight_time: float,
                          min_time: float, max_time: float, coef_fig: float,
                          init_sol: bool, phase_count: int = PHASE_COUNT):
    for p in range(phase_count):
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True,
                       weight=weight_control, phase=p)
        objectives.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight_time,
                       min_bound=min_time, max_bound=max_time, phase=p)
    if not init_sol:
        for p in range(phase_count):
            objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1.0,
                           derivative=True, phase=p)
            for name, w in {"Elbow": 5, "KneeR": 10, "FootR": 2}.items():
                objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q",
                               index=IDX[name], target=0, weight=w * coef_fig, phase=p)

def add_symmetry_constraints(constraints: ConstraintList, phase_count: int = PHASE_COUNT):
    from bioptim import ConstraintFcn
    pairs = [("RxThighR","RxThighL",-1), ("RyThighR","RyThighL",1), ("KneeR","KneeL",1), ("FootR","FootL",1)]
    for p in range(phase_count):
        for a, b, c in pairs:
            constraints.add(ConstraintFcn.PROPORTIONAL_STATE, key="q", phase=p, node=Node.ALL,
                            first_dof=IDX[a], second_dof=IDX[b], coef=c)

def make_x_bounds(models, total_mass: float, stiffness: float) -> BoundsList:
    xb = BoundsList()
    for p in range(PHASE_COUNT):
        xb.add("q",    bounds=models[0].bounds_from_ranges("q"),    phase=p)
        xb.add("qdot", bounds=models[0].bounds_from_ranges("qdot"), phase=p)
    # initial node
    xb[0]["q"][:, 0] = 0
    xb[0]["q"][IDX["RyHands"], 0] = ROT_START
    xb[0]["q"][1, 0] = - total_mass * 9.81 / stiffness  # equilibrium in z
    xb[0]["qdot"][:, 0] = 0
    # end of phase 1
    xb[1]["q"][IDX["RyHands"], -1] = -np.pi
    return xb

def stack_states(sol) -> Tuple[np.ndarray, np.ndarray]:
    parts = sol.decision_states(to_merge=[SolutionMerge.NODES])
    q  = np.hstack([p["q"]    for p in parts])
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
