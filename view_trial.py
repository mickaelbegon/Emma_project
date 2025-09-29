import os
import pickle

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

num = 500
mode = 'anteversion',# 'retroversion'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "applied_examples/results")

with open(os.path.join(RESULTS_DIR, f"Sol_athlete{num}_complet_{mode}.pkl"), "wb") as file:
    sol = pickle.load(file)

viewer = "pyorerun"
sol.animate(n_frames=0, viewer=viewer, show_now=True)
