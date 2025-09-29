import os
import pickle
import numpy as np
import pandas as pd
from pyorerun import BiorbdModel, PhaseRerun

def load_data(filename):

    file1 = filename + ".pkl"
    if not os.path.exists(file1) or os.path.getsize(file1) == 0:
        raise FileNotFoundError(f"Fichier absent ou vide: {file1}")

    with open(file1, "rb") as f:
        data = pickle.load(f)

    q    = data["q"]
    qdot = data["qdot"]
    tau  = data["tau"]
    time  = data["time"]

    file2 = filename + "_sol.pkl"
    if not os.path.exists(file2) or os.path.getsize(file2) == 0:
        raise FileNotFoundError(f"Fichier absent ou vide: {file2}")

    with open(file2, "rb") as f:
        sol = pickle.load(f)

    sol.cost


    return max(tau)



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "applied_examples", "results2")  # <- bon dossier
masses = pd.read_csv(CURRENT_DIR + "/applied_examples/masses.csv")

for athlete in range(1,20):
    modelname = os.path.join(CURRENT_DIR, "applied_examples", f"athlete_{athlete}_deleva.bioMod")
    model = BiorbdModel(modelname)
    masse = masses["total_mass"][athlete - 1]

    base_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_base")#
    retro_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_retroversion")
    ante_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_anteversion")

    load_data(base_pkl)

