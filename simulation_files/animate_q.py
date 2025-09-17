import matplotlib
matplotlib.use("Qt5Agg")

from pyorerun import BiorbdModel, PhaseRerun

import os
import pickle

athlete = 50
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "applied_examples", "results2")  # <- bon dossier
modelname = os.path.join(CURRENT_DIR, "applied_examples", f"athlete_{athlete}_deleva.bioMod")
model = BiorbdModel(modelname)


file_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_base.pkl")#
file_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_retroversion.pkl")
#file_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_anteversion.pkl")

if not os.path.exists(file_pkl) or os.path.getsize(file_pkl) == 0:
    raise FileNotFoundError(f"Fichier absent ou vide: {file_pkl}")

with open(file_pkl, "rb") as f:
    data = pickle.load(f)

qs    = data["q"]
qdots = data["qdot"]
taus  = data["tau"]
time  = data["time"]         # <- tu sauvegardes 'time' dans tes pkl
# taudots = data["taudot"]   # <- supprimé: cette clé n'existe pas dans tes pkl

n_shooting = (25, 25, 50)

viz = PhaseRerun(time)
viz.add_animated_model(model, qs)
viz.rerun("swing")


