import pickle
import pandas as pd
from itertools import product

from marchmadpy.markov import MarkovModel
from marchmadpy.poisson import PoissonModel
from marchmadpy.empirical import EmpiricalModel

def train(model_cls):
    data_path = f"data/scores.csv"
    model_path = f"model/{model_cls.__name__}.pkl"
    
    games = pd.read_csv(data_path, parse_dates=["date"])
    model = model_cls()
    model.fit(games)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)


for model in [MarkovModel, PoissonModel]:
    train(model)
