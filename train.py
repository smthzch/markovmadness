import pickle
import pandas as pd
from itertools import product

from marchmadpy.markov import MarkovModel
from marchmadpy.poisson import PoissonModel

def train(model_cls, mens_league: bool=True):
    suffix = "xy" if mens_league else "xx"
    data_path = f"data/scores_{suffix}.csv"
    model_path = f"model/{model_cls.__name__}_{suffix}.pkl"
    
    games = pd.read_csv(data_path, parse_dates=["date"])
    model = model_cls()
    model.fit(games)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)


for model, mens_league in product(
    [MarkovModel, PoissonModel],
    [True, False]
):
    train(model, mens_league)
