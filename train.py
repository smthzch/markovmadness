import typer
import pickle
import pandas as pd

from model import MarkovModel

def main(mens_league: bool=True):
    suffix = "xy" if mens_league else "xx"
    data_path = f"data/scores_{suffix}.csv"
    model_path = f"model/mm{suffix}.pkl"
    
    games = pd.read_csv(data_path, parse_dates=["date"])
    model = MarkovModel(games)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    typer.run(main)