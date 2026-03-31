from marchmadpy.data import load_games
from marchmadpy.markov import MarkovModel
from marchmadpy.poisson import PoissonModel, SuperPoissonModel
from marchmadpy.bernoulli import BernoulliModel
from marchmadpy.empirical import EmpiricalModel
from marchmadpy.leastsquares import LeastSquares

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import pandas as pd
import matplotlib.pyplot as plt

def evaluate(model_cls):
    # load data
    data_path = f"data/scores.csv"
    games = load_games(data_path, latest_only=model.__name__ != "SuperPoissonModel")
    # train/test temporal split
    dates = games.date.unique()
    mid = 7 * int(len(dates) / 8)
    dt = dates[mid]
    train = games.query(f"date < @pd.Timestamp('{dt}')")
    test = games.query(f"date >= @pd.Timestamp('{dt}')")

    # fit model
    model = model_cls()
    model.fit(train, boot=False)

    # drop test where test team not in train
    test = test.query(f"team1 in {model.teams} and team2 in {model.teams}")

    prob = model.predict(test.team1.values, test.team2.values, odds=False, proxy=True)
    y = test.t1win.values
    p_ = prob["prob"]
    y_ = 1 * (p_ > 0.5)

    print(f"Accuracy: {accuracy_score(y, y_)}")
    print(f"-Log Loss: {log_loss(y, p_)}")
    print(f"Brier score: {brier_score_loss(y, p_)}")

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [2, 1]})
    CalibrationDisplay.from_predictions(y, p_, ax=ax1, strategy="quantile")
    ax2.hist(p_, bins=20, range=(0, 1), edgecolor="black")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction distribution")
    plt.tight_layout()
    plt.savefig(f"eval/{model_cls.__name__}_calibration.png")

for model in [LeastSquares, PoissonModel, SuperPoissonModel]:
    print(f"\nEvaluating {model.__name__}")
    evaluate(model)
