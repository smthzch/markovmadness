import pickle
import pandas as pd
from itertools import product

from marchmadpy.markov import MarkovModel
from marchmadpy.poisson import PoissonModel
from marchmadpy.bernoulli import BernoulliModel

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate(model_cls):
    data_path = f"data/scores.csv"
    
    games = pd.read_csv(data_path, parse_dates=["date"])
    games = games.assign(
        tie = lambda x: x["score1"] == x["score2"],
        t1win = lambda x: (x["score1"] > x["score2"]),
        t2win = lambda x: (~x["t1win"]) * (~x["tie"])
    )
    model = model_cls()

    dates = games.date.unique()
    mid = 7 * int(len(dates) / 8)
    dt = dates[mid]
    train = games.query(f"date < @pd.Timestamp('{dt}')")
    test = games.query(f"date >= @pd.Timestamp('{dt}')")

    model.fit(train)

    y = []
    y_ = []
    p_ = []
    for i, row in tqdm(test.iterrows(), total=test.shape[0]):
        t1 = row.team1
        t2 = row.team2
        
        if t1 not in model.ranks or t2 not in model.ranks:
            continue

        prob = model.predict(t1, t2, odds=False)
        y += [row.t1win]
        p_ += [prob] if isinstance(prob, float) else [prob["prob"]]
        y_ += [1 * (p_[-1] > 0.5)]

    prob_true, prob_pred = calibration_curve(y, p_, n_bins=5)

    print(f"Accuracy: {accuracy_score(y, y_)}")
    print(f"-Log Loss: {log_loss(y, p_)}")

    plt.clf()
    plt.scatter(x=prob_true, y=prob_pred)
    plt.plot([0,1], [0,1])
    plt.savefig(f"eval/{model_cls.__name__}_calibration.png")

for model in [MarkovModel, BernoulliModel, PoissonModel]:
    print(f"\nEvaluating {model.__name__}")
    evaluate(model)
