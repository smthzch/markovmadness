#%%
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from marchmadpy.empirical import EmpiricalModel
from marchmadpy.markov import MarkovModel
from marchmadpy.poisson import PoissonModel

# %%
raw_dat = pd.read_csv("data/kaggle_m.csv").query("Season == 2025")
raw_dat["team1"] = list(map(str, raw_dat["WTeamID"]))
raw_dat["team2"] = list(map(str, raw_dat["LTeamID"]))
raw_dat["score1"] = raw_dat["WScore"]
raw_dat["score2"] = raw_dat["LScore"]

raw_dat1 = pd.read_csv("data/kaggle_w.csv").query("Season == 2025")
raw_dat1["team1"] = list(map(str, raw_dat1["WTeamID"]))
raw_dat1["team2"] = list(map(str, raw_dat1["LTeamID"]))
raw_dat1["score1"] = raw_dat1["WScore"]
raw_dat1["score2"] = raw_dat1["LScore"]

raw_dat = pd.concat([raw_dat, raw_dat1])


submission = pd.read_csv("data/SampleSubmissionStage2.csv")
assert (submission["ID"].str.split("_").str[1] < submission["ID"].str.split("_").str[2]).all()

# %%
def predict(gid):
    _, t1, t2 = gid.split("_")
    res = model.predict(t1, t2, odds=False, proxy=True)
    return res["prob"]

for model_cls in [EmpiricalModel, MarkovModel]:
    model = model_cls()
    model.fit(raw_dat, rank=False)
    submission["Pred"] = Parallel(10)(
        delayed(predict)(submission.iloc[i,0]) 
        for i in tqdm(range(len(submission)))
    )
    submission["Pred"] = submission["Pred"].where(~submission["Pred"].isna(), 0.5)

    submission.to_csv(f"data/submission_{model.__class__.__name__}.csv", index=False)

#%%
model = PoissonModel()
model.fit(raw_dat, rank=False)

preds = model.predict(submission["ID"].str.split("_").str[1].tolist(), submission["ID"].str.split("_").str[2].tolist())
submission["Pred"] = preds["prob"]

submission.to_csv(f"data/submission_{model.__class__.__name__}.csv", index=False)
