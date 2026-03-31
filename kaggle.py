#%%
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from marchmadpy.data import load_kaggle
from marchmadpy.leastsquares import LeastSquares
from marchmadpy.poisson import SuperPoissonModel

# %%
raw_dat = load_kaggle(minyear=2025, maxyear=2025)

submission = pd.read_csv("data/SampleSubmissionStage25.csv")
assert (submission["ID"].str.split("_").str[1] < submission["ID"].str.split("_").str[2]).all()

# %%
def predict(gid):
    _, t1, t2 = gid.split("_")
    res = model.predict(t1, t2, odds=False)
    return res["prob"][0]

for model_cls in [LeastSquares]:
    model = model_cls()
    model.fit(raw_dat, boot=True)
    submission["Pred"] = Parallel(-1)(
        delayed(predict)(submission.iloc[i,0]) 
        for i in tqdm(range(len(submission)))
    )
    submission["Pred"] = submission["Pred"].where(~submission["Pred"].isna(), 0.5)

    submission.to_csv(f"data/submission_{model.__class__.__name__}.csv", index=False)

#%%
model = LeastSquares()
model.fit(raw_dat, boot=True, cv=True)

preds = model.predict(
    submission["ID"].str.split("_").str[1].values,
    submission["ID"].str.split("_").str[2].values
)

submission["Pred"] = preds["prob"]
submission["Pred"] = submission["Pred"].where(~submission["Pred"].isna(), 0.5)
submission.to_csv(f"data/submission_{model.__class__.__name__}_25.csv", index=False)

#%%
model = SuperPoissonModel()
model.fit(raw_dat, rank=False)

preds = model.predict(submission["ID"].str.split("_").str[1].tolist(), submission["ID"].str.split("_").str[2].tolist())
submission["Pred"] = preds["prob"]

submission.to_csv(f"data/submission_{model.__class__.__name__}.csv", index=False)

# %%
