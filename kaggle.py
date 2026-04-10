#%%
import pandas as pd

from marchmadpy.data import load_kaggle
from marchmadpy.leastsquares import LeastSquares
from marchmadpy.poisson import SuperPoissonModel
from marchmadpy.bernoulli import BernoulliModel

# %%
raw_dat = load_kaggle(minyear=2026, maxyear=2026)

submission = pd.read_csv("data/SampleSubmissionStage2.csv")
assert (submission["ID"].str.split("_").str[1] < submission["ID"].str.split("_").str[2]).all()


#%%
model = LeastSquares()
model.fit(raw_dat, boot=False, cv=False)

preds = model.predict(
    submission["ID"].str.split("_").str[1].values,
    submission["ID"].str.split("_").str[2].values
)

submission["Pred"] = preds["prob"]
submission["Pred"] = submission["Pred"].where(~submission["Pred"].isna(), 0.5)
submission.to_csv(f"data/submission_{model.__class__.__name__}.csv", index=False)

#%%
model = BernoulliModel()
model.fit(raw_dat)

preds = model.predict(
    submission["ID"].str.split("_").str[1].values,
    submission["ID"].str.split("_").str[2].values
)
submission["Pred"] = preds["prob"]

submission.to_csv(f"data/submission_{model.__class__.__name__}.csv", index=False)

# %%
