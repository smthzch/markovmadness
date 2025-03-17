import pickle
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff

from marchmadpy.markov import MarkovModel
from marchmadpy.poisson import PoissonModel
from marchmadpy.empirical import EmpiricalModel


# load models
models = {}
for model_cls in [PoissonModel, MarkovModel, EmpiricalModel]:
    mname = model_cls.__name__
    models[mname] = {}
    with open(f"model/{mname}.pkl", "rb") as f:
        models[model_cls.__name__] = pickle.load(f)

model_type = st.selectbox("Model", ["Empirical", "Poisson", "Markov"])

st.title(f"Zach's {model_type} Model Madness")

model = models[f"{model_type}Model"]

teams = sorted(model.teams)

# select teams matchup
st.header("Matchup Predictions")
team1 = st.selectbox("Team 1", teams, index=teams.index("UConn"))
team2 = st.selectbox("Team 2", teams, index=teams.index("Houston"))
podds = st.selectbox("Probability or Odds?", ["Probability", "Odds"], index=1)

preds = model.predict(team1, team2, odds=(podds == "Odds"), proxy=False)

# make score histogram if poisson model
if model_type == "Poisson":
    pred = preds["prob"]
    t1_scores = preds["t1_score"]
    t2_scores = preds["t2_score"]
    
    hist_data = [t1_scores, t2_scores]
    group_labels = [f"{team1} score", f"{team2} score"]

    fig = ff.create_distplot(hist_data, group_labels)
    st.plotly_chart(fig, use_container_width=True)
else:
    pred = preds["prob"]


# odds of team 1 winning
st.text(f"{podds} of {team1} winning: {round(pred, 2)}")

# list team ranks
st.header("Rankings")
ranks = model.ranks.sort_values(ascending=False)
ranks /= ranks.max()
st.dataframe(ranks)

st.markdown("[github repo](https://github.com/smthzch/markovmadness)")
