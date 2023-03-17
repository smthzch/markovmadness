import pickle
import pandas as pd
import streamlit as st
from model import MarkovModel

with open("model/mm_xy.pkl", "rb") as f:
    model_xy = pickle.load(f)
with open("model/mm_xx.pkl", "rb") as f:
    model_xx = pickle.load(f)

models = {
    "Men": model_xy,
    "Women": model_xx
}

st.title("Zach's Markov Model Madness")

# select men or women
league = st.selectbox("League", ["Men", "Women"])
model = models[league]

teams = sorted(model.teams)
stationary = model.stationary.sort_values(ascending=False)
stationary /= stationary.max()

# select teams matchup
st.header("Matchup Predictions")
team1 = st.selectbox("Team 1", teams, index=teams.index("Purdue"))
team2 = st.selectbox("Team 2", teams, index=teams.index("Alabama"))
podds = st.selectbox("Probability or Odds?", ["Probability", "Odds"], index=1)

pred = model.predict(team1, team2, odds = (podds == "Odds"))

# list team ranks
st.text(f"{podds} of {team1} winning: {round(pred, 2)}")

st.header("Rankings")
st.dataframe(stationary)
