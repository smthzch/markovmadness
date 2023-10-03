# March Madness App

This repo contains code to:

1. Get NCAA basketball games scores: `python get_data.py --help`
2. Fit models: `python train.py`
3. Run an app to make matchup predictions: `streamlit run app.py`

A dockerfile is also included in order to package the streamlit app for deployment to Google Cloud Run.

## Setup

This was tested with python 3.10

```shell
pip install -r requirements.txt
```

## Models

This repo contains two models

## PoissonModel

The Poisson model is a Bayesian model that models each team as having an `offense` and a `defense` parameter. The expected score for a team during a game is the interaction of its own `offense` parameter and the opposing teams `defense` parameter.

```math
\begin{aligned}
& \lambda = \exp{(\text{offense}_1 - \text{defense}_2)} \\
& \text{score}_1 \sim \text{Poisson}(\lambda)
\end{aligned}
```
## MarkovModel

The markov model treats games as a markov process where the states are the teams and a transition occurs from one team to another with the probability equal to the fraction of games that the team has lost against the opponent.
Ranks are determined by the stationary distribution.
