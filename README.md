# March Madness App

This repo contains code to:

1. Get NCAA basketball games scores: `python get_data.py --help`
2. Fit models: `python train.py`
3. Run an app to make matchup predictions: `streamlit run app.py`

A dockerfile is also included in order to package the streamlit app for deployment to Google Cloud Run.

## Setup

This was tested with python 3.10

```shell
mkvirtualenv marchmad
pip install -r requirements.txt
```

# Models

This repo contains two models

## Bernoulli Model

The Bernoulli model is a Bayesian model that models each team as having a `power` level parameter. The probability of a team winning a matchup is based on the interaction between the two teams' power levels.

```math
\begin{aligned}
& \text{power}_i \sim \text{Normal}(0,1) \\
& p_{i,j} = \frac{1}{1 + \exp{(\text{power}_i - \text{power}_j)}} \\
& \text{win}_{i,j} \sim \text{Bernoulli}(p_{i,j}) \\
\end{aligned}
```

## Poisson Model

The Poisson model is a Bayesian model that models each team as having an `offense` and a `defense` parameter. The expected score for a team during a game is the interaction of its own `offense` parameter and the opposing team's `defense` parameter.

```math
\begin{aligned}
& \lambda = \exp{(\text{offense}_1 - \text{defense}_2)} \\
& \text{score}_1 \sim \text{Poisson}(\lambda)
\end{aligned}
```
## ~~Markov Model~~

:warning: This model is deprecated.

~~The markov model treats games as a markov process where the states are the teams and a transition occurs from one team to another with the probability equal to the fraction of games that the team has lost against the opponent.
Ranks are determined by the stationary distribution.~~
