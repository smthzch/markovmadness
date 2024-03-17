import numpy as np
import pandas as pd

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
from jax.random import PRNGKey

class BernoulliModel:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def prepare_games(self, games):
        games = games.assign(
            team1 = lambda x: x["team1"].str.replace("'", ""),
            team2 = lambda x: x["team2"].str.replace("'", ""),
            t1_win = lambda x: (x["score1"] > x["score2"]) + 0.5 * (x["score1"] == x["score2"])
        )

        # get unique teams
        team1 = set(games.team1)
        team2 = set(games.team2)
        teams = team1.union(team2)
        self.teams = list(set(teams))

        # unpack data
        self.n_teams = len(self.teams)
        self.t1 = np.array([self.teams.index(team) for team in games["team1"]], dtype=int)
        self.t2 = np.array([self.teams.index(team) for team in games["team2"]], dtype=int)
        self.t1_win = games["t1_win"].values
        

    @staticmethod
    def model(n_teams, t1, t2, t1_win=None):
        # hyperpriors
        sd = numpyro.sample("sd", dist.HalfNormal(1))

        # team level params
        with numpyro.plate("n_teams", n_teams):
            power = numpyro.sample("power", dist.Normal(0, sd))

        # game observations
        with numpyro.plate("games", t1.shape[0]):
            numpyro.sample(
                "t1_win", 
                dist.Bernoulli(
                    logits=power[t1] - power[t2]
                ),
                obs=t1_win
            )

    def fit(self, games, mcmc_kwargs={}):
        # prep data
        self.prepare_games(games)

        # mcmc
        kernel = NUTS(BernoulliModel.model)
        self.mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
        self.mcmc.run(PRNGKey(0), self.n_teams, self.t1, self.t2, self.t1_win)
        if self.verbose:
            self.mcmc.print_summary()

        samples = self.mcmc.get_samples()
        self.predictive = Predictive(BernoulliModel.model, posterior_samples=samples)

        ranks = np.array((samples["power"]).mean(axis=0))
        self.ranks = pd.Series(ranks, index=self.teams, name="rank")

    def predict(self, team1, team2, odds=True, seed=1):
        assert hasattr(self, "predictive"), "Model must be fit before predicting."
        
        t1_ix = self.teams.index(team1)
        t2_ix = self.teams.index(team2)

        # posterior predictions
        preds = self.predictive(
            PRNGKey(seed), 
            self.n_teams, 
            np.array([t1_ix], dtype=int), 
            np.array([t2_ix], dtype=int)
        )
        preds["t1_win"] = np.array(preds["t1_win"]).flatten()

        # summary calculations
        prob_t1_win = (preds["t1_win"]).mean().item()
        preds["prob"] = prob_t1_win / (1 - prob_t1_win) if odds else prob_t1_win

        return preds
