import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
from jax.random import PRNGKey

class PoissonModel:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def prepare_games(self, games):
        games = games.assign(
            team1 = lambda x: x["team1"].str.replace("'", ""),
            team2 = lambda x: x["team2"].str.replace("'", "")
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
        self.t1_score = games["score1"].values
        self.t2_score = games["score2"].values
        

    @staticmethod
    def model(n_teams, t1, t2, t1_score=None, t2_score=None):
        # hyperpriors
        of_sd = numpyro.sample("of_sd", dist.HalfNormal(1))
        df_sd = numpyro.sample("df_sd", dist.HalfNormal(1))

        # team level params
        with numpyro.plate("n_teams", n_teams):
            offense = numpyro.sample("offense", dist.Normal(0, of_sd))
            defense = numpyro.sample("defense", dist.Normal(0, df_sd))

        # game observations
        with numpyro.plate("games", t1.shape[0]):
            numpyro.sample(
                "t1_score", 
                dist.Poisson(
                    jnp.exp(offense[t1] - defense[t2])
                ),
                obs=t1_score
            )
            numpyro.sample(
                "t2_score", 
                dist.Poisson(
                    jnp.exp(offense[t2] - defense[t1])
                ),
                obs=t2_score
            )

    def fit(self, games, mcmc_kwargs={}):
        # prep data
        self.prepare_games(games)

        # mcmc
        kernel = NUTS(PoissonModel.model)
        self.mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
        self.mcmc.run(PRNGKey(0), self.n_teams, self.t1, self.t2, self.t1_score, self.t2_score)
        if self.verbose:
            self.mcmc.print_summary()

        samples = self.mcmc.get_samples()
        self.predictive = Predictive(PoissonModel.model, posterior_samples=samples)

        ranks = np.array((samples["offense"] + samples["defense"]).mean(axis=0))
        self.ranks = pd.Series(ranks, index=self.teams, name="rank")

    def predict(self, team1, team2, odds=False, seed=1, **kwargs):
        assert hasattr(self, "predictive"), "Model must be fit before predicting."
        
        t1_ix = self.teams.index(team1)
        t2_ix = self.teams.index(team2)

        # posterior predictions
        jax.clear_caches()
        preds = self.predictive(
            PRNGKey(seed), 
            self.n_teams, 
            np.array([t1_ix], dtype=int), 
            np.array([t2_ix], dtype=int)
        )
        preds["t1_score"] = np.array(preds["t1_score"]).flatten()
        preds["t2_score"] = np.array(preds["t2_score"]).flatten()

        # summary calculations
        prob_t1_win = (preds["t1_score"] > preds["t2_score"]).mean().item()
        preds["prob"] = prob_t1_win / (1 - prob_t1_win) if odds else prob_t1_win
        preds["winner"] = team1 if prob_t1_win > 0.5 else team2
        
        return preds
