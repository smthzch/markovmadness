import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.infer.reparam import LocScaleReparam
from jax.random import PRNGKey


class PoissonModel:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def prepare_games(self, games):
        games = games.query(f"season_year == {games['season_year'].max()}")
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
        # priors
        mu = numpyro.sample("mu", dist.Normal(0, 1))
        sd = numpyro.sample("sd", dist.HalfNormal(1))

        # team level params
        with numpyro.plate("n_teams", n_teams):
            offense = numpyro.sample("offense", dist.Normal(0, sd))
            defense = numpyro.sample("defense", dist.Normal(0, sd))

        # game observations
        with numpyro.plate("games", t1.shape[0]):
            numpyro.sample(
                "t1_score", 
                dist.Poisson(
                    jnp.exp(offense[t1] - defense[t2] + mu)
                ),
                obs=t1_score
            )
            numpyro.sample(
                "t2_score", 
                dist.Poisson(
                    jnp.exp(offense[t2] - defense[t1] + mu)
                ),
                obs=t2_score
            )

    def fit(self, games, mcmc_kwargs={}, rank=True, **kwargs):
        # prep data
        self.prepare_games(games)

        # mcmc
        kernel = NUTS(PoissonModel.model)
        self.mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000)
        self.mcmc.run(PRNGKey(0), self.n_teams, self.t1, self.t2, self.t1_score, self.t2_score)
        if self.verbose:
            self.mcmc.print_summary()

        samples = self.mcmc.get_samples()
        self.predictive = Predictive(PoissonModel.model, posterior_samples=samples)

        ranks = np.array((samples["offense"] + samples["defense"]).mean(axis=0))
        self.ranks = pd.Series(ranks, index=self.teams, name="rank")

    def predict(self, team1, team2, odds=False, seed=1, **kwargs):
        assert hasattr(self, "predictive"), "Model must be fit before predicting."
        if not isinstance(team1, (list, np.ndarray)):
            team1 = [team1]
            team2 = [team2]

        t1_ix = [self.teams.index(t1) for t1 in team1]
        t2_ix = [self.teams.index(t2) for t2 in team2]

        # posterior predictions
        jax.clear_caches()
        preds = self.predictive(
            PRNGKey(seed), 
            self.n_teams, 
            np.array(t1_ix, dtype=int), 
            np.array(t2_ix, dtype=int)
        )

        t1_score = np.array(preds["t1_score"]).T
        t2_score = np.array(preds["t2_score"]).T
        # use masked array to account for ties
        mask = t1_score == t2_score
        t1_win = np.ma.array(t1_score > t2_score, mask=mask)
        prob_t1_win = t1_win.mean(axis=1).data
        
        return {
            "t1_score": t1_score,
            "t2_score": t2_score,
            "prob": prob_t1_win / (1 - prob_t1_win) if odds else prob_t1_win,
            "winner": np.where(prob_t1_win > 0.5, team1, team2)
        }

class SuperPoissonModel:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def prepare_games(self, games):
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
        self.season = games['season_index'].values
        self.n_seasons = self.season.max() + 1
        
    @staticmethod
    def model(n_teams, n_seasons, t1, t2, season, t1_score=None, t2_score=None):
        mu = numpyro.sample("mu", dist.Normal(0, 10))
        global_sd = numpyro.sample("global_sd", dist.HalfNormal(1))
        seasonal_sd = numpyro.sample("seasonal_sd", dist.HalfNormal(0.1))

        with numpyro.plate("n_teams", n_teams):
            offense_global = numpyro.sample("offense_global", dist.Normal(0, global_sd))
            defense_global = numpyro.sample("defense_global", dist.Normal(0, global_sd))

        with numpyro.plate("seasons", n_seasons, dim=-2):
            with numpyro.plate("teams_in_seasons", n_teams, dim=-1):
                offense = numpyro.sample("offense", dist.Normal(offense_global, seasonal_sd))
                defense = numpyro.sample("defense", dist.Normal(defense_global, seasonal_sd))
                

        with numpyro.plate("games", t1.shape[0]):
            # Use the deterministic results for the Poisson rate
            t1_off = offense[season, t1]
            t2_def = defense[season, t2]
            t2_off = offense[season, t2]
            t1_def = defense[season, t1]

            numpyro.sample("t1_score", dist.Poisson(jnp.exp(t1_off - t2_def + mu)), obs=t1_score)
            numpyro.sample("t2_score", dist.Poisson(jnp.exp(t2_off - t1_def + mu)), obs=t2_score)

    def fit(self, games, mcmc_kwargs={}, rank=True, **kwargs):
        # prep data
        self.prepare_games(games)

        # mcmc
        reparam_config = {
            "offense": LocScaleReparam(centered=0),
            "defense": LocScaleReparam(centered=0)
        }
        reparam_model = numpyro.handlers.reparam(SuperPoissonModel.model, config=reparam_config)
        
        kernel = NUTS(reparam_model)
        self.mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000)
        self.mcmc.run(
            PRNGKey(0), 
            self.n_teams,
            self.n_seasons,
            self.t1, 
            self.t2,
            self.season, 
            self.t1_score, 
            self.t2_score
        )
        if self.verbose:
            self.mcmc.print_summary()

        self.samples = self.mcmc.get_samples()
        self.predictive = Predictive(reparam_model, posterior_samples=self.samples)

    def predict(self, team1, team2, odds=False, seed=1, **kwargs):
        assert hasattr(self, "predictive"), "Model must be fit before predicting."
        if not isinstance(team1, (list, np.ndarray)):
            team1 = [team1]
            team2 = [team2]

        t1_ix = [self.teams.index(t1) for t1 in team1]
        t2_ix = [self.teams.index(t2) for t2 in team2]
        season = [self.season.max() for _ in team1]

        # posterior predictions
        jax.clear_caches()
        preds = self.predictive(
            PRNGKey(seed), 
            self.n_teams,
            self.n_seasons,
            np.array(t1_ix, dtype=int), 
            np.array(t2_ix, dtype=int),
            np.array(season, dtype=int),
        )

        t1_score = np.array(preds["t1_score"]).T
        t2_score = np.array(preds["t2_score"]).T
        # use masked array to account for ties
        mask = t1_score == t2_score
        t1_win = np.ma.array(t1_score > t2_score, mask=mask)
        prob_t1_win = t1_win.mean(axis=1).data
        
        return {
            "t1_score": t1_score,
            "t2_score": t2_score,
            "prob": prob_t1_win / (1 - prob_t1_win) if odds else prob_t1_win,
            "winner": np.where(prob_t1_win > 0.5, team1, team2)
        }