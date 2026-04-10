import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

class EmpiricalModel:
    """
    A non-parametric model that predicts game outcomes based on historical
    score differentials between teams via shared common opponents.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def prepare_games(self, games):
        """
        Encode teams as integer indices and store game data as a DataFrame
        of index-based team IDs and scores.
        """
        teams = pd.concat([games["team1"], games["team2"]]).unique()
        self.teams = list(teams)
        self.n_teams = len(self.teams)

        # Map team names to integer indices
        t1 = games["team1"].map(lambda team: self.teams.index(team)).values
        t2 = games["team2"].map(lambda team: self.teams.index(team)).values
        t1_score = games["score1"].values
        t2_score = games["score2"].values

        self.dat = pd.DataFrame({"t1": t1, "t2": t2, "s1": t1_score, "s2": t2_score})

    def extract_t_games(self, ti):
        """
        For team `ti`, return a DataFrame of mean score differentials against
        each opponent the team has faced (positive = team `ti` won by that margin).
        """
        tdat = self.dat[(self.dat["t1"] == ti) | (self.dat["t2"] == ti)].copy()
        # Compute score diff from ti's perspective regardless of home/away column
        tdat["d"] = np.where(tdat["t1"] == ti, tdat["s1"] / tdat["s2"], tdat["s2"] / tdat["s1"])
        # Identify the opponent in each game
        tdat["t"] = np.where(tdat["t1"] != ti, tdat["t1"], tdat["t2"])
        return tdat.drop(columns=["t1", "t2", "s1", "s2"]).groupby("t").mean().reset_index(drop=False)

    def proxy_matchup(self, dat_ti, ti_null):
        """
        Estimate team `ti`'s effective differential against `ti_null` using
        shared intermediary opponents (transitive comparisons).

        dat_ti  : score differentials for the primary team against its opponents
        ti_null : index of the team that has no direct data with the primary team
        """
        # Find games where ti_null played any team that ti also played
        tdat = self.dat[
            ((self.dat["t1"].isin(dat_ti["t"])) & (self.dat["t2"] == ti_null)) |
            ((self.dat["t2"].isin(dat_ti["t"])) & (self.dat["t1"] == ti_null))
        ].copy()

        # Score diff from the shared opponent's perspective (not ti_null's)
        tdat["d"] = np.where(tdat["t1"] != ti_null, tdat["s1"] / tdat["s2"], tdat["s2"] / tdat["s1"])
        tdat["t"] = np.where(tdat["t1"] != ti_null, tdat["t1"], tdat["t2"])
        tdat["t_"] = ti_null
        tdat = tdat.drop(columns=["t1", "t2", "s1", "s2"])

        # Combine: ti vs shared_opponent + shared_opponent vs ti_null
        proxy = dat_ti.merge(tdat, how="inner", on="t")
        proxy["d"] = proxy["d_x"] * proxy["d_y"]  # transitive differential
        proxy["t"] = proxy["t_"]
        return proxy.drop(columns=["d_x", "d_y", "t_"])

    def fit(self, games, rank=False, **kwargs):
        """
        Prepare game data and optionally compute a global team ranking by
        simulating all head-to-head matchups in parallel.
        """
        self.prepare_games(games)

        if rank:
            # Build all ordered pairs (i, j) and predict each matchup
            args = [(i, j) for i in range(len(self.teams)) for j in range(len(self.teams))]
            wins = Parallel(n_jobs=-1)(delayed(self._compute_wins)(i, j) for i, j in tqdm(args))
            wins = np.array(wins).reshape((self.n_teams, self.n_teams))
            # Net wins = wins - losses summed across all opponents
            wins = (wins - wins.T).sum(axis=1)

            self.ranks = pd.Series(wins, index=self.teams, name="rank")

    def _compute_wins(self, i, j):
        """Return +1 if team i beats team j, -1 otherwise."""
        winner = self.predict(self.teams[i], self.teams[j])["winner"]
        return 1 if winner == self.teams[i] else -1

    def predict(self, teams1, teams2, odds=False, proxy=False, **kwargs):
        """
        Predict the winner(s) of one or more matchups.

        teams1, teams2 : team name(s) — scalars or lists
        odds           : if True, return win probability as odds ratio instead
        proxy          : if True, augment missing head-to-head data with
                         transitive comparisons through common opponents

        Returns a dict of numpy arrays with keys:
            winner, prob, over_under, n, tie, tossup
        """
        # Normalize scalar inputs to lists
        if not isinstance(teams1, (list, np.ndarray)):
            teams1 = [teams1]
            teams2 = [teams2]

        res = []
        # Show progress bar only for batch predictions
        items = tqdm(zip(teams1, teams2), total=len(teams1)) if len(teams1) > 1 else zip(teams1, teams2)

        for team1, team2 in items:
            t1i = self.teams.index(team1)
            t2i = self.teams.index(team2)

            # Get mean score differentials for each team against their opponents
            t1dat = self.extract_t_games(t1i)
            t2dat = self.extract_t_games(t2i)

            # Outer join to identify opponents with missing data for either team
            ddat = t1dat.merge(t2dat, how="outer", on="t")
            ddat["d"] = ddat["d_x"] / ddat["d_y"]

            if proxy:
                # Fill missing team1 data via transitive comparisons
                t1_null = ddat[ddat["d_x"].isnull()].reset_index()
                proxy_t1s = [self.proxy_matchup(t1dat, ti_null) for ti_null in t1_null["t"].unique()]
                if len(proxy_t1s) > 0:
                    proxy_t1s = pd.concat(proxy_t1s, ignore_index=True)

                # Fill missing team2 data via transitive comparisons
                t2_null = ddat[ddat["d_y"].isnull()].reset_index()
                proxy_t2s = [self.proxy_matchup(t2dat, ti_null) for ti_null in t2_null["t"].unique()]
                if len(proxy_t2s) > 0:
                    proxy_t2s = pd.concat(proxy_t2s, ignore_index=True)

                # Merge proxy records back into each team's history
                if len(proxy_t1s) > 0:
                    t1dat = pd.concat([t1dat, proxy_t1s], ignore_index=True).groupby("t").mean().reset_index(drop=False)
                if len(proxy_t2s) > 0:
                    t2dat = pd.concat([t2dat, proxy_t2s], ignore_index=True).groupby("t").mean().reset_index(drop=False)

            # Inner join: only use common opponents for the final comparison
            ddat = t1dat.merge(t2dat, how="inner", on="t")
            ddat["d"] = ddat["d_x"] / ddat["d_y"]  # positive = team1 favorable

            tie, tossup = False, False
            if len(ddat) == 0:
                # No shared opponents at all — pure coin flip
                p1_win = 0.5
                tossup = True
            else:
                # Fraction of common-opponent comparisons favoring team1
                p1_win = (ddat.query("d != 1")["d"] > 1).mean()
                # Shrink slightly toward 50% to regularize extreme estimates
                alpha = 0.1
                p1_win = (1 - alpha) * p1_win + alpha * 0.5

            if p1_win == 0.5:
                # Break ties using raw mean differential direction
                tie = True
                p1_win = 0.75 if ddat["d"].mean() > 1 else 0.25

            res += [{
                "winner": team1 if p1_win > 0.5 else team2,
                "prob": p1_win if not odds else p1_win / (1 - p1_win),
                "over_under": ddat["d"].mean(),  # expected margin (team1 perspective)
                "n": len(ddat),                  # number of shared opponents used
                "tie": tie,
                "tossup": tossup,
            }]

        res = pd.DataFrame(res)
        return {col: res[col].to_numpy() for col in res.columns}
