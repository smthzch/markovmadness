import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

class EmpiricalModel:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def prepare_games(self, games):
        games = games.assign(
            team1 = games["team1"].str.replace("'", ""),
            team2 = games["team2"].str.replace("'", "")
        )

        teams = pd.concat([games["team1"], games["team2"]]).unique()
        self.teams = list(teams)
        self.n_teams = len(self.teams)

        t1 = games["team1"].map(lambda team: self.teams.index(team)).values
        t2 = games["team2"].map(lambda team: self.teams.index(team)).values
        t1_score = games["score1"].values
        t2_score = games["score2"].values

        self.dat = pd.DataFrame({"t1": t1, "t2": t2, "s1": t1_score, "s2": t2_score})

    def extract_t_games(self, ti):
        tdat = self.dat[(self.dat["t1"] == ti) | (self.dat["t2"] == ti)].copy()
        tdat["d"] = np.where(tdat["t1"] == ti, tdat["s1"] - tdat["s2"], tdat["s2"] - tdat["s1"])
        tdat["t"] = np.where(tdat["t1"] != ti, tdat["t1"], tdat["t2"])
        return tdat.drop(columns=["t1", "t2", "s1", "s2"]).groupby("t").mean().reset_index(drop=False)

    def proxy_matchup(self, dat_ti, ti_null):
        tdat = self.dat[((self.dat["t1"].isin(dat_ti["t"])) & (self.dat["t2"] == ti_null)) | ((self.dat["t2"].isin(dat_ti["t"])) & (self.dat["t1"] == ti_null))].copy()
        tdat["d"] = np.where(tdat["t1"] != ti_null, tdat["s1"] - tdat["s2"], tdat["s2"] - tdat["s1"])
        tdat["t"] = np.where(tdat["t1"] != ti_null, tdat["t1"], tdat["t2"])
        tdat["t_"] = ti_null
        tdat = tdat.drop(columns=["t1", "t2", "s1", "s2"])

        proxy = dat_ti.merge(tdat, how="inner", on="t")
        proxy["d"] = proxy["d_x"] + proxy["d_y"]
        proxy["t"] = proxy["t_"]
        return proxy.drop(columns=["d_x", "d_y", "t_"])

    def fit(self, games, rank=False):
        self.prepare_games(games)

        if rank:
            args = [(i, j) for i in range(len(self.teams)) for j in range(i+1, len(self.teams))]
            wins = Parallel(n_jobs=-1)(delayed(self._compute_wins)(i, j) for i, j in tqdm(args))
            wins = np.array(wins).reshape((self.n_teams, self.n_teams))
            wins = (wins - wins.T).sum(axis=1)

            self.ranks = pd.Series(wins, index=self.teams, name="rank")

    def _compute_wins(self, i, j):
        winner = self.predict(self.teams[i], self.teams[j])["winner"]
        return 1 if winner == self.teams[i] else -1

    def predict(self, team1, team2, odds=False, proxy=True):
        t1i = self.teams.index(team1)
        t2i = self.teams.index(team2)

        t1dat = self.extract_t_games(t1i)
        t2dat = self.extract_t_games(t2i)

        ddat = t1dat.merge(t2dat, how="outer", on="t")
        ddat["d"] = ddat["d_x"] - ddat["d_y"]
        if proxy:
            t1_null = ddat[ddat["d_x"].isnull()].reset_index()
            proxy_t1s = [self.proxy_matchup(t1dat, ti_null) for ti_null in t1_null["t"].unique()]
            if len(proxy_t1s) > 0:
                proxy_t1s = pd.concat(proxy_t1s, ignore_index=True)

            t2_null = ddat[ddat["d_y"].isnull()].reset_index()
            proxy_t2s = [self.proxy_matchup(t2dat, ti_null) for ti_null in t2_null["t"].unique()]
            if len(proxy_t2s) > 0:
                proxy_t2s = pd.concat(proxy_t2s, ignore_index=True)

            if len(proxy_t1s) > 0:
                t1dat = pd.concat([t1dat, proxy_t1s], ignore_index=True).groupby("t").mean().reset_index(drop=False)
            if len(proxy_t2s) > 0:
                t2dat = pd.concat([t2dat, proxy_t2s], ignore_index=True).groupby("t").mean().reset_index(drop=False)

        ddat = t1dat.merge(t2dat, how="inner", on="t")
        ddat["d"] = ddat["d_x"] - ddat["d_y"]
        tie, tossup = False, False
        if len(ddat) == 0:
            p1_win = 0.5
            tossup = True
        else:
            p1_win = (ddat.query("d != 0")["d"] > 0).mean()
            alpha = 0.1
            p1_win = (1 - alpha) * p1_win + alpha * 0.5
        if p1_win == 0.5:
            tie = True
            p1_win = 0.75 if ddat["d"].mean() > 0 else 0.25

        return {
            "winner": team1 if p1_win > 0.5 else team2,
            "prob": p1_win if not odds else p1_win / (1 - p1_win),
            "over_under": ddat["d"].mean(),
            "n": len(ddat),
            "tie": tie,
            "tossup": tossup
        }
