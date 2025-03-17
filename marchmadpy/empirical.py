import numpy as np
import pandas as pd
from tqdm import tqdm


class EmpiricalModel:
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
        t1 = np.array([self.teams.index(team) for team in games["team1"]], dtype=int)
        t2 = np.array([self.teams.index(team) for team in games["team2"]], dtype=int)
        t1_score = games["score1"].values
        t2_score = games["score2"].values

        self.dat = pd.DataFrame({"t1": t1, "t2": t2, "s1": t1_score, "s2": t2_score})

    def extract_t_games(self, ti):
        tdat = (
            self.dat
            .query(f"t1 == {ti} or t2 == {ti}")
            .assign(d = lambda x: x["s1"] - x["s2"])
        )
        tdat["d"] = tdat["d"].where(tdat["t1"] == ti, tdat["s2"] - tdat["s1"], axis=0)
        tdat["t"] = tdat["t1"].where(tdat["t1"] != ti, tdat["t2"])
        return tdat.drop(columns=["t1", "t2", "s1", "s2"]).groupby("t").mean().reset_index(drop=False)
    
    def proxy_matchup(self, dat_ti, ti_null):
        tdat = self.dat.query(
            f'(t1 in {dat_ti["t"].tolist()} and t2 == {ti_null}) or (t2 in {dat_ti["t"].tolist()} and t1 == {ti_null})'
        ).assign(d = lambda x: x["s1"] - x["s2"])
        tdat["d"] = tdat["d"].where(tdat["t1"] != ti_null, tdat["s2"] - tdat["s1"], axis=0)
        tdat["t"] = tdat["t1"].where(tdat["t1"] != ti_null, tdat["t2"], axis=0)
        tdat["t_"] = ti_null
        tdat = tdat.drop(columns=["t1", "t2", "s1", "s2"])

        proxy = (
            dat_ti
            .merge(tdat, how="inner", on="t")
            .assign(
                d = lambda x: x["d_x"] + x["d_y"],
                t = lambda x: x["t_"]
            )
            .drop(columns=["d_x", "d_y", "t_"])
        )
        return proxy
    
    def fit(self, games, rank=True):
        self.prepare_games(games)

        if rank:
            t2, t1 = np.meshgrid(self.teams, self.teams)
            wins = np.zeros_like(t1, dtype=int)
            args = [(i, j) for i in range(len(self.teams)) for j in range(i+1, len(self.teams))]
            for i, j in tqdm(args):
                winner = self.predict(self.teams[i], self.teams[j])["winner"]
                wins[i,j] = 1 if winner == t1[i,j] else -1

            wins = (wins - wins.T).sum(axis=1)

            self.ranks = pd.Series(wins, index=self.teams, name="rank")
    
    def predict(self, team1, team2, odds=False, proxy=True):
        t1i = self.teams.index(team1)
        t2i = self.teams.index(team2)

        # extract games played by each team
        t1dat = self.extract_t_games(t1i)
        t2dat = self.extract_t_games(t2i)

        # extract second order matchups and add to data
        ddat = t1dat.merge(t2dat, how="outer", on="t").assign(d = lambda x: x["d_x"] - x["d_y"])
        if proxy:
            # t1
            t1_null = ddat[ddat["d_x"].isnull()].reset_index()
            proxy_t1s = []
            for ti_null in t1_null["t"].unique():
                proxy_t1 = self.proxy_matchup(t1dat, ti_null)
                proxy_t1s.append(proxy_t1)
            proxy_t1s = pd.concat(proxy_t1s, ignore_index=True)

            # t2
            t2_null = ddat[ddat["d_y"].isnull()].reset_index()
            proxy_t2s = []
            for ti_null in t2_null["t"].unique():
                proxy_t2 = self.proxy_matchup(t2dat, ti_null)
                proxy_t2s.append(proxy_t2)
            proxy_t2s = pd.concat(proxy_t2s, ignore_index=True)

            # add data
            t1dat = pd.concat([t1dat, proxy_t1s], ignore_index=True).groupby("t").mean().reset_index(drop=False)
            t2dat = pd.concat([t2dat, proxy_t2s], ignore_index=True).groupby("t").mean().reset_index(drop=False)
            
        # match up two teams and take score difference
        ddat = t1dat.merge(t2dat, how="inner", on="t").assign(d = lambda x: x["d_x"] - x["d_y"])

        tie, tossup = False, False
        if len(ddat) == 0:
            p1_win =  0.5#np.random.rand()
            tossup = True
        else:
            p1_win = (ddat.query("d != 0")["d"] > 0).mean()
            #p1_win = np.random.uniform(0.51, 1) if ddat["d"].mean() > 0 else np.random.uniform(0, 0.5)
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
