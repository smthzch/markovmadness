import numpy as np
import pandas as pd

class MarkovModel:
    def __init__(self, games, eps=1e-3):
        self.eps = eps # for markov model regularization

        self.games = games.assign(
            tie = lambda x: x["score1"] == x["score2"],
            t1win = lambda x: (x["score1"] > x["score2"]),
            t2win = lambda x: (~x["t1win"]) * (~x["tie"])
        )

        # get unique teams
        team1 = set(games.team1)
        team2 = set(games.team2)
        teams = [
            team.replace("'", "")
            for team in team1.union(team2)
        ]
        self.teams = list(set(teams))

        self.make_count_matrix()
        self.fit()
    
    def make_count_matrix(self):
        print("Make count matrix.")
        count = pd.DataFrame(0.0, index=self.teams, columns=self.teams)
        for ix, row in self.games.iterrows():
            w1 = row["t1win"] * 1
            w2 = row["t2win"] * 1

            count.loc[row["team1"], row["team1"]] += w1
            count.loc[row["team2"], row["team2"]] += w2
            count.loc[row["team1"], row["team2"]] += w2
            count.loc[row["team2"], row["team1"]] += w1

        # drop teams with either no wins or no losses
        no_wl = count.sum(axis=0) * count.sum(axis=1)
        wl_teams = no_wl[no_wl > 0].index
        self.teams = wl_teams.tolist()
        count = count.loc[wl_teams, wl_teams]

        self.count = count

    def fit(self):
        print("Solve stationary distribution.")
        # normalize rows to 1
        self.count += 1e-3 # add eps to prevent singular matrix
        trans = self.count / self.count.sum(axis=1)
        
        A = trans - np.eye(trans.shape[0])
        A.iloc[:,-1] = 1
        
        stat = np.linalg.inv(A)[-1,:]
        self.stationary = pd.Series(stat, index=self.teams, name="rank")

    def predict(self, team1, team2, odds=True):
        p1 = self.stationary[team1]
        p2 = self.stationary[team2]

        return p1 / p2 if odds else p1 / (p1 + p2)
