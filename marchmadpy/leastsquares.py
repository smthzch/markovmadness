import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

class LeastSquares:
    def __init__(self, alpha=1, link="log"):
        assert link in ["log", "linear"]
        self.alpha = alpha
        if link == "log":
            self.link = np.log
            self.inv_link = np.exp
        elif link == "linear":
            self.link = lambda x: x
            self.inv_link = lambda x: x
        assert 1.0 == self.inv_link(self.link(1.0))

    def prepare_games(self, games):
        self.dat = games.assign(
            team1 = games["team1"].str.replace("'", ""),
            team2 = games["team2"].str.replace("'", "")
        )

        teams = pd.concat([self.dat.team1, self.dat.team2]).unique()
        self.teams = list(teams)
        self.n_teams = len(self.teams)

        self.n_beta = 2 * self.n_teams
        self.mu = (self.link(self.dat.score1).mean() + self.link(self.dat.score2).mean()) / 2

    def gen_xs(self, team1, team2):
        assert len(team1) == len(team2)
        x1 = np.zeros((len(team1), self.n_beta))
        x2 = np.zeros((len(team2), self.n_beta))

        for i, (t1, t2) in enumerate(zip(team1, team2)):
            t1 = self.teams.index(t1)
            t2 = self.teams.index(t2) 
        
            x1[i, 2 * t1] = 1
            x1[i, 2 * t2 + 1] = -1
            x2[i, 2 * t2] = 1
            x2[i, 2 * t1 + 1] = -1

        return x1, x2
    
    @staticmethod
    def wild_bootstrap(x, y_hat, residuals, gamma):
        weights = np.random.choice([-1, 1], size=len(y_hat))
        y_star = y_hat + residuals * weights
        return LeastSquares.estimate_beta(x, y_star, gamma)
    
    @staticmethod
    def estimate_beta(x, y, gamma):
        # regularized least squares estimation
        return (np.linalg.inv(x.T @ x + gamma) @ x.T @ y)


    def fit(self, games, boot=False, cv=False):
        self.boot = boot
        self.prepare_games(games)

        # create X matrix of game matchups
        x1, x2 = self.gen_xs(self.dat.team1, self.dat.team2)

        x = np.concat([x1, x2])
        y = np.concat([self.link(self.dat.score1) - self.mu, self.link(self.dat.score2) - self.mu])

        if cv:
            print("CV find alpha")
            kf = KFold(5, shuffle=False)
            alphas = np.logspace(-10, 1, 12)
            losses = np.zeros_like(alphas)
            for i, alpha in tqdm(enumerate(alphas), total=len(alphas)):
                gamma = alpha * np.eye(self.n_beta)
                for train_ix, test_ix in kf.split(x):
                    beta = self.estimate_beta(x[train_ix], y[train_ix], gamma)
                    y_hat = x[test_ix] @ beta
                    losses[i] += ((y_hat - y[test_ix])**2).sum()
            alpha = alphas[np.argmin(losses)]
            print(f"Best alpha: {alpha}")
        else:
            alpha = self.alpha

        gamma = alpha * np.eye(self.n_beta) # regularization
        full_beta = self.estimate_beta(x, y, gamma)

        # wild bootstrap estimates
        if boot:
            n_boot = 1000
            y_hat = x @ full_beta
            residuals = y - y_hat
            res = Parallel(n_jobs=20)(delayed(LeastSquares.wild_bootstrap)(x, y_hat, residuals, gamma) for _ in tqdm(range(n_boot)))
            self.betas = np.stack(res, axis=0)
        else:
            self.betas = full_beta[None,:]

    def predict(self, team1, team2, odds=False, **kwargs):
        if not isinstance(team1, (list, np.ndarray)):
            team1 = [team1]
            team2 = [team2]
        assert len(team1) == len(team2)

        x1, x2 = self.gen_xs(team1, team2)

        s1 = self.inv_link(x1 @ self.betas.T + self.mu)
        s2 = self.inv_link(x2 @ self.betas.T + self.mu)
        if self.boot:
            mask = s1 == s2
            t1_win = np.ma.array(s1 > s2, mask=mask)
            prob_t1_win = t1_win.mean(axis=1).data
        else:
            # dummy probabilities if not boot
            tie = s1[:,0] == s2[:,0]
            t1_win = s1[:,0] > s2[:,0]
            prob_t1_win = np.where(tie, 0.5, np.where(t1_win, 2/3, 1/3))
        
        return {
            "t1_score": s1,
            "t2_score": s2,
            "prob": prob_t1_win / (1 - prob_t1_win) if odds else prob_t1_win,
            "winner": np.where(prob_t1_win > 0.5, team1, team2),
        }
    