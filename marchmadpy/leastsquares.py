import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import KFold

class LeastSquares:
    def __init__(self, alpha=1e-4):
        self.alpha = alpha

    def prepare_games(self, games):
        self.dat = games.assign(
            team1 = games["team1"].str.replace("'", ""),
            team2 = games["team2"].str.replace("'", "")
        )

        teams = pd.concat([self.dat.team1, self.dat.team2]).unique()
        self.teams = list(teams)
        self.n_teams = len(self.teams)

        self.n_beta = 2 * self.n_teams
        self.mu = (np.log(self.dat.score1).mean() + np.log(self.dat.score2).mean()) / 2

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
    
    def est_beta(self, x, y, gamma):
        # regularized least squares estimation
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        gamma = torch.from_numpy(gamma).cuda()

        return (torch.linalg.inv(x.T @ x + gamma.T @ gamma) @ x.T @ y).cpu().numpy()


    def fit(self, games, boot=False, cv=False):
        self.boot = boot
        self.prepare_games(games)

        # create X matrix of game matchups
        x1, x2 = self.gen_xs(self.dat.team1, self.dat.team2)

        x = np.concat([x1, x2])
        y = np.concat([np.log(self.dat.score1) - self.mu, np.log(self.dat.score2) - self.mu])

        if cv:
            print("CV find alpha")
            kf = KFold(10, shuffle=True)
            alphas = np.logspace(-10, 1, 12)
            losses = np.zeros_like(alphas)
            for i, alpha in tqdm(enumerate(alphas), total=len(alphas)):
                gamma = alpha * np.eye(self.n_beta)
                for train_ix, test_ix in kf.split(x):
                    beta = self.est_beta(x[train_ix], y[train_ix], gamma)
                    y_hat = x[test_ix] @ beta
                    losses[i] += ((y_hat - y[test_ix])**2).sum()
            alpha = alphas[np.argmin(losses)]
            print(f"Best alpha: {alpha}")
        else:
            alpha = self.alpha

        gamma = alpha * np.eye(self.n_beta) # regularization

        # bootstrap estimates
        n_boot = 100 if boot else 1
        self.betas = np.zeros((n_boot, self.n_beta))
        for i in tqdm(range(n_boot)):
            bix = np.random.choice(x.shape[0], x.shape[0], replace=True) if boot else np.arange(x.shape[0])
            x_train = x[bix]
            y_train = y[bix]
            self.betas[i] = self.est_beta(x_train, y_train, gamma)

    def predict(self, team1, team2, odds=False, **kwargs):
        if not isinstance(team1, (list, np.ndarray)):
            team1 = [team1]
            team2 = [team2]
        assert len(team1) == len(team2)

        x1, x2 = self.gen_xs(team1, team2)

        s1 = np.exp(x1 @ self.betas.T + self.mu)
        s2 = np.exp(x2 @ self.betas.T + self.mu)
        if self.boot:
            mask = s1 == s2
            t1_win = np.ma.array(s1 > s2, mask=mask)
            prob_t1_win = t1_win.mean(axis=1).data
        else:
            # fudge probabilities if not boot
            tie = s1[:,0] == s2[:,0]
            t1_win = s1[:,0] > s2[:,0]
            prob_t1_win = np.where(tie, 0.5, np.where(t1_win, 2/3, 1/3))
        
        return {
            "t1_score": s1,
            "t2_score": s2,
            "prob": prob_t1_win / (1 - prob_t1_win) if odds else prob_t1_win,
            "winner": np.where(prob_t1_win > 0.5, team1, team2),
            #"tie": tie
        }
    