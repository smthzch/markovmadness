#%%
import pandas as pd
import pickle
from marchmadpy.empirical import EmpiricalModel
from marchmadpy.markov import MarkovModel
from marchmadpy.poisson import PoissonModel
from marchmadpy.leastsquares import LeastSquares


PROXY = True

def generate_initial_matchups(teams):
    """
    Generates the first-round matchups based on seeds.
    """
    matchup_tree = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    teams = sorted(teams.items(), key=lambda x: x[1])  # Sort teams by seed
    matchups = [(teams[matchup_tree[i]-1], teams[matchup_tree[i+1]-1]) for i in range(0, len(teams), 2)]
    return matchups

def simulate_game(team1, team2, model):
    """
    Simulates a game between two teams, favoring the lower-seeded team.
    """
    res = model.predict(team1[0], team2[0], proxy=PROXY)
    if res.get("tie", False):
        print(f"\ttie: {team1[0]} V {team2[0]}")
    if res.get("tossup", False):
        print(f"\ttossup: {team1[0]} V {team2[0]}")
    return team1 if res["winner"] == team1[0] else team2

def play_round(matchups, model):
    """
    Plays a round of the tournament and returns the winners.
    """
    winners = [simulate_game(team1, team2, model) for team1, team2 in matchups]
    return winners

def generate_bracket(teams, model):
    """
    Generates the entire bracket given initial seeds and plays through the tournament.
    """
    bracket = []
    round_num = 1
    matchups = generate_initial_matchups(teams)
    
    while len(matchups) > 1:
        print(f"Round {round_num} matchups:")
        for game in matchups:
            print(f"{game[0][0]} (Seed {game[0][1]}) vs {game[1][0]} (Seed {game[1][1]})")
        
        winners = play_round(matchups, model)
        matchups = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        bracket.append(winners)
        round_num += 1
    
    # Championship round
    print(f"Final matchup:")
    game = matchups[0]
    print(f"{game[0][0]} (Seed {game[0][1]}) vs {game[1][0]} (Seed {game[1][1]})")
    champion = play_round(matchups, model)[0]
    print(f"Champion: {champion[0]} (Seed {champion[1]})")
    return champion

bracket = {
    "East": {
        1: "Duke",
        2: "UConn",
        3: "Michigan St.",
        4: "Kansas",
        5: "St. Johns (NY)",
        6: "Louisville",
        7: "UCLA",
        8: "Ohio St.",
        9: "TCU",
        10: "UCF",
        11: "South Fla.",
        12: "UNI",
        13: "California Baptist",
        14: "North Dakota St.",
        15: "Furman",
        16: "Siena"
    },
    "Midwest": {
        1: "Michigan",
        2: "Iowa St.",
        3: "Virginia",
        4: "Alabama",
        5: "Texas Tech",
        6: "Tennessee",
        7: "Kentucky",
        8: "Georgia",
        9: "Saint Louis",
        10: "Santa Clara",
        11: "Miami (OH)", # or SMU
        12: "Akron",
        13: "Hofstra",
        14: "Wright St.",
        15: "Tennessee St.",
        16: "UMBC" # or  Howard
    },
    "South": {
        1: "Florida",
        2: "Houston",
        3: "Illinois",
        4: "Nebraska",
        5: "Vanderbilt",
        6: "North Carolina",
        7: "Saint Marys (CA)",
        8: "Clemson",
        9: "Iowa",
        10: "Texas A&M",
        11: "VCU",
        12: "McNeese",
        13: "Troy",
        14: "Penn",
        15: "Idaho",
        16: "Lehigh" # or Prairie View
    },
    "West": {
        1: "Arizona",
        2: "Purdue",
        3: "Gonzaga",
        4: "Arkansas",
        5: "Wisconsin",
        6: "BYU",
        7: "Miami (FL)",
        8: "Villanova",
        9: "Utah St.",
        10: "Missouri",
        11: "NC State", # or Texas
        12: "High Point",
        13: "Hawaii",
        14: "Kennesaw St.",
        15: "Queens (NC)",
        16: "LIU"
    }
}


def main(mtype):
    data_path = f"data/scores.csv"
    games = pd.read_csv(data_path, parse_dates=["date"])

    if mtype == "empirical":
        model = EmpiricalModel()
        model.fit(games, rank=False)
    elif mtype == "markov":
        model = MarkovModel()
        model.fit(games)
    elif mtype == "poisson":
        #with open("model/PoissonModel.pkl", "rb") as f:
        #    model = pickle.load(f)
        model = PoissonModel()
        model.fit(games)
    elif mtype == "leastsquares":
        model = LeastSquares()
        model.fit(games, boot=True)
    
    # Generate and simulate the NCAA tournament bracket
    region_champs = {}
    for region in bracket:
        print(f"Region: {region}")
        teams = {bracket[region][i]: i for i in range(1, 17)}
        region_champs[region] = generate_bracket(teams, model)[0]
        print("")


    print(region_champs)

    sw_champ = model.predict(region_champs["South"], region_champs["West"], proxy=PROXY)["winner"]
    print(f"S v W winner: {sw_champ}")

    em_champ = model.predict(region_champs["East"], region_champs["Midwest"], proxy=PROXY)["winner"]
    print(f"E v M winner: {em_champ}")

    pred_champ = model.predict(sw_champ, em_champ, proxy=PROXY)
    champ = pred_champ["winner"]
    p_champ = pred_champ["prob"] if champ == sw_champ else 1 - pred_champ["prob"]
    comb_score = (pred_champ["t1_score"] + pred_champ["t2_score"]).mean()
    print(f"Winner: {champ} at {p_champ*100}%, combined score: {comb_score}")


if __name__ == "__main__":
    for mtype in ["poisson"]:# "leastsquares", "empirical", "markov", "poisson"]:
        print("======================================================")
        print(f"MODEL: {mtype}\n")
        main(mtype)
        print("======================================================\n")
