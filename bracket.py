#%%
import pandas as pd
import pickle
from marchmadpy.empirical import EmpiricalModel
from marchmadpy.markov import MarkovModel
from marchmadpy.poisson import PoissonModel


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
    "South": {
        1: "Auburn",
        2: "Michigan St.",
        3: "Iowa St.",
        4: "Texas A&M",
        5: "Michigan",
        6: "Ole Miss",
        7: "Marquette",
        8: "Louisville",
        9: "Creighton",
        10: "New Mexico",
        11: "San Diego St.",
        12: "UC San Diego",
        13: "Yale",
        14: "Lipscomb",
        15: "Bryant",
        16: "Alabama St."
    },
    "East": {
        1: "Duke",
        2: "Alabama",
        3: "Wisconsin",
        4: "Arizona",
        5: "Oregon",
        6: "BYU",
        7: "Saint Marys (CA)",
        8: "Mississippi St.",
        9: "Baylor",
        10: "Vanderbilt",
        11: "VCU",
        12: "Liberty",
        13: "Akron",
        14: "Montana",
        15: "Robert Morris",
        16: "American"
    },
    "Midwest": {
        1: "Houston",
        2: "Tennessee",
        3: "Kentucky",
        4: "Purdue",
        5: "Clemson",
        6: "Illinois",
        7: "UCLA",
        8: "Gonzaga",
        9: "Georgia",
        10: "Utah St.",
        11: "Texas",
        12: "McNeese",
        13: "High Point",
        14: "Troy",
        15: "Wofford",
        16: "SIUE"
    },
    "West": {
        1: "Florida",
        2: "St. Johns (NY)",
        3: "Texas Tech",
        4: "Maryland",
        5: "Memphis",
        6: "Missouri",
        7: "Kansas",
        8: "UConn",
        9: "Oklahoma",
        10: "Arkansas",
        11: "Drake",
        12: "Colorado St.",
        13: "Grand Canyon",
        14: "UNCW",
        15: "Omaha",
        16: "Norfolk St."
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
        with open("model/PoissonModel.pkl", "rb") as f:
            model = pickle.load(f)
    
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

    champ = model.predict(sw_champ, em_champ, proxy=PROXY)["winner"]
    print(f"Winner: {champ}")


if __name__ == "__main__":
    for mtype in ["empirical", "markov", "poisson"]:
        print("======================================================")
        print(f"MODEL: {mtype}\n")
        main(mtype)
        print("======================================================\n")
