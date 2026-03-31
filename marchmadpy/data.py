import numpy as np
import pandas as pd


def load_games(data_path="data/scores.csv", latest_only=False):
    games = pd.read_csv(data_path, parse_dates=["date"])
    games["date"] = pd.to_datetime(games["date"], utc=True)
    games['date'] = games['date'].dt.tz_localize(None)
    games = games.assign(
        team1 = games["team1"].str.replace("'", ""),
        team2 = games["team2"].str.replace("'", ""),
        tie = games["score1"] == games["score2"],
        t1win = (games["score1"] > games["score2"]),
        t2win = lambda x: (~x["t1win"]) * (~x["tie"]),
    )

    # determine season
    season_index, season_year = get_relative_season(games, 'date')
    games['season_index'] = season_index
    games['season_year'] = season_year
    if latest_only:
        games = games.query(f"season_year == {games['season_year'].max()}")
    return games

def get_relative_season(df, date_col):
    # Logic: If month <= 5, the season started the previous year. 
    # If month >= 10, the season started the current year.
    # Otherwise (June-Sept), it is off-season.
    month = df[date_col].dt.month
    year = df[date_col].dt.year
    
    # Calculate the starting year for the season
    season_year = np.select(
        [month >= 10, month <= 5],
        [year + 1, year],
        default=np.nan
    )
    
    # Calculate index relative to the first season in the data
    first_season = np.nanmin(season_year)
    return (season_year - first_season).astype(int), season_year

def load_kaggle(minyear=2022, maxyear=2026):
    raw_dat = pd.read_csv("data/kaggle_m.csv").query(f"(Season >= {minyear}) and (Season <= {maxyear})") 
    raw_dat1 = pd.read_csv("data/kaggle_w.csv").query(f"(Season >= {minyear}) and (Season <= {maxyear})")
    all_dat = pd.concat([raw_dat, raw_dat1])

    all_dat["score1"] = all_dat["WScore"]
    all_dat["score2"] = all_dat["LScore"]
    all_dat = all_dat.assign(
        team1 = list(map(str, all_dat["WTeamID"])),
        team2 = list(map(str, all_dat["LTeamID"])),
        season_index = all_dat["Season"] - all_dat["Season"].min(),
        score1 = all_dat["WScore"],
        score2 = all_dat["LScore"],
        tie = all_dat["WScore"] == all_dat["LScore"],
        t1win = [True for _ in range(len(all_dat))],
        t2win = [False for _ in range(len(all_dat))],
    )
    return all_dat
