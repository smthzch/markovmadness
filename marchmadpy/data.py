import numpy as np
import pandas as pd


def load_games(data_path="data/scores.csv"):
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
    games['season_index'] = get_relative_season(games, 'date')

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
        [year, year - 1],
        default=np.nan
    )
    
    # Calculate index relative to the first season in the data
    first_season = np.nanmin(season_year)
    return (season_year - first_season).astype(int)
