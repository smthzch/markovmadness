
import typer
import requests
import pandas as pd
import pendulum
from bs4 import BeautifulSoup as bs

BASE_URL = "https://www.ncaa.com/scoreboard/basketball-men/d1/{}"


def get_games_date(date):
    url_date = date.format("YYYY/MM/DD")
    data_date = date.format("YYYY-MM-DD")
    print(f"getting {date}")

    r = requests.get(BASE_URL.format(url_date))
    soup = bs(r.text, "html.parser")
    games = soup.find_all("ul", class_="gamePod-game-teams")

    games_list = []
    for game in games:
        teams = game.find_all("li")
        try:
            t1 = teams[0].find("span", class_="gamePod-game-team-name").text.replace("'", "")
            s1 = int(teams[0].find("span", class_="gamePod-game-team-score").text)
            t2 = teams[1].find("span", class_="gamePod-game-team-name").text.replace("'", "")
            s2 = int(teams[1].find("span", class_="gamePod-game-team-score").text)
            if t1 != "" and t2 != "" and s1 > 0 and s2 > 0:
                games_list.append({
                    "team1": t1,
                    "team2": t2,
                    "score1": s1,
                    "score2": s2,
                    "date": pd.to_datetime(date.isoformat())
                })
        except:
            print(f"error parsing game on: {date}")

    return pd.DataFrame(games_list)


def main(
    start_date: str="2023-11-06", 
    end_date: str=pendulum.now().subtract(days=1).format("YYYY-MM-DD"),
    append: bool=True
):
    out_path = f"data/scores.csv"
    
    # load existing game data if exists
    if append:
        try:
            games = pd.read_csv(out_path, parse_dates=["date"])
            start_date = (
                    games["date"].max() + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d")
        except:
            append = False

    # determine date range of new data to get
    start = pendulum.parse(start_date, tz="America/New_York")
    end = pendulum.parse(end_date, tz="America/New_York")
    assert end >= start, f"{end} < {start}"

    period = pendulum.period(start, end)

    # get new game data
    new_games = [
        get_games_date(dt)
        for dt in period.range("days")
    ]
    new_games = [game for game in new_games if not game.empty]
    new_games = [games] + new_games if append else new_games
   
    # combine and save
    games = pd.concat(new_games)
    games.to_csv(out_path, index=False)

    print("done")

if __name__ == "__main__":
    typer.run(main)
