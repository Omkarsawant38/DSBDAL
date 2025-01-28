import pandas as pd
data = {
    "Match 1": [45, 60, 20, 80, 55],
    "Match 2": [30, 50, 40, 90, 65],
    "Match 3": [70, 20, 35, 60, 40],
    "Match 4": [25, 45, 15, 100, 75],
}
players = ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"]
performance = pd.DataFrame(data, index=players)
print(performance)
# print(performance.head(3))
print(performance.shape)  
print(performance.columns) 
print(performance.index)  
print(performance.dtypes)
print(performance.isnull().sum())
print(performance.loc["Player 4"])
print(performance["Match 2"])
print(performance.loc[["Player 1", "Player 3"], ["Match 1", "Match 4"]])
print(performance.iloc[:3, :2])
print(performance.loc["Player 5", "Match 3"])
performance.loc["Player 2", "Match 4"] = 50
performance.loc["Player 6"] = [50, 40, 60, 70]
performance += 10
performance.loc["Player 3"] -= 5
percentage_contribution = performance / 120 * 100
print(percentage_contribution)
total_runs = performance.sum(axis=1)
print(total_runs)
total_runs_per_match = performance.sum(axis=0)
print(total_runs_per_match)
max_runs_player = total_runs.idxmax()
print(max_runs_player)
min_runs_match = total_runs_per_match.idxmin()
print(min_runs_match)
average_runs = performance.mean(axis=1)
print(average_runs)
performance["Total Runs"] = total_runs
performance_sorted = performance.sort_values(by="Total Runs", ascending=False)
print(performance_sorted)
players_above_200 = performance[performance["Total Runs"] > 200]
print(players_above_200)
