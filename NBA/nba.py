import pandas
#   inflating: data/games.csv          
#   inflating: data/games_details.csv  
#   inflating: data/players.csv        
#   inflating: data/ranking.csv        
#   inflating: data/teams.csv   
data  = pandas.read_csv('./data/games.csv')
print(data.columns)
data  = pandas.read_csv('./data/players.csv')
print(data.columns)
data  = pandas.read_csv('./data/teams.csv')
print(data.columns)
data  = pandas.read_csv('./data/games_details.csv')
print(data.columns)