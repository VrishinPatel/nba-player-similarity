import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Scrape NBA 2022-23 player totals from Basketball Reference
url = "https://www.basketball-reference.com/leagues/NBA_2023_totals.html"
res = requests.get(url)
soup = BeautifulSoup(res.text, "html.parser")
table = soup.find(name="table", attrs={"id": "totals_stats"})

# Convert HTML table to DataFrame
df = pd.read_html(StringIO(str(table)))[0]
df = df[df['Player'] != 'Player']  # remove repeated headers

# Keep only the last entry for players who played on multiple teams
df = df.drop_duplicates(subset='Player', keep='last')
df = df.reset_index(drop=True)

# Select relevant stats
stats_cols = ['Player', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%', 'FT%']
df = df[stats_cols]

# Convert all columns except 'Player' to numeric, coerce errors to NaN
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values (important before scaling)
df = df.dropna().reset_index(drop=True)

# Normalize features
features = df.iloc[:, 1:]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(scaled_features)
sim_df = pd.DataFrame(similarity_matrix, index=df['Player'], columns=df['Player'])

# Query function to get top 10 most similar players
def get_similar_players(player_name):
    if player_name not in sim_df:
        return []
    return sim_df[player_name].sort_values(ascending=False)[1:11]

# Queries
queries = ["Stephen Curry", "Nikola Jokic", "Jimmy Butler"]

# Results
top_similar = {}
for player in queries:
    top_similar[player] = get_similar_players(player)

# Print results
for player, similar in top_similar.items():
    print(f"\nTop 10 players similar to {player}:")
    print(similar)

# Optional: Plot similarity scores for one player
similar_scores = top_similar["Stephen Curry"]
similar_scores.plot(kind='bar', title='Top 10 Similar Players to Stephen Curry')
plt.ylabel('Cosine Similarity')
plt.tight_layout()
plt.show()
