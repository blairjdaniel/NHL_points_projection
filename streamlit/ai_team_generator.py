import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.metrics.pairwise import euclidean_distances

# Define the performance metrics
performance_metrics = [
    'goals', 'high_danger_goals', 'high_danger_shots', 'hits',
    'penalty_minutes', 'points', 'shifts', 
    'shots_on_goal', 'games_played', 
    'assists', 
]

# Function to preprocess data
def preprocess_data(df):
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df[performance_metrics]), columns=performance_metrics)
    scaled_df['name'] = df['name'].values
    scaled_df['team'] = df['team'].values
    scaled_df['position'] = df['position'].values
    scaled_df['salary'] = df['salary'].values
    return scaled_df, scaler

# Function to recommend the next skater based on cosine similarity
def recommend_next_player_skater(selected_player, players_pool, remaining_cap, performance_metrics, reserve=0):
    # Filter candidates by salary
    pos_candidates = players_pool[players_pool['salary'] <= (remaining_cap - reserve)].copy()
    
    if pos_candidates.empty:
        return None, remaining_cap
    
    # # Compute cosine similarity between the selected player and candidates
    # pos_candidates['similarity'] = cosine_similarity(
    #     pos_candidates[performance_metrics], 
    #     selected_player[performance_metrics].values.reshape(1, -1)
    # )[:, 0]

    # Compute the Euclidean Distances to test against cosine similarities
    pos_candidates['distances'] = euclidean_distances(
        pos_candidates[performance_metrics],
        selected_player[performance_metrics].values.reshape(1, -1)
    )[:, 0]
    
    # # Select the candidate with the highest similarity
    # best_candidate = pos_candidates.sort_values(by='similarity', ascending=False).iloc[0]
    # remaining_cap -= best_candidate['salary']

    # Select the candidate with the highest similarity in distances
    best_candidate = pos_candidates.sort_values(by='distances', ascending=True).iloc[0]
    remaining_cap -= best_candidate['salary']
    
    return best_candidate, remaining_cap

# Function to generate AI team
def generate_ai_team(all_players_data, defense_rec, forwards_rec, initial_player):
    
     # If initial_player is a dict, convert it to a full Series by looking it up
    if not isinstance(initial_player, pd.Series):
        initial_player = all_players_data[all_players_data['name'] == initial_player['name']].iloc[0]
        
    selected_player = initial_player

   # Remove the initial player from candidate pools to avoid duplication.
    forwards_rec = forwards_rec[forwards_rec['name'] != selected_player['name']]
    defense_rec = defense_rec[defense_rec['name'] != selected_player['name']]
    
    remaining_cap = 30000000 - selected_player['salary']
    ai_team = [selected_player]
    print(f"AI selected initial player: {selected_player['name']} ({selected_player['position']}) - ${selected_player['salary']:,}")
    
    # Adjust team composition based on the selected player's position
    team_composition = {'F': 3, 'D': 2}
    if selected_player['position'] == 'F':
        team_composition['F'] -= 1
    else:
        team_composition['D'] -= 1
    
    
    # Shuffle the positions to pick players from different positions in a random order
    positions = ['F', 'D']
    random.shuffle(positions)
    
    # Recommend players for each position
    for position in positions:
        count = team_composition[position]
        for _ in range(count):
            # Determine reserve: if this is not the final pick overall, reserve $1,000,000.
            reserve = 1000000 if len(ai_team) < 4 else 0
            
            if position == 'F':
                next_player, remaining_cap = recommend_next_player_skater(selected_player, forwards_rec, remaining_cap, performance_metrics, reserve)
                if next_player is not None:
                    ai_team.append(next_player)
                    forwards_rec.drop(next_player.name, inplace=True)
                    print(f"AI added {next_player['name']} ({next_player['position']}) - ${next_player['salary']:,}")
                else:
                    print(f"No suitable forward candidate found with at least $1,000,000 reserved.")
                    break
            elif position == 'D':
                next_player, remaining_cap = recommend_next_player_skater(selected_player, defense_rec, remaining_cap, performance_metrics, reserve)
                if next_player is not None:
                    ai_team.append(next_player)
                    defense_rec.drop(next_player.name, inplace=True)
                    print(f"AI added {next_player['name']} ({next_player['position']}) - ${next_player['salary']:,}")
                else:
                    print(f"No suitable defense candidate found with at least $1,000,000 reserved.")
                    break
    
    # Final check: must have exactly 5 players (3 F, 2 D)
    if (len([p for p in ai_team if p['position'] == 'F']) == 3 and
        len([p for p in ai_team if p['position'] == 'D']) == 2):
        return ai_team, remaining_cap
    else:
        print("Failed to meet the team composition requirements.")
        return [], remaining_cap