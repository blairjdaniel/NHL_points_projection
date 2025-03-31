import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

# Initialize sets to keep track of recommended players
recommended_defense = set()
recommended_forward = set()
recommended_goalie = set()

# Select the features to use for the recommender engine
features = [
    'd_zone_shift_starts',
    'giveaways',
    'goals',
    'high_danger_goals',
    'high_danger_shots',
    'hits',
    'low_danger_goals',
    'low_danger_shots',
    'medium_danger_goals',
    'medium_danger_shots',
    'missed_shots',
    'o_zone_shift_starts',
    'penalty_minutes',
    'points',
    'rebound_goals',
    'rebounds',
    'shifts',
    'shot_attempts',
    'shots_on_goal',
    'takeaways',
    'faceoffs_lost',
    'faceoffs_won',
    'icetime',
    'on_ice_corsi_percentage',
    'on_ice_fenwick_percentage',
    'penalties_drawn',
    'shots_blocked_by_player',
    'assists',  
]

def preprocess_data(player_data):

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(player_data[features])

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)  # Adjust the number of components as needed
    pca_features = pca.fit_transform(scaled_data)

    # Create a new DataFrame with the PCA features
    pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca_features.shape[1])])
    pca_df['name'] = player_data['name'].values
    pca_df['team'] = player_data['team'].values
    pca_df['position_encoded'] = player_data['position_encoded'].values
    pca_df['salary'] = player_data['salary'].values

    # Create a mapping dictionary for position_encoded
    position_mapping = {0: 'D', 1: 'F'}  # Adjust the mapping as needed

    # Apply the mapping to the position_encoded column
    pca_df['position'] = pca_df['position_encoded'].map(position_mapping)

    # Drop the position_encoded column
    pca_df = pca_df.drop(columns=['position_encoded'])
  
    return pca_df, scaler

# Function to find the 5 closest players
def find_closest_defense(player_name, df, n=5):
    # Get the feature values for the target player
    target_player = df[df['name'] == player_name].drop(columns=['name', 'position', 'team', 'salary'])
    
    # Calculate the Euclidean distances between the target player and all other players
    distances = euclidean_distances(df.drop(columns=['name', 'position', 'salary', 'team']), target_player).flatten()
    
    # Get the indices of the 5 closest players
    closest_indices = np.argsort(distances)[1:n+1]  # Exclude the target player itself
    
    # Get the player names of the closest players
    closest_players = df.iloc[closest_indices][['name', 'position', 'team', 'salary']]
    
    return closest_players



# Function to find the 5 closest players
def find_closest_forwards(player_name, df, n=5):
    # Get the feature values for the target player
    target_player = df[df['name'] == player_name].drop(columns=['name', 'position', 'salary', 'team'])
    
    # Calculate the Euclidean distances between the target player and all other players
    distances = euclidean_distances(df.drop(columns=['name', 'position', 'salary', 'team']), target_player).flatten()
    
    # Get the indices of the 5 closest players
    closest_indices = np.argsort(distances)[1:n+1]  # Exclude the target player itself
    
    # Get the player names of the closest players
    closest_players = df.iloc[closest_indices][['name', 'position','team', 'salary']]
    
    return closest_players

def find_closest_goalie(player_name, df, n=5):
    # Get the feature values for the target player
    target_player = df[df['name'] == player_name].drop(columns=['name', 'position', 'salary'])
    
    # Calculate the Euclidean distances between the target player and all other players
    distances = euclidean_distances(df.drop(columns=['name', 'position', 'salary']), target_player).flatten()
    
    # Get the indices of the n closest players
    closest_indices = np.argsort(distances)[1:n+1]  # Exclude the target player itself
    
    # Get the player names of the closest players
    closest_players = df.iloc[closest_indices][['name', 'position', 'salary']]
    
    # Update the set of recommended players
    recommended_goalie.update(closest_players['name'])
    
    return closest_players