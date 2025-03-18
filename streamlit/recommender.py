import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Initialize sets to keep track of recommended players
recommended_defense = set()
recommended_forward = set()
recommended_goalie = set()

def preprocess_data(player_data):
    # Standardize the data excluding 'name', 'position', and 'salary'
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(player_data.drop(columns=['name', 'position', 'salary']))

    # Create a new DataFrame with the scaled features
    scaled_df = pd.DataFrame(scaled_data, columns=player_data.columns.drop(['name', 'position', 'salary']))
    scaled_df['name'] = player_data['name'].values
    scaled_df['position'] = player_data['position'].values
    scaled_df['salary'] = player_data['salary'].values
    
    return scaled_df, scaler

def find_closest_defense(player_name, df, n=5):
    # Get the feature values for the target player
    target_player = df[df['name'] == player_name].drop(columns=['name', 'position', 'salary'])
    
    # Calculate the Euclidean distances between the target player and all other players
    distances = euclidean_distances(df.drop(columns=['name', 'position', 'salary']), target_player).flatten()
    
    # Get the indices of the n closest players
    closest_indices = np.argsort(distances)[1:n+1]  # Exclude the target player itself
    
    # Get the player names of the closest players
    closest_players = df.iloc[closest_indices][['name', 'position', 'salary']]
    
    # Update the set of recommended players
    recommended_defense.update(closest_players['name'])
    
    return closest_players

def find_closest_forward(player_name, df, n=5):
    # Get the feature values for the target player
    target_player = df[df['name'] == player_name].drop(columns=['name', 'position', 'salary'])
    
    # Calculate the Euclidean distances between the target player and all other players
    distances = euclidean_distances(df.drop(columns=['name', 'position', 'salary']), target_player).flatten()
    
    # Get the indices of the n closest players
    closest_indices = np.argsort(distances)[1:n+1]  # Exclude the target player itself
    
    # Get the player names of the closest players
    closest_players = df.iloc[closest_indices][['name', 'position', 'salary']]
    
    # Update the set of recommended players
    recommended_forward.update(closest_players['name'])
    
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