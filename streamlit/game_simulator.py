import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def preprocess_player_data(df):
    performance_metrics = [
        'd_zone_shift_starts', 'giveaways', 'goals', 'high_danger_goals', 'high_danger_shots', 'hits',
        'low_danger_goals', 'low_danger_shots', 'medium_danger_goals', 'medium_danger_shots', 'missed_shots',
        'o_zone_shift_starts', 'penalty_minutes', 'points', 'rebound_goals', 'rebounds', 'shifts', 'shot_attempts',
        'shots_on_goal', 'takeaways', 'faceoffs_lost', 'faceoffs_won', 'games_played', 'icetime','on_ice_corsi_percentage',
        'on_ice_fenwick_percentage', 'penalties_drawn', 'shots_blocked_by_player', 'assists'
    ]
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df[performance_metrics]), columns=performance_metrics)
    scaled_df['name'] = df['name'].values
    scaled_df['position'] = df['position'].values
    scaled_df['salary'] = df['salary'].values
    return scaled_df, scaler

def generate_feature_loadings(df, features, filepath):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    
    loadings = pca.components_[0]
    loadings_df = pd.DataFrame({
        'feature': features,
        'loading': loadings
    })
    
    loadings_df.to_csv(filepath, index=False)
    return loadings_df

def load_feature_loadings(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Please generate the feature loadings first.")
    loadings_df = pd.read_csv(filepath)
    loadings_dict = dict(zip(loadings_df['feature'], loadings_df['loading']))
    return loadings_dict

def compute_team_score(team_df, loadings):
    score = 0
    for feature, weight in loadings.items():
        if feature in team_df.columns:
            feature_sum = team_df[feature].sum()
            weighted_sum = feature_sum * weight
            score += weighted_sum
    return score

def simulate_game(user_team_filepath, ai_team_filepath, loadings_filepath, scaling_factor=1000):
    # Load the feature loadings
    loadings_dict = load_feature_loadings(loadings_filepath)

    # Load user and AI teams
    user_team = pd.read_csv(user_team_filepath)
    ai_team = pd.read_csv(ai_team_filepath)

    # Compute the scores
    user_score = compute_team_score(user_team, loadings_dict)
    ai_score = compute_team_score(ai_team, loadings_dict)

    # Normalize the scores
    normalized_user_score = user_score / scaling_factor
    normalized_ai_score = ai_score / scaling_factor

    # Round the scores
    if normalized_user_score > normalized_ai_score:
        final_user_score = round(normalized_user_score)
        final_ai_score = round(normalized_ai_score)
    else:
        final_user_score = round(normalized_user_score)
        final_ai_score = round(normalized_ai_score)

    # Ensure the winner's score is rounded up if necessary
    if final_user_score == final_ai_score:
        if user_score > ai_score:
            final_user_score += 1
        else:
            final_ai_score += 1

    return final_user_score, final_ai_score

if __name__ == "__main__":
    user_team_filepath = 'user_team.csv'
    ai_team_filepath = 'ai_team.csv'
    loadings_filepath = 'feature_loadings.csv'
    
    # Generate feature loadings if the file doesn't exist
    if not os.path.exists(loadings_filepath):
        # Load your data
        forwards = pd.read_csv('path_to_forwards.csv')
        defense = pd.read_csv('path_to_defense.csv')
        players = pd.concat([forwards, defense], ignore_index=True)
        
        # List of features
        features = ['d_zone_shift_starts', 'giveaways', 'goals', 'high_danger_goals', 'high_danger_shots', 'hits',
                    'low_danger_goals', 'low_danger_shots', 'medium_danger_goals', 'medium_danger_shots', 'missed_shots',
                    'o_zone_shift_starts', 'penalty_minutes', 'points', 'rebound_goals', 'rebounds', 'shifts', 'shot_attempts',
                    'shots_on_goal', 'takeaways', 'faceoffs_lost', 'faceoffs_won', 'games_played', 'icetime','on_ice_corsi_percentage',
                    'on_ice_fenwick_percentage', 'penalties_drawn', 'shots_blocked_by_player', 'assists']
        
        generate_feature_loadings(players, features, loadings_filepath)
    
    final_user_score, final_ai_score = simulate_game(user_team_filepath, ai_team_filepath, loadings_filepath)
    print(f"Team User: {final_user_score}")
    print(f"Team AI: {final_ai_score}")

    if final_user_score > final_ai_score:
        print("Team User wins!")
    else:
        print("Team AI wins!")