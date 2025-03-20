import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

performance_metrics = [
    'd_zone_shift_starts', 'giveaways', 'goals', 'high_danger_goals', 'high_danger_shots', 'hits',
    'low_danger_goals', 'low_danger_shots', 'medium_danger_goals', 'medium_danger_shots', 'missed_shots',
    'o_zone_shift_starts', 'penalty_minutes', 'points', 'rebound_goals', 'rebounds', 'shifts', 'shot_attempts',
    'shots_on_goal', 'takeaways', 'faceoffs_lost', 'faceoffs_won', 'games_played', 'icetime','on_ice_corsi_percentage',
    'on_ice_fenwick_percentage', 'penalties_drawn', 'shots_blocked_by_player', 'assists'
]

def preprocess_player_data(df):
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

def compute_player_scores(team_df, loadings):
    player_scores = {}
    for index, player in team_df.iterrows():
        score = 0
        for feature, weight in loadings.items():
            if feature in player:
                score += player[feature] * weight
        player_scores[player['name']] = score
    return player_scores

def compute_individual_contributions(team_df, loadings, metrics):
    contributions = {metric: {} for metric in metrics}
    for index, player in team_df.iterrows():
        for metric in metrics:
            if metric in player:
                contributions[metric][player['name']] = player[metric] * loadings[metric]
    return contributions

def simulate_game(user_team_filepath, ai_team_filepath, loadings_filepath, scaling_factor=1000):
    # Load the feature loadings
    loadings_dict = load_feature_loadings(loadings_filepath)

    # Load user and AI teams
    user_team = pd.read_csv(user_team_filepath)
    ai_team = pd.read_csv(ai_team_filepath)

    # Compute the scores
    user_player_scores = compute_player_scores(user_team, loadings_dict)
    ai_player_scores = compute_player_scores(ai_team, loadings_dict)

    # Compute individual contributions for all metrics
    user_contributions = compute_individual_contributions(user_team, loadings_dict, performance_metrics)
    ai_contributions = compute_individual_contributions(ai_team, loadings_dict, performance_metrics)

    # Sum the scores to get the team scores
    user_score = sum(user_player_scores.values())
    ai_score = sum(ai_player_scores.values())

    # Normalize the scores
    normalized_user_score = user_score / scaling_factor
    normalized_ai_score = ai_score / scaling_factor

    # Round the scores
    final_user_score = round(normalized_user_score)
    final_ai_score = round(normalized_ai_score)

    # Ensure the winner's score is rounded up if necessary
    if final_user_score == final_ai_score:
        if user_score > ai_score:
            final_user_score += 1
        else:
            final_ai_score += 1

    return final_user_score, final_ai_score, user_player_scores, ai_player_scores, user_contributions, ai_contributions

def plot_player_scores(user_player_scores, ai_player_scores, user_contributions, ai_contributions):
    user_names = list(user_player_scores.keys())
    user_scores = list(user_player_scores.values())
    ai_names = list(ai_player_scores.keys())
    ai_scores = list(ai_player_scores.values())

    # Ensure the names and scores match in length
    all_names = sorted(set(user_names + ai_names))
    user_scores = [user_player_scores.get(name, 0) for name in all_names]
    ai_scores = [ai_player_scores.get(name, 0) for name in all_names]

    fig, ax = plt.subplots(6, 6, figsize=(20, 20))

    ax[0, 0].barh(all_names, user_scores, color='blue', label='User Team')
    ax[0, 0].barh(all_names, ai_scores, color='red', label='AI Team', left=user_scores)
    ax[0, 0].set_xlabel('Score')
    ax[0, 0].set_title('Player Contributions to Team Scores')
    ax[0, 0].legend()

    for i, metric in enumerate(performance_metrics):
        row, col = divmod(i + 1, 6)
        user_metric_scores = [user_contributions[metric].get(name, 0) for name in all_names]
        ai_metric_scores = [ai_contributions[metric].get(name, 0) for name in all_names]
        ax[row, col].barh(all_names, user_metric_scores, color='blue', label='User Team')
        ax[row, col].barh(all_names, ai_metric_scores, color='red', label='AI Team', left=user_metric_scores)
        ax[row, col].set_xlabel(f'{metric.capitalize()} Contribution')
        ax[row, col].set_title(f'Player Contributions to {metric.capitalize()}')
        ax[row, col].legend()

    plt.tight_layout()
    plt.show()


def create_results_table(user_player_scores, ai_player_scores, user_contributions, ai_contributions):
    all_names = sorted(set(user_player_scores.keys()).union(ai_player_scores.keys()))
    
    # Create a DataFrame to hold the results
    results_df = pd.DataFrame(index=all_names, columns=['Team', 'Score'] + performance_metrics)
    
    # Fill in the DataFrame with user team data
    for name in user_player_scores.keys():
        results_df.at[name, 'Team'] = 'User'
        results_df.at[name, 'Score'] = user_player_scores[name]
        for metric in performance_metrics:
            results_df.at[name, metric] = user_contributions[metric].get(name, 0)
    
    # Fill in the DataFrame with AI team data
    for name in ai_player_scores.keys():
        results_df.at[name, 'Team'] = 'AI'
        results_df.at[name, 'Score'] = ai_player_scores[name]
        for metric in performance_metrics:
            results_df.at[name, metric] = ai_contributions[metric].get(name, 0)
    
    return results_df


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
    
    final_user_score, final_ai_score, user_player_scores, ai_player_scores, user_contributions, ai_contributions = simulate_game(user_team_filepath, ai_team_filepath, loadings_filepath)
    print(f"Team User: {final_user_score}")
    print(f"Team AI: {final_ai_score}")

    if final_user_score > final_ai_score:
        print("Team User wins!")
    else:
        print("Team AI wins!")

    # Plot player scores
    #plot_player_scores(user_player_scores, ai_player_scores, user_contributions, ai_contributions)
    # Create and display the results table
    results_table = create_results_table(user_player_scores, ai_player_scores, user_contributions, ai_contributions)
    print(results_table)