import sys
import os
sys.path.append(os.path.abspath('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection'))

import streamlit as st
import pandas as pd
from ai_team_generator import generate_ai_team
from recommender import find_closest_defense, find_closest_forward, preprocess_data
from game_simulator import simulate_game, generate_feature_loadings, preprocess_player_data, create_results_table
import streamlit.components.v1 as components

# Load all player data from a single CSV
all_players_data = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/master_copies/player_team_2025.csv')

# Map 'position_encoded' to 'position'
all_players_data['position'] = all_players_data['position_encoded'].map({0: 'D', 1: 'F'})

# Preprocess data for defense and forward players
defense_data = all_players_data[all_players_data['position'] == 'D']
forward_data = all_players_data[all_players_data['position'] == 'F']

# Preprocess data
scaled_defense_df, defense_scaler = preprocess_data(defense_data)
scaled_forward_df, forward_scaler = preprocess_data(forward_data)

# Generate feature loadings if the file doesn't exist
loadings_filepath = '/Users/blairjdaniel/lighthouse/lighthouse/NHL/files/master_copies/feature_loadings.csv'
if not os.path.exists(loadings_filepath):
    generate_feature_loadings(all_players_data, scaled_defense_df.columns[:-3], loadings_filepath)

# Initialize session state for selected players, closest options, AI team, & salary caps
if 'selected_players' not in st.session_state:
    st.session_state.selected_players = []
if 'closest_options' not in st.session_state:
    st.session_state.closest_options = []  # This will be used if "Find Similar Players" is clicked
if 'ai_generated_team' not in st.session_state:
    st.session_state.ai_generated_team = []  # Placeholder for AI-generated team
if 'user_salary_cap' not in st.session_state:
    st.session_state.user_salary_cap = 30000000  # $30,000,000 for the user
if 'ai_salary_cap' not in st.session_state:
    st.session_state.ai_salary_cap = 30000000  # $30,000,000 for the AI
if 'pending_player' not in st.session_state:
    st.session_state.pending_player = None  # Track the player pending confirmation
if 'first_selected_player' not in st.session_state:
    st.session_state.first_selected_player = None  # Track the first player selected by the user
if 'filtered_players' not in st.session_state:
    st.session_state.filtered_players = None  # Track the filtered players by salary

# Team composition requirements
MAX_FORWARDS = 3
MAX_DEFENSE = 2

st.title('NHL Player Recommender')

# Add a reset button to clear all session state variables
if st.button('Reset All'):
    # Clear all session state variables
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

player_type = st.selectbox('Select player type:', ['Defense', 'Forward'])
target_player_name = st.text_input('Enter the name of the target player:')
salary_limit = st.number_input('Enter maximum salary:', min_value=0, value=0)

def make_clickable(name):
    return f'<a href="https://puckpedia.com/player/{name.replace(" ", "-")}" class="pp-player">{name}</a>'

if st.button('Find Similar Players'):
    # Validate target player's name.
    if target_player_name == "" or target_player_name not in all_players_data['name'].tolist():
        st.warning("Please enter a valid target player name from the data.")
    else:
        # Retrieve the target player's salary
        target_player_row = all_players_data[all_players_data['name'] == target_player_name]
        if not target_player_row.empty:
            target_player_salary = target_player_row.iloc[0]['salary']
            st.write(f"{target_player_name}'s Salary: ${target_player_salary:,}")
        
        # Filter the data based on the player type
        if player_type == 'Defense':
            filtered_data = all_players_data[all_players_data['position'] == 'D']
            closest_players = find_closest_defense(target_player_name, filtered_data)
        elif player_type == 'Forward':
            filtered_data = all_players_data[all_players_data['position'] == 'F']
            closest_players = find_closest_forward(target_player_name, filtered_data)
        
        if closest_players.empty:
            st.warning(f"No similar players found for {target_player_name}.")
        else:
            st.write(f'Because you like {target_player_name}, you might also enjoy these similar players:')
            original_names = [target_player_name] + closest_players['name'].tolist()
            st.session_state.closest_options = original_names  # Save for later use
            
            # Convert names to clickable links for display purposes
            closest_players['name'] = closest_players['name'].apply(make_clickable)

            # Select the columns to display
            display_columns = ['name', 'position', 'team', 'salary']
            closest_players_html = closest_players[display_columns].to_html(escape=False, index=False)
            st.write(closest_players_html, unsafe_allow_html=True)

if st.button('Filter Players by Salary'):
    if salary_limit > 0:
        filtered_players = all_players_data[all_players_data['salary'] <= salary_limit]
        if filtered_players.empty:
            st.warning(f"No players found with salary under ${salary_limit:,}.")
        else:
            st.session_state.filtered_players = filtered_players
            filtered_players['name'] = filtered_players['name'].apply(make_clickable)
            display_columns = ['name', 'position', 'team', 'salary']
            filtered_players_html = filtered_players[display_columns].to_html(escape=False, index=False)
            st.write(filtered_players_html, unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid salary limit.")

# Layout for user-selected team and AI-generated team
st.markdown('<div class="team-container">', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="team-section">', unsafe_allow_html=True)
    st.header(f'Your Team (Remaining Cap: ${st.session_state.user_salary_cap:,})')
    
    # Form to add a player.
    with st.form(key='player_form'):
        if st.session_state.closest_options:
            options = st.session_state.closest_options
        else:
            options = all_players_data['name'].tolist()
        
        selected_player = st.selectbox('Select a player to add to your team:', options)
        submit_button = st.form_submit_button(label='Add Selected Player')
        
        if submit_button:
            # Lookup player's info
            player_row = all_players_data[all_players_data['name'] == selected_player]
            if player_row.empty:
                st.error("Player not found in data.")
            else:
                position_encoded = player_row.iloc[0]['position_encoded']
                salary = player_row.iloc[0]['salary']
                team = player_row.iloc[0]['team']
                
                # Count current team composition
                count_forwards = sum(1 for p in st.session_state.selected_players if p['position_encoded'] == 1)
                count_defense = sum(1 for p in st.session_state.selected_players if p['position_encoded'] == 0)
                
                # Check team composition rules and salary cap
                if selected_player in [p['name'] for p in st.session_state.selected_players]:
                    st.warning(f"{selected_player} is already on the team.")
                elif position_encoded == 1 and count_forwards >= MAX_FORWARDS:
                    st.warning("The team already has 3 forwards.")
                elif position_encoded == 0 and count_defense >= MAX_DEFENSE:
                    st.warning("The team already has 2 defensemen.")
                elif st.session_state.user_salary_cap - salary < 0:
                    st.warning("Adding this player would exceed your salary cap.")
                else:
                    st.session_state.pending_player = {
                        'name': selected_player,
                        'position_encoded': position_encoded,
                        'team': team,
                        'salary': salary
                    }
                    st.warning(f"Are you sure you want to add {selected_player} to your team?")

    if st.session_state.pending_player:
        if st.button('Confirm'):
            player = st.session_state.pending_player
            st.session_state.selected_players.append(player)
            st.session_state.user_salary_cap -= player['salary']
            st.session_state.pending_player = None
            st.success(f"Added {player['name']} to your team.")
            if st.session_state.first_selected_player is None:
                st.session_state.first_selected_player = player
    
st.write('Selected Players:')
selected_players_df = pd.DataFrame(st.session_state.selected_players)

# Map 'position_encoded' to 'position' for display
if 'position_encoded' in selected_players_df.columns:
    selected_players_df['position'] = selected_players_df['position_encoded'].map({0: 'D', 1: 'F'})

# Ensure 'team' and 'salary' columns are included in the table
if 'team' in selected_players_df.columns and 'salary' in selected_players_df.columns:
    selected_players_df = selected_players_df[['name', 'position', 'team', 'salary']]
    st.write(selected_players_df)
else:
    st.warning("Team or salary information is missing for some players.")

# Add remove buttons for each player
for index, row in selected_players_df.iterrows():
    col1, col2 = st.columns([1, 4])
    if col1.button('Remove', key=f"remove_{index}"):
        st.session_state.user_salary_cap += row['salary']
        st.session_state.selected_players = [p for p in st.session_state.selected_players if p['name'] != row['name']]
        st.experimental_rerun()
    col2.write(f"{row['name']} ({row['position']}) - ${row['salary']:,}")

st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="team-section">', unsafe_allow_html=True)
    st.header(f'AI-Generated Team (Remaining Cap: ${st.session_state.ai_salary_cap:,})')
    if st.button('Generate AI Team'):
        if st.session_state.first_selected_player:
            st.session_state.ai_generated_team, st.session_state.ai_salary_cap = generate_ai_team(
                all_players_data, defense_data, forward_data, st.session_state.first_selected_player
            )
        else:
            st.warning("Please select a player for your team first.")
    # Display AI-generated team table
    ai_generated_df = pd.DataFrame(st.session_state.ai_generated_team)
    if not ai_generated_df.empty:
        ai_generated_df['position'] = ai_generated_df['position_encoded'].map({0: 'D', 1: 'F'})
        ai_generated_df = ai_generated_df[['name', 'position', 'team', 'salary']]
        st.write('AI-Generated Players:')
        st.write(ai_generated_df)
    else:
        st.warning("No AI-generated team available.")

st.markdown('</div>', unsafe_allow_html=True)

# Simulate the game
if st.button('Simulate Game'):
    user_team_df = pd.DataFrame(st.session_state.selected_players)
    ai_team_df = pd.DataFrame(st.session_state.ai_generated_team)

    user_team_filepath = '/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/team_data/user_team.csv'
    ai_team_filepath = '/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/team_data/ai_team.csv'

    user_team_df.to_csv(user_team_filepath, index=False)
    ai_team_df.to_csv(ai_team_filepath, index=False)

    final_user_score, final_ai_score, user_player_scores, ai_player_scores, user_contributions, ai_contributions = simulate_game(
        user_team_filepath, ai_team_filepath, loadings_filepath
    )

    st.write(f"Team User: {final_user_score}")
    st.write(f"Team AI: {final_ai_score}")

    if final_user_score > final_ai_score:
        st.success("Team User wins!")
    else:
        st.success("Team AI wins!")

    results_table = create_results_table(user_player_scores, ai_player_scores, user_contributions, ai_contributions)
    st.write(results_table)