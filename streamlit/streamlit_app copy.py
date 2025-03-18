import sys
import os
import time
import json
sys.path.append(os.path.abspath('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/python_files'))

import streamlit as st
import pandas as pd
from recommender import preprocess_data, find_closest_defense, find_closest_forward, find_closest_goalie
import streamlit.components.v1 as components

# Load player data (make sure CSVs contain "name", "position", and "salary" columns)
defense_data = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/files/master_copies/defense_rec_two.csv', index_col=0)
forward_data = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/files/master_copies/forwards_rec_two.csv', index_col=0)
goalie_data = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/files/master_copies/goalies_rec_two.csv', index_col=0)

# Combine all player data into a single DataFrame
all_players_data = pd.concat([defense_data, forward_data, goalie_data])

# Preprocess data
scaled_defense_df, defense_scaler = preprocess_data(defense_data)
scaled_forward_df, forward_scaler = preprocess_data(forward_data)
scaled_goalie_df, goalie_scaler = preprocess_data(goalie_data)

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

# Team composition requirements
MAX_FORWARDS = 3
MAX_DEFENSE = 2
MAX_GOALIE = 1

st.title('Player Recommender')

player_type = st.selectbox('Select player type:', ['Defense', 'Forward', 'Goalie'])
target_player_name = st.text_input('Enter the name of the target player:')

def make_clickable(name):
    return f'<a href="https://puckpedia.com/player/{name.replace(" ", "-")}" class="pp-player">{name}</a>'

if st.button('Find Similar Players'):
    # Validate target player's name.
    if target_player_name == "" or target_player_name not in all_players_data['name'].tolist():
        st.warning("Please enter a valid target player name from the data.")
    else:
        if player_type == 'Defense':
            closest_players = find_closest_defense(target_player_name, scaled_defense_df)
        elif player_type == 'Forward':
            closest_players = find_closest_forward(target_player_name, scaled_forward_df)
        elif player_type == 'Goalie':
            closest_players = find_closest_goalie(target_player_name, scaled_goalie_df)
        
        st.write(f'Because you like {target_player_name}, you might also enjoy these similar players:')
        original_names = [target_player_name] + closest_players['name'].tolist()
        st.session_state.closest_options = original_names  # Save for later use
        
        # Convert names to clickable links for display purposes
        closest_players['name'] = closest_players['name'].apply(make_clickable)
        closest_players_html = closest_players.to_html(escape=False, index=False)
        st.write(closest_players_html, unsafe_allow_html=True)


# Layout for user-selected team and AI-generated team
col1, col2 = st.columns(2)

with col1:
    # Display Remaining Salary Cap alongside the header.
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
                position = player_row.iloc[0]['position']
                salary = player_row.iloc[0]['salary']
                
                # Count current team composition
                count_forwards = sum(1 for p in st.session_state.selected_players if p['position'] == 'F')
                count_defense  = sum(1 for p in st.session_state.selected_players if p['position'] == 'D')
                count_goalie   = sum(1 for p in st.session_state.selected_players if p['position'] == 'G')
                
                # Check team composition rules and salary cap
                if selected_player in [p['name'] for p in st.session_state.selected_players]:
                    st.warning(f"{selected_player} is already on the team.")
                elif position == 'F' and count_forwards >= MAX_FORWARDS:
                    st.warning("The team already has 3 forwards.")
                elif position == 'D' and count_defense >= MAX_DEFENSE:
                    st.warning("The team already has 2 defensemen.")
                elif position == 'G' and count_goalie >= MAX_GOALIE:
                    st.warning("The team already has 1 goalie.")
                elif st.session_state.user_salary_cap - salary < 0:
                    st.warning("Adding this player would exceed your salary cap.")
                else:
                    st.session_state.pending_player = {'name': selected_player, 'position': position, 'salary': salary}
                    st.warning(f"Are you sure you want to add {selected_player} to your team?")

    if st.session_state.pending_player:
        if st.button('Confirm'):
            player = st.session_state.pending_player
            st.session_state.selected_players.append(player)
            st.session_state.user_salary_cap -= player['salary']
            st.session_state.pending_player = None
            st.success(f"Added {player['name']} to your team.")
    
    st.write('Selected Players:')
    selected_players_df = pd.DataFrame(st.session_state.selected_players)
    st.write(selected_players_df)

    # Add remove buttons for each player
    for index, row in selected_players_df.iterrows():
        col1, col2 = st.columns([1, 3])
        if col1.button('Remove', key=f"remove_{index}"):
            st.session_state.user_salary_cap += row['salary']
            st.session_state.selected_players = [p for p in st.session_state.selected_players if p['name'] != row['name']]
            st.experimental_set_query_params()
        col2.write(f"{row['name']} ({row['position']}) - ${row['salary']:,}")

with col2:
    st.header(f'AI-Generated Team (Remaining Cap: ${st.session_state.ai_salary_cap:,})')
    if not st.session_state.ai_generated_team:
        st.session_state.ai_generated_team = [
            {'name': 'AI Player 1', 'position': 'Forward', 'salary': 950000},
            {'name': 'AI Player 2', 'position': 'Forward', 'salary': 2000000},
            {'name': 'AI Player 3', 'position': 'Forward', 'salary': 4000000},
            {'name': 'AI Player 4', 'position': 'Defense', 'salary': 6000000},
            {'name': 'AI Player 5', 'position': 'Defense', 'salary': 8000000},
            {'name': 'AI Player 6', 'position': 'Goalie', 'salary': 10000000}
        ]
    st.write('AI-Generated Players:')
    ai_generated_players_df = pd.DataFrame(st.session_state.ai_generated_team)
    st.write(ai_generated_players_df)

# Debug log in browser console.
components.html(f"""
<script>
    console.log("Selected Players: {json.dumps(st.session_state.selected_players)}");
    console.log("AI-Generated Players: {json.dumps(st.session_state.ai_generated_team)}");
</script>
""", height=0)

# PuckPedia connector for clickable links.
components.html("""
<link rel="stylesheet" href="https://puckpedia.com/connector/styles" />
<script src="https://puckpedia.com/connector/script" defer></script>
<script>
  var ppOptions = {
    targetSelector: ".pp-player",
    scope: "all",
    linkMode: "popup",
    linkTarget: "_blank"
  };
</script>
""", height=0)

# Optional: Custom CSS to widen DataFrame container.
st.markdown(
    """
    <style>
    .stDataFrame div[data-testid="stDataFrameResizable"] {
        width: 100% !important;
    }
    .dataframe tbody tr td {
        min-width: 150px;
    }
    .dataframe thead th {
        min-width: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)