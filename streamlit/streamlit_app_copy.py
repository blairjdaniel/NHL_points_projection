import sys
import os
sys.path.append(os.path.abspath('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection'))

import streamlit as st
import pandas as pd
from ai_team_generator import generate_ai_team, preprocess_data
from recommender import find_closest_defense, find_closest_forward
from game_simulator import simulate_game, generate_feature_loadings, preprocess_player_data, create_results_table
import streamlit.components.v1 as components



# Load player data (make sure CSVs contain "name", "position", and "salary" columns)
defense_data = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/master_copies/defense_rec_two.csv', index_col=0)
forward_data = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/master_copies/forwards_rec_two.csv', index_col=0)
# Load active players CSV (contains only active players)
active_players_df = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/master_copies/players_2025.csv')  # update path as needed
active_names = active_players_df["name"].tolist()

# Combine all player data into a single DataFrame
all_players_data = pd.concat([defense_data, forward_data])

# Optional: Filter all_players_data to only include active players
all_players_data = all_players_data[all_players_data['name'].isin(active_names)]

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
    st.session_state.filtered_players = None # Track the filtered players by salary

# Team composition requirements
MAX_FORWARDS = 3
MAX_DEFENSE = 2

st.title('NHL DFS Player Recommender')

# Add a reset button to clear all session state variables
if st.button('Reset All'):
    # Clear all session state variables
    for key in st.session_state.keys():
        del st.session_state[key]
    st.query_params  # Trigger a rerun
    
     

player_type = st.selectbox('Select player type:', ['Defense', 'Forward'])
target_player_name = st.text_input('Enter the name of the target player:')
salary_limit = st.number_input('Enter maximum salary:', min_value=0, value=0)

def make_clickable(name):
    return f'<a href="https://puckpedia.com/player/{name.replace(" ", "-")}" class="pp-player">{name}</a>'

if st.button('Find Similar Players'):
     # Validate target player's name: must be in all_players_data and in active_names
    if target_player_name == "" or target_player_name not in all_players_data['name'].tolist():
        st.warning("Please enter a valid target player name from the data.")
    elif target_player_name not in active_names:
        st.warning(f"{target_player_name} is not active.")
    else:
        # Validate target player's name.
        if target_player_name == "" or target_player_name not in all_players_data['name'].tolist():
            st.warning("Please enter a valid target player name from the data.")
        else:
            # Retrieve the target player's salary
            target_player_row = all_players_data[all_players_data['name'] == target_player_name]
            if not target_player_row.empty:
                target_player_salary = target_player_row.iloc[0]['salary']
                st.write(f"{target_player_name}'s Salary: ${target_player_salary:,}")
        
            if player_type == 'Defense':
                closest_players = find_closest_defense(target_player_name, scaled_defense_df)
            elif player_type == 'Forward':
                closest_players = find_closest_forward(target_player_name, scaled_forward_df)
            
            if closest_players.empty:
                st.warning(f"No similar players found for {target_player_name}.")
            else:
                st.write(f'Because you like {target_player_name}, you might also enjoy these similar players:')
                original_names = [target_player_name] + closest_players['name'].tolist()
                st.session_state.closest_options = original_names  # Save for later use
                
                # Convert names to clickable links for display purposes
                closest_players['name'] = closest_players['name'].apply(make_clickable)

                # Select the columns to display
                display_columns = ['name', 'position', 'salary']
                closest_players_html = closest_players[display_columns].to_html(escape=False, index=False)

                #closest_players_html = closest_players.to_html(escape=False, index=False)
                st.write(closest_players_html, unsafe_allow_html=True)

if st.button('Filter Players by Salary'):
    if salary_limit > 0:
        filtered_players = all_players_data[all_players_data['salary'] <= salary_limit]
        if filtered_players.empty:
            st.warning(f"No players found with salary under ${salary_limit:,}.")
        else:
            st.session_state.filtered_players = filtered_players
            #st.write(f'Players with salary under ${salary_limit:,}:')
            filtered_players['name'] = filtered_players['name'].apply(make_clickable)
            display_columns = ['name', 'position', 'salary']
            filtered_players_html = filtered_players[display_columns].to_html(escape=False, index=False)
            #st.write(filtered_players_html, unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid salary limit.")    

    # Create a close button
    col1, col2 = st.columns([10, 1])
    with col1:
        st.markdown(
            f"""
            <div style="overflow-y: scroll; height: 300px;">
                {filtered_players_html}
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        if st.button('X'):
            st.session_state.filtered_players = None

            
# Layout for user-selected team and AI-generated team
st.markdown(
    """
    <style>
      /* Flex container for the two team sections */
      .team-container {
          display: flex;
          flex-direction: row;
          justify-content: space-between;
          align-items: flex-start;
      }
      .team-section {
          width: 48%;
          padding: 10px;
          box-sizing: border-box;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="team-container">', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="team-section">', unsafe_allow_html=True)
    # User-selected team section (col1 code)
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
                
                # Check team composition rules and salary cap
                if selected_player in [p['name'] for p in st.session_state.selected_players]:
                    st.warning(f"{selected_player} is already on the team.")
                elif position == 'F' and count_forwards >= MAX_FORWARDS:
                    st.warning("The team already has 3 forwards.")
                elif position == 'D' and count_defense >= MAX_DEFENSE:
                    st.warning("The team already has 2 defensemen.")
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
            if st.session_state.first_selected_player is None:
                st.session_state.first_selected_player = player
    
    st.write('Selected Players:')
    selected_players_df = pd.DataFrame(st.session_state.selected_players)
    st.write(selected_players_df)

    # Add remove buttons for each player
    for index, row in selected_players_df.iterrows():
        col1, col2 = st.columns([1, 4])
        if col1.button('Remove', key=f"remove_{index}"):
            st.session_state.user_salary_cap += row['salary']
            st.session_state.selected_players = [p for p in st.session_state.selected_players if p['name'] != row['name']]
            st.query_params()  # Trigger a rerun
        col2.write(f"{row['name']} ({row['position']}) - ${row['salary']:,}")

    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="team-section">', unsafe_allow_html=True)
    # AI-generated team section (col2 code)
    st.header(f'AI-Generated Team (Remaining Cap: ${st.session_state.ai_salary_cap:,})')
    if st.button('Generate AI Team'):
        if st.session_state.first_selected_player:
            st.session_state.ai_generated_team, st.session_state.ai_salary_cap = generate_ai_team(
                all_players_data, defense_data, forward_data, st.session_state.first_selected_player
            )
        else:
            st.warning("Please select a player for your team first.")
    # Display AI-generated team table (subset to name, position, salary)
    ai_generated_players_df = pd.DataFrame(st.session_state.ai_generated_team)
    if not ai_generated_players_df.empty:
        ai_generated_players_df = ai_generated_players_df[['name', 'position', 'salary']]
    st.write('AI-Generated Players:')
    st.write(ai_generated_players_df)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


user_team_filepath = '/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/team_data/user_team.csv'
ai_team_filepath = '/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/team_data/ai_team.csv'
loadings_filepath = '/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/master_copies/feature_loadings.csv'
# Simulate the game
if st.button('Simulate Game'):
    # Save the current teams to CSV files
    user_team_df = pd.DataFrame(st.session_state.selected_players)
    ai_team_df = pd.DataFrame(st.session_state.ai_generated_team)

    #  # Add a unique identifier to the player names to distinguish between user and AI teams
    # user_team_df['name'] = 'User_' + user_team_df['name']
    # ai_team_df['name'] = 'AI_' + ai_team_df['name']

      # Merge with all_players_data to include all performance metrics
    user_team_df = user_team_df.merge(all_players_data, on=['name', 'position', 'salary'], how='left')
    ai_team_df = ai_team_df.merge(all_players_data, on=['name', 'position', 'salary'], how='left')
    
    # Drop duplicate columns
    user_team_df = user_team_df.loc[:, ~user_team_df.columns.str.endswith('_y')]
    user_team_df.columns = user_team_df.columns.str.replace('_x', '', regex=False)
    
    ai_team_df = ai_team_df.loc[:, ~ai_team_df.columns.str.endswith('_y')]
    ai_team_df.columns = ai_team_df.columns.str.replace('_x', '', regex=False)  

    # Drop unnecessary columns
    user_team_df = user_team_df.drop(columns=['Unnamed: 0', 'distances'], errors='ignore')
    ai_team_df = ai_team_df.drop(columns=['Unnamed: 0', 'distances'], errors='ignore')

    # Fill NaN values with 0
    user_team_df = user_team_df.fillna(0)
    ai_team_df = ai_team_df.fillna(0)

    user_team_df.to_csv(user_team_filepath, index=False)
    ai_team_df.to_csv(ai_team_filepath, index=False)
    
    final_user_score, final_ai_score, user_player_scores, ai_player_scores, user_contributions, ai_contributions = simulate_game(user_team_filepath, ai_team_filepath, loadings_filepath)
    
    st.write(f"Team User: {final_user_score}")
    st.write(f"Team AI: {final_ai_score}")

    if final_user_score > final_ai_score:
        st.success("Team User wins!")
    else:
        st.success("Team AI wins!")

    # Create and display the results table
    results_table = create_results_table(user_player_scores, ai_player_scores, user_contributions, ai_contributions)
    st.write(results_table)

    # Plot player scores
    #plot_player_scores(user_player_scores, ai_player_scores, user_contributions, ai_contributions)

# #Debug log in browser console.
# components.html(f"""
# <script>
#     console.log("Selected Players: {json.dumps(st.session_state.selected_players)}");
# #    console.log("AI-Generated Players: {json.dumps(st.session_state.ai_generated_team)}");
# </script>
# """, height=0)

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