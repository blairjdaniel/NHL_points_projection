# This is a Streamlit demo showing off just how easy it is to set up and run a Streamlit app to use our model in a
# more user-friendly way

# Import modules
import streamlit as st
import pandas as pd
import numpy as np
from player_predictions import load_model, create_new_data, run_prediction

# Example DataFrame mapping playerId_all to player names
player_info = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/files/other/name_playerId.csv')

# Define page title configuration (sets the name of the page as well as gives it a logo, note the wide layout)
st.set_page_config(
    page_title='Predict NHL Player',
    page_icon=':hockey:',
    layout='wide'
)

# Write some introductory message
st.write('''
            # NHL Player Prediction
         
            Input the requested parameters to predict the hockey player career goals :hockeynet:
         ''')

# Set up inputs for each of our features for our model
cols = st.columns(9)

with cols[0]:
    I_F_highDangerGoals_all = st.number_input('High Danger Goals (All)', value=0.0)
with cols[1]:
    I_F_highDangerShots_all = st.number_input('High Danger Shots (All)', value=0.0)
with cols[2]:
    I_F_lowDangerGoals_all = st.number_input('Low Danger Goals (All)', value=0.0)
with cols[3]:
    I_F_lowDangerShots_all = st.number_input('Low Danger Shots (All)', value=0.0)
with cols[4]:
    I_F_mediumDangerGoals_all = st.number_input('Medium Danger Goals (All)', value=0.0)
with cols[5]:
    I_F_mediumDangerShots_all = st.number_input('Medium Danger Shots (All)', value=0.0)
with cols[6]:
    I_F_shotAttempts_all = st.number_input('Shot Attempts (All)', value=0.0)
with cols[7]:
    I_F_shotsOnGoal_all = st.number_input('Shots On Goal (All)', value=0.0)
with cols[8]:
    PositionEn_encoded = st.number_input('Position Encoded', value=0)

# Load the model from the pickle file
model = load_model('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/models/skater_gradient_boosting_model.pkl')

# Package the inputs into the appropriate structure
X = create_new_data(
    I_F_highDangerGoals_all,	
    I_F_highDangerShots_all,	
    I_F_lowDangerGoals_all,	
    I_F_lowDangerShots_all,	
    I_F_mediumDangerGoals_all,	
    I_F_mediumDangerShots_all,		
    I_F_shotAttempts_all,	
    I_F_shotsOnGoal_all,	
    PositionEn_encoded
)

# Get our predicted value
y = run_prediction(model, X)
predicted_goals = int(y[0])


# Calculate the absolute difference between the predicted player ID and all player IDs
player_info['difference'] = np.abs(player_info['I_F_goals_all'] - predicted_goals)

# Find the player ID with the smallest difference
closest_playerId = player_info.loc[player_info['difference'].idxmin(), 'playerId_all']
closest_player_name = player_info.loc[player_info['difference'].idxmin(), 'name']

# Display the prediction and closest player's name on our page
st.write(f":hockey: Predicted Player Career Goals: {predicted_goals}")
st.write(f":hockey: Closest Player ID: {closest_playerId}")
st.write(f":hockey: Closest Player Name: {closest_player_name}")