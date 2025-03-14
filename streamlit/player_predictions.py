import numpy as np
import pandas as pd
import joblib

# Function to load the model
def load_model(model_file):
    '''
    This function loads a model from a pickle file.

    Inputs:
        model_file: string indicating the file location
    Outputs:
        model: the trained model
    '''
    model = joblib.load(model_file)
    return model

# Function to create new data
def create_new_data(
        I_F_highDangerGoals_all,	
        I_F_highDangerShots_all,	
        I_F_lowDangerGoals_all,	
        I_F_lowDangerShots_all,	
        I_F_mediumDangerGoals_all,	
        I_F_mediumDangerShots_all,		
        I_F_shotAttempts_all,	
        I_F_shotsOnGoal_all,	
        PositionEn_encoded):
    '''
    This function accepts each feature value as a separate argument and packages them into
    a dataframe.

    Inputs:
        I_F_highDangerGoals_all: float
        I_F_highDangerShots_all: float
        I_F_lowDangerGoals_all: float
        I_F_lowDangerShots_all: float
        I_F_mediumDangerGoals_all: float
        I_F_mediumDangerShots_all: float
        I_F_shotAttempts_all: float
        I_F_shotsOnGoal_all: float
        PositionEn_encoded: int

    Outputs:
        df_new: data point packaged as a dataframe object
    '''
    # Set up feature columns
    columns = [
        'I_F_highDangerGoals_all',	
        'I_F_highDangerShots_all',	
        'I_F_lowDangerGoals_all',	
        'I_F_lowDangerShots_all',	
        'I_F_mediumDangerGoals_all',	
        'I_F_mediumDangerShots_all',		
        'I_F_shotAttempts_all',	
        'I_F_shotsOnGoal_all',	
        'PositionEn_encoded'
    ]

    # Construct array from inputs
    X_new = np.array([
        I_F_highDangerGoals_all,	
        I_F_highDangerShots_all,	
        I_F_lowDangerGoals_all,	
        I_F_lowDangerShots_all,	
        I_F_mediumDangerGoals_all,	
        I_F_mediumDangerShots_all,		
        I_F_shotAttempts_all,	
        I_F_shotsOnGoal_all,	
        PositionEn_encoded
    ])
        
    # Package data into a dataframe
    df_new = pd.DataFrame(data=X_new.reshape(1, -1), columns=columns)

    return df_new

# Function to run the model and make a prediction
def run_prediction(model, X):
    '''
    This function runs the model to make a prediction.

    Inputs:
        model: trained model
        X: dataframe or numpy array of input data

    Outputs:
        y: predicted value
    '''
    # Make prediction
    y = model.predict(X)
    return y

# Main function (runs when the file is run through a terminal)
if __name__ == "__main__":
    # Load the model from a pickle file
    model = load_model('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/models/gradient_boosting_model.pkl')

    # Package new data
    X = create_new_data(
        28.0,	
        70.0,	
        62.0,	
        2022.0,	
        24.0,	
        190.0,		
        3168.0,	
        1686.0,	
        1
    )

    # Run the model to predict
    y = run_prediction(model, X)

    # Print the results
    print(f"Predicted value: {y[0]}")