import pandas as pd
from datetime import datetime

# Load your full schedule (adjust the path and CSV reading as needed)
schedule_df = pd.read_csv('/Users/blairjdaniel/lighthouse/lighthouse/NHL/NHL_points_projection/files/schedule/schedule.csv')

# Convert the 'Date' column to datetime (if not already)
schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])

# Filter the schedule to just today's games (normalize both timestamps)
today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
today_schedule = schedule_df[schedule_df['Date'].dt.normalize() == today]