import joblib
import pandas as pd
model1=joblib.load('finalmodel','w+')
input_df = pd.DataFrame(
        {'battingteam': ["Australia"], 'bowlingteam': ["srilanka"],'current_score': [97],
         'ball_count':[45],'wickets_left': [7], 'crr': [8.67],"city":["Melbourne"] ,'last_five': [43]})
result = model1.predict(input_df)
#result = model1.predict(input_df)    
print(result)                      
    