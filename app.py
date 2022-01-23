from flask import Flask,render_template,jsonify,request
import pandas as pd
import numpy as np
import joblib
import xgboost
from xgboost import XGBRegressor




app=Flask(__name__)
scroe_predictor=pd.read_csv("artificats/file/cleaned.csv")
model1=joblib.load('model','w+')


@app.route("/",methods=['GET','POST'])
def index():
    batting_team=sorted(scroe_predictor["battingteams"].unique())
    batting_team.insert(0,"india")
    bowling_team=sorted(scroe_predictor["bowlingteam"].unique(),reverse=True)
    city=sorted(scroe_predictor["city"].unique())
    html=render_template("index.html",batting_teams=batting_team,bowling_teams=bowling_team,cities=city)
    return(html)
@app.route("/predict",methods=["POST"])
def predict():
    batting_team=request.form.get("batting")
    bowling_team=request.form.get("bowling")
    currentscore=int(request.form.get("score"))
    city=request.form.get("city")
    wicketdown=int(request.form.get("wicket"))
    over=int(request.form.get("over"))
    runs5=int(request.form.get("run"))
    balls_count = (over * 6)
    wickets_left = 10 - wicketdown
    crr = (currentscore / over)
    # battingteams, bowlingteam,current_score,wickets_left,crr,city,last_five

    input_df = pd.DataFrame(
        {'battingteam': [batting_team], 'bowlingteam': [bowling_team],'current_score': [currentscore],
         'ball_count':[balls_count],'wickets_left': [wickets_left], 'crr': [crr],"city":[city] ,'last_five': [runs5]})
    result = model1.predict(input_df)
    print(result)
    #prediction=model.predict(pd.DataFrame(columns=['battingteams','bowlingteam','current_score','wickets_left', 'crr','city','last_five'],
                              #data=np.array([batting_team,bowling_team, currentscore, wicketdown, crr,city, runs5]).reshape(1, 7)))
    return  str(result)
    
    
if __name__=="__main__":
    app.run(debug=True)