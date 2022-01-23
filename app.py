"""
    email: developer.akash104@gmail.com
    author: akash modi
"""



from flask import Flask,render_template,jsonify,request
from flask_cors import CORS,cross_origin
import pandas as pd
import joblib
import xgboost
from xgboost import XGBRegressor




app=Flask(__name__)
cors=CORS(app)
scroe_predictor=pd.read_csv("artificats/file/cleaned_.csv")
model1=joblib.load('finalmodel','w+')


@app.route("/",methods=['GET','POST'])
def index():
    batting_team=sorted(scroe_predictor["battingteams"].unique())
    batting_team.insert(0,"india")
    bowling_team=sorted(scroe_predictor["bowlingteam"].unique(),reverse=True)
    city=sorted(scroe_predictor["city"].unique())
    html=render_template("index.html",batting_teams=batting_team,bowling_teams=bowling_team,cities=city)
    return(html)
@app.route("/predict",methods=["POST"])
@cross_origin()
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
    try:

        input_df = pd.DataFrame(
            {'battingteam': [batting_team], 'bowlingteam': [bowling_team],'current_score': [currentscore],
            'ball_count':[balls_count],'wickets_left': [wickets_left], 'crr': [crr],"city":[city] ,'last_five': [runs5]})
        result = model1.predict(input_df)
        print("jhal",result)
        return  str(result)
    except Exception as e:
        print(e)
    #prediction=model.predict(pd.DataFrame(columns=['battingteams','bowlingteam','current_score','wickets_left', 'crr','city','last_five'],
                              #data=np.array([batting_team,bowling_team, currentscore, wicketdown, crr,city, runs5]).reshape(1, 7)))
    #input_df = pd.DataFrame(
        #{'battingteam': ["Australia"], 'bowlingteam': ["srilanka"],'current_score': [97],
        # 'ball_count':[45],'wickets_left': [7], 'crr': [8.67],"city":["Melbourne"] ,'last_five': [43]})
    #result = model1.predict(input_df)                          
    
    
    
if __name__=="__main__":
    app.run(debug=True)