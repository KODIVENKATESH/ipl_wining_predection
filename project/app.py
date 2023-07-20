import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



st.set_page_config(page_title='IPL prediction', layout='wide')
text = "Team Selection And Winning Prediction In Cricket Using Machine Learning Techniques"

st.write(f"<h1 style='text-align: center'>{text}</h1>", unsafe_allow_html=True)
st.write("")
st.header("Team Selection")

background_style = """
<style>
body {
    background-image: url('IPLBots.jpeg');
    background-size: contain;
}
</style>
"""

# Use the custom CSS style to set the background image
st.markdown(background_style, unsafe_allow_html=True)

# best batsman
@st.cache_data
def best_batsmen():
    data=pd.read_csv("all_season_batting_card.csv")
    data.drop(data[data['season'] <= 2017].index, inplace = True)
    data=data.drop(['match_id','match_name','home_team','away_team','venue','city','country','current_innings','innings_id','name','minutes','captain','runningScore','runningOver','shortText','commentary','link','strikeRate'],axis=1)
    names=data['fullName'].unique()
    runs=[]
    ballsfaced=[]
    fours=[]
    sixes=[]
    strikerate=[]

    for i in range(len(names)):
        temp1=0
        temp2=0
        temp3=0
        temp4=0
        temp5=0
        for ri,rd in data.iterrows():
            if(rd['fullName']==names[i]):
                temp1+=rd['runs']
                temp2+=rd['ballsFaced']
                temp3+=rd['fours']
                temp4+=rd['sixes']
        runs.append(temp1)
        ballsfaced.append(temp2)
        fours.append(temp3)
        sixes.append(temp4)
        if(temp2==0):
            strikerate.append(0)
        else:
            strikerate.append(temp1*100/temp2)

    d={"Name":names,"runs":runs,"balls":ballsfaced, "fours":fours,"sixes":sixes,"sr":strikerate}
    alldata=pd.DataFrame(d)
    d2=alldata.sort_values(["runs","sr"], ascending=[False,True])
    return d2

# best bowling   
@st.cache_data
def best_bowlers():
    df = pd.read_csv("IPL_Ball_by_Ball_2008_2022.csv")
    # df[df['ballnumber']==10]
    d1 = df.groupby(['bowler','ID','overs']).sum()
    d1.reset_index()
    d2 = d1.drop(['innings','ballnumber','batsman_run','extras_run','non_boundary'], axis=1).reset_index()
    d2 = d2.drop(['ID','overs'],axis=1)
    d2['total_run'] = d2['total_run']/6
    d2.rename(columns={'total_run':'Economy'}, inplace=True)
    d2 = d2[d2['isWicketDelivery'] != 0]
    d2.reset_index(inplace=True,drop=True)
    d2['Economy']=(d2['Economy'].apply(lambda x:x*-1))*(-1)
    d2=d2.sort_values(["isWicketDelivery","Economy"], ascending=[False,True])
    return d2

# ipl match winning prediction
@st.cache_data
def matchWinningPrediction():
    balls = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
    matches = pd.read_csv('IPL_Matches_2008_2022.csv')
    # matches['City'].value_counts().iplot()
    total_score = balls.groupby(['ID', 'innings']).sum()['total_run'].reset_index()
    total_score = total_score[total_score['innings']==1]
    total_score['target'] = total_score['total_run'] + 1
    match_df = matches.merge(total_score[['ID','target']], on='ID')
    teams = [
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad', 
    'Delhi Capitals', 
    'Chennai Super Kings',
    'Gujarat Titans', 
    'Lucknow Super Giants', 
    'Kolkata Knight Riders',
    'Punjab Kings', 
    'Mumbai Indians'
    ]
    match_df['Team1'] = match_df['Team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['Team2'] = match_df['Team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')

    match_df['Team1'] = match_df['Team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
    match_df['Team2'] = match_df['Team2'].str.replace('Kings XI Punjab', 'Punjab Kings')
    match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')


    match_df['Team1'] = match_df['Team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['Team2'] = match_df['Team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df = match_df[match_df['Team1'].isin(teams)]
    match_df = match_df[match_df['Team2'].isin(teams)]
    match_df = match_df[match_df['WinningTeam'].isin(teams)]
    match_df = match_df[match_df['method'].isna()]
    match_df = match_df[['ID','City','Team1','Team2','WinningTeam','target']].dropna()
    balls['BattingTeam'] = balls['BattingTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')
    balls['BattingTeam'] = balls['BattingTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    balls['BattingTeam'] = balls['BattingTeam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

    balls = balls[balls['BattingTeam'].isin(teams)]
    balls_df = match_df.merge(balls, on='ID')
    balls_df = balls_df[balls_df['innings']==2]
    balls_df['current_score'] = balls_df.groupby('ID')['total_run'].cumsum()
    balls_df['runs_left'] = np.where(balls_df['target']-balls_df['current_score']>=0, balls_df['target']-balls_df['current_score'], 0)
    balls_df['balls_left'] = np.where(120 - balls_df['overs']*6 - balls_df['ballnumber']>=0,120 - balls_df['overs']*6 - balls_df['ballnumber'], 0)
    balls_df['wickets_left'] = 10 - balls_df.groupby('ID')['isWicketDelivery'].cumsum()
    balls_df['current_run_rate'] = (balls_df['current_score']*6)/(120-balls_df['balls_left'])
    balls_df['required_run_rate'] = np.where(balls_df['balls_left']>0, balls_df['runs_left']*6/balls_df['balls_left'], 0)
    def result(row):
        return 1 if row['BattingTeam'] == row['WinningTeam'] else 0
    balls_df['result'] = balls_df.apply(result, axis=1)
    index1 = balls_df[balls_df['Team2']==balls_df['BattingTeam']]['Team1'].index
    index2 = balls_df[balls_df['Team1']==balls_df['BattingTeam']]['Team2'].index
    balls_df.loc[index1, 'BowlingTeam'] = balls_df.loc[index1, 'Team1']
    balls_df.loc[index2, 'BowlingTeam'] = balls_df.loc[index2, 'Team2']
    final_df = balls_df[['BattingTeam', 'BowlingTeam','City','runs_left','balls_left','wickets_left','current_run_rate','required_run_rate','target','result']]
    trf = ColumnTransformer([('trf', OneHotEncoder(sparse=False,drop='first'),['BattingTeam','BowlingTeam','City'])],remainder = 'passthrough')
    X = final_df.drop('result', axis=1)
    y = final_df['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    pipe = Pipeline(steps=[('step1',trf),('step2',RandomForestClassifier())])
    pipe.fit(X_train, y_train)
    return pipe



batsData=best_batsmen()
bowlingData = best_bowlers()
predictionIpl = matchWinningPrediction()


col1, col2 = st.columns(2)
slider_style = """
    .streamlit-slider {
        width: 100%;
    }
    .streamlit-slider .stSlider label {
        font-size: 20px;
        color: red;
    }
"""
with col1:
    # st.subheader("Top Batsmen")
    value = st.slider('Top Batsmen', 0, 100, step=5, value=10, format="%d")
    st.dataframe(batsData.head(value),1000,400)

with col2:
   
    value = st.slider('Top Bowlers', 0, 100, step=5, value=10)
    st.dataframe(bowlingData.head(value),1000,400)


st.header("Winner Prediction")
   
    
col3,epty,col4,epty1,col5 = st.columns([1,0.1,1,0.1,1])
teams=['Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad', 
    'Delhi Capitals', 
    'Chennai Super Kings',
    'Gujarat Titans', 
    'Lucknow Super Giants', 
    'Kolkata Knight Riders',
    'Punjab Kings', 
    'Mumbai Indians']
cities = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
       'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
       'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
       'Bangalore', 'Raipur', 'Ranchi', 'Cuttack', 'Dharamsala', 'Nagpur',
       'Johannesburg', 'Centurion', 'Durban', 'Bloemfontein',
       'Port Elizabeth', 'Kimberley', 'East London', 'Cape Town']

with col3:
    battingTeam = st.selectbox("Select Batting Team",teams,index=teams.index("Royal Challengers Bangalore"))
with epty:
    st.write("")
with col4:
    bowlingTeam = st.selectbox("Select Bowling Team",teams)
with epty1:
    st.write("")
with col5:
    city = st.selectbox("Select city",cities)

col6,epty2,col7,epty3,col8=st.columns([1,0.1,1,0.1,1])
with col6:
    required_runs = st.number_input("Enter required runs",step=1)
# st.write(required_runs)
with epty2:
    st.write("")
with col7:
    crr = st.number_input("Current run rate")
with epty3:
    st.write("")
with col8:
    rrr = st.number_input("Required run rate")

col9,empty,col10=st.columns([1,0.1,1])
with col9:
    ballsLeft = st.slider('Balls Left', 1, 120, step=1, value=120)
with empty:
    st.write("")
with col10:
    wicketsLeft = st.slider('Wickets left', 1, 10, value=10)


col11,empty1,col12=st.columns([1,0.1,1])
with col11:
    target = st.number_input("Target",step=1)
with empty1:
    st.write("")
with col12:
    st.write("")
    st.write("")

    if st.button('Click me!'):
        inputToModel ={"BattingTeam":[battingTeam],"BowlingTeam":[bowlingTeam],"City":[city],"runs_left":[required_runs],
                       "balls_left":[ballsLeft],"wickets_left":[wicketsLeft],"current_run_rate":[crr],
                       "required_run_rate":[rrr],"target":[target]}
        fdf = pd.DataFrame(inputToModel)
        # st.write(fdf)
        arr=predictionIpl.predict(fdf)
       
        if arr[0]==0:
            st.write(fdf["BowlingTeam"].to_string(index=False))
        else:
            st.write(fdf["BattingTeam"].to_string(index=False))
        # st.write(predictionIpl.predict_proba(fdf)*100)
      