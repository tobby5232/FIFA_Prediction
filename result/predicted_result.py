# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 18:46:41 2022

@author: ch406
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xG_prediction_linearRegression import xG_predicted_method as xg
from Prediction_randonForest import result_prediction as RP

#%%
#Dataframe information

df_group_stage = pd.read_csv('data_group_stage.csv')

# print(df_group_stage.info())

#%%
#Import xG_prediction
xG_predicted=xg(df_group_stage[['Elo_rating_dif']])
# print(xG_predicted)

df_group_stage.insert(len(df_group_stage.columns),'xG_predicted',xG_predicted)
# print(df_group_stage.info())

#%%
#Aveage xG per team
xG_home=df_group_stage.groupby("home_team").mean()[['xG_predicted']]
xG_away=df_group_stage.groupby("away_team").mean()[['xG_predicted']].apply(lambda x: x*(-1), axis = 1)
xG_total=xG_home.add(xG_away, fill_value=0)
# print(xG_total)

#%%
#Predict result

X=df_group_stage.drop(labels=['home_team','away_team',
                              'ovr_home','ovr_away',
                              'att_home','att_away',
                              'mid_home','mid_away',
                              'defe_home','defe_away',
                              'Previous_Rank_home','Previous_Rank_away',
                              'Total_Points_home','Total_Points_away',
                              'Elo_ranking_home','Elo_ranking_away',
                              'Elo_rating_home','Elo_rating_away'],axis=1)
result =RP(X)

df_group_stage.insert(len(df_group_stage.columns),'result_predicted',result)
print(df_group_stage)

#%%
#Import group table

df_group_table = pd.read_csv('group_table.csv')
df_group_table.set_index(keys = ["group","Nation"],inplace=True)
# print(df_group_table)

#%%
#Calculate group points

for i in range(len(df_group_stage)):
    if df_group_stage['result_predicted'][i]=='win':
        for j in range(len(df_group_table)):
            if df_group_stage['home_team'][i]==df_group_table.index[j][1]:
                df_group_table['game'][j]=df_group_table['game'][j]+1
                df_group_table['W'][j]=df_group_table['W'][j]+1
                df_group_table['points'][j]=df_group_table['points'][j]+3
            if df_group_stage['away_team'][i]==df_group_table.index[j][1]:
                df_group_table['game'][j]=df_group_table['game'][j]+1
                df_group_table['L'][j]=df_group_table['L'][j]+1
    elif df_group_stage['result_predicted'][i]=='lose':
        for j in range(len(df_group_table)):
            if df_group_stage['away_team'][i]==df_group_table.index[j][1]:
                df_group_table['game'][j]=df_group_table['game'][j]+1
                df_group_table['W'][j]=df_group_table['W'][j]+1
                df_group_table['points'][j]=df_group_table['points'][j]+3
            if df_group_stage['home_team'][i]==df_group_table.index[j][1]:
                df_group_table['game'][j]=df_group_table['game'][j]+1
                df_group_table['L'][j]=df_group_table['L'][j]+1
    else:
        for j in range(len(df_group_table)):
            if df_group_stage['away_team'][i]==df_group_table.index[j][1]:
                df_group_table['game'][j]=df_group_table['game'][j]+1
                df_group_table['D'][j]=df_group_table['D'][j]+1
                df_group_table['points'][j]=df_group_table['points'][j]+1
            if df_group_stage['home_team'][i]==df_group_table.index[j][1]:
                df_group_table['game'][j]=df_group_table['game'][j]+1
                df_group_table['D'][j]=df_group_table['D'][j]+1
                df_group_table['points'][j]=df_group_table['points'][j]+1
                
df_group_table=df_group_table.sort_values(by=['group','points'], ascending=(True,False))
# print(df_group_table)

#%%
#Sort ranking by xG_avg if same group points

xG_avg=[]

for i in range(len(df_group_table)):
    for j in range(len(xG_total)):
            if df_group_table.index[i][1]==xG_total.index[j]:
                xG_avg.append(xG_total['xG_predicted'][j])
                break
df_group_table.insert(len(df_group_table.columns),'xG_avg',xG_avg)
df_group_table=df_group_table.sort_values(by=['group','points','xG_avg'], ascending=(True,False,False))
print(df_group_table)

#%%
#Fillter 16th round teams
group=['Group_A','Group_B','Group_C','Group_D',
        'Group_E','Group_F','Group_G','Group_H']

home_team=[]
away_team=[]

for i in range(len(group)):
    if i%2==1:
        home_team.append(df_group_table.loc[group[i]].iloc[:,:].index[0])
        away_team.append(df_group_table.loc[group[i-1]].iloc[:,:].index[1])
    else:
        home_team.append(df_group_table.loc[group[i]].iloc[:,:].index[0])
        away_team.append(df_group_table.loc[group[i+1]].iloc[:,:].index[1])

df_16th=pd.DataFrame({'home_team':home_team,'away_team':away_team})
# print(df_16th)

#%%
#Import ovr/att/mid/defe/privious_ranking/points/elo_ranking/elo_rating in 16th match

ovr_home,ovr_away=[[],[]]
att_home,att_away=[[],[]]
mid_home,mid_away=[[],[]]
defe_home,defe_away=[[],[]]
Previous_Rank_home,Previous_Rank_away=[[],[]]
Total_Points_home,Total_Points_away=[[],[]]
Elo_ranking_home,Elo_ranking_away=[[],[]]
Elo_rating_home,Elo_rating_away=[[],[]]

for i in range(len(df_16th)):
    for j in range(len(df_group_stage)):
            if df_16th['home_team'][i]==df_group_stage['home_team'][j]:
                ovr_home.append(df_group_stage['ovr_home'][j])
                att_home.append(df_group_stage['att_home'][j])
                mid_home.append(df_group_stage['mid_home'][j])
                defe_home.append(df_group_stage['defe_home'][j])
                Previous_Rank_home.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_home.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_home.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_home.append(df_group_stage['Elo_rating_home'][j])
                break
    for j in range(len(df_group_stage)):
            if df_16th['away_team'][i]==df_group_stage['home_team'][j]:
                ovr_away.append(df_group_stage['ovr_home'][j])
                att_away.append(df_group_stage['att_home'][j])
                mid_away.append(df_group_stage['mid_home'][j])
                defe_away.append(df_group_stage['defe_home'][j])
                Previous_Rank_away.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_away.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_away.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_away.append(df_group_stage['Elo_rating_home'][j])
                break

insert_titles=['ovr_home','ovr_away','att_home','att_away','mid_home','mid_away','defe_home','defe_away',
                'Previous_Rank_home','Previous_Rank_away','Total_Points_home','Total_Points_away',
                'Elo_ranking_home','Elo_ranking_away','Elo_rating_home','Elo_rating_away']
insert_items=[ovr_home,ovr_away,att_home,att_away,mid_home,mid_away,defe_home,defe_away,
                Previous_Rank_home,Previous_Rank_away,Total_Points_home,Total_Points_away,
                Elo_ranking_home,Elo_ranking_away,Elo_rating_home,Elo_rating_away]
for i in range(len(insert_items)):
    df_16th.insert(len(df_16th.columns),insert_titles[i],insert_items[i])
# print(df_16th.info())

#%%
#Inset 16th difference

dif_title=['ovr_dif','att_dif','mid_dif','defe_dif',
            'rank_dif','points_dif','Elo_ranking_dif','Elo_rating_dif']

for i in range(0,len(insert_titles),2):
    df_16th[dif_title[i//2]] = df_16th.apply(lambda x: x[insert_titles[i]] - x[insert_titles[i+1]], axis = 1)
# print(df_16th.info())

#%%
#Import 16_xG_prediction
xG_predicted16=xg(df_16th[['Elo_rating_dif']])
# print(xG_predicted16)

df_16th.insert(len(df_16th.columns),'xG_predicted',xG_predicted16)
# print(df_16th[['home_team','away_team','Elo_rating_dif','xG_predicted']])

#%%
#Predict 16th result

X16=df_16th.drop(labels=['home_team','away_team',
                              'ovr_home','ovr_away',
                              'att_home','att_away',
                              'mid_home','mid_away',
                              'defe_home','defe_away',
                              'Previous_Rank_home','Previous_Rank_away',
                              'Total_Points_home','Total_Points_away',
                              'Elo_ranking_home','Elo_ranking_away',
                              'Elo_rating_home','Elo_rating_away'],axis=1)
result_16 =RP(X16)

df_16th.insert(len(df_16th.columns),'result_predicted',result_16)
print(df_16th)

#%%
#Fillter 8th round teams
home_team=['t1','t2','t3','t4']
away_team=['t5','t6','t7','t8']

for i in range(len(df_16th)):
    if i%2==0:
        if i==0 or i==4:
            if df_16th['result_predicted'][i]=='win':
                home_team[i//2]=df_16th['home_team'][i]
            elif df_16th['result_predicted'][i]=='lose':
                home_team[i//2]=df_16th['away_team'][i]
            else:
                if df_16th['xG_predicted'][i] >0:
                    home_team[i//2]=df_16th['home_team'][i]
                else:
                    home_team[i//2]=df_16th['away_team'][i]
        else:
            if df_16th['result_predicted'][i]=='win':
                away_team[i//2-1]=df_16th['home_team'][i]
            elif df_16th['result_predicted'][i]=='lose':
                away_team[i//2-1]=df_16th['away_team'][i]
            else:
                if df_16th['xG_predicted'][i] >0:
                    away_team[i//2-1]=df_16th['home_team'][i]
                else:
                    away_team[i//2-1]=df_16th['away_team'][i]
    else:
        if i==1 or i==5:
            if df_16th['result_predicted'][i]=='win':
                home_team[int(0.5*i+0.5)]=df_16th['home_team'][i]
            elif df_16th['result_predicted'][i]=='lose':
                home_team[int(0.5*i+0.5)]=df_16th['away_team'][i]
            else:
                if df_16th['xG_predicted'][i] >0:
                    home_team[int(0.5*i+0.5)]=df_16th['home_team'][i]
                else:
                    home_team[int(0.5*i+0.5)]=df_16th['away_team'][i]
        else:
            if df_16th['result_predicted'][i]=='win':
                away_team[int(0.5*i-0.5)]=df_16th['home_team'][i]
            elif df_16th['result_predicted'][i]=='lose':
                away_team[int(0.5*i-0.5)]=df_16th['away_team'][i]
            else:
                if df_16th['xG_predicted'][i] >0:
                    away_team[int(0.5*i-0.5)]=df_16th['home_team'][i]
                else:
                    away_team[int(0.5*i-0.5)]=df_16th['away_team'][i]
# print(home_team)
# print(away_team)
df_8th=pd.DataFrame({'home_team':home_team,'away_team':away_team})
# print(df_8th)
#%%
#Import ovr/att/mid/defe/privious_ranking/points/elo_ranking/elo_rating in 8th match

ovr_home,ovr_away=[[],[]]
att_home,att_away=[[],[]]
mid_home,mid_away=[[],[]]
defe_home,defe_away=[[],[]]
Previous_Rank_home,Previous_Rank_away=[[],[]]
Total_Points_home,Total_Points_away=[[],[]]
Elo_ranking_home,Elo_ranking_away=[[],[]]
Elo_rating_home,Elo_rating_away=[[],[]]

for i in range(len(df_8th)):
    for j in range(len(df_group_stage)):
            if df_8th['home_team'][i]==df_group_stage['home_team'][j]:
                ovr_home.append(df_group_stage['ovr_home'][j])
                att_home.append(df_group_stage['att_home'][j])
                mid_home.append(df_group_stage['mid_home'][j])
                defe_home.append(df_group_stage['defe_home'][j])
                Previous_Rank_home.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_home.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_home.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_home.append(df_group_stage['Elo_rating_home'][j])
                break
    for j in range(len(df_group_stage)):
            if df_8th['away_team'][i]==df_group_stage['home_team'][j]:
                ovr_away.append(df_group_stage['ovr_home'][j])
                att_away.append(df_group_stage['att_home'][j])
                mid_away.append(df_group_stage['mid_home'][j])
                defe_away.append(df_group_stage['defe_home'][j])
                Previous_Rank_away.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_away.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_away.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_away.append(df_group_stage['Elo_rating_home'][j])
                break

insert_titles=['ovr_home','ovr_away','att_home','att_away','mid_home','mid_away','defe_home','defe_away',
                'Previous_Rank_home','Previous_Rank_away','Total_Points_home','Total_Points_away',
                'Elo_ranking_home','Elo_ranking_away','Elo_rating_home','Elo_rating_away']
insert_items=[ovr_home,ovr_away,att_home,att_away,mid_home,mid_away,defe_home,defe_away,
                Previous_Rank_home,Previous_Rank_away,Total_Points_home,Total_Points_away,
                Elo_ranking_home,Elo_ranking_away,Elo_rating_home,Elo_rating_away]
for i in range(len(insert_items)):
    df_8th.insert(len(df_8th.columns),insert_titles[i],insert_items[i])
# print(df_8th.info())

#%%
#Inset 8th difference
dif_title=['ovr_dif','att_dif','mid_dif','defe_dif',
            'rank_dif','points_dif','Elo_ranking_dif','Elo_rating_dif']

for i in range(0,len(insert_titles),2):
    df_8th[dif_title[i//2]] = df_8th.apply(lambda x: x[insert_titles[i]] - x[insert_titles[i+1]], axis = 1)
# print(df_8th.info())

#%%
#Import 8_xG_prediction
xG_predicted8=xg(df_8th[['Elo_rating_dif']])
# print(xG_predicted8)

df_8th.insert(len(df_8th.columns),'xG_predicted',xG_predicted8)
# print(df_8th[['home_team','away_team','Elo_rating_dif','xG_predicted']])

#%%
#Predict 8th result

X8=df_8th.drop(labels=['home_team','away_team',
                              'ovr_home','ovr_away',
                              'att_home','att_away',
                              'mid_home','mid_away',
                              'defe_home','defe_away',
                              'Previous_Rank_home','Previous_Rank_away',
                              'Total_Points_home','Total_Points_away',
                              'Elo_ranking_home','Elo_ranking_away',
                              'Elo_rating_home','Elo_rating_away'],axis=1)
result_8 =RP(X8)

df_8th.insert(len(df_8th.columns),'result_predicted',result_8)
print(df_8th)

#%%
#Fillter 4th round teams
home_team=['t1','t2']
away_team=['t3','t4']

for i in range(len(df_8th)):
    if i%2==0:
        if i==0:
            if df_8th['result_predicted'][i]=='win':
                home_team[i//2]=df_8th['home_team'][i]
            elif df_8th['result_predicted'][i]=='lose':
                home_team[i//2]=df_8th['away_team'][i]
            else:
                if df_8th['xG_predicted'][i] >0:
                    home_team[i//2]=df_8th['home_team'][i]
                else:
                    home_team[i//2]=df_8th['away_team'][i]
        else:
            if df_8th['result_predicted'][i]=='win':
                away_team[i//2-1]=df_8th['home_team'][i]
            elif df_8th['result_predicted'][i]=='lose':
                away_team[i//2-1]=df_8th['away_team'][i]
            else:
                if df_8th['xG_predicted'][i] >0:
                    away_team[i//2-1]=df_8th['home_team'][i]
                else:
                    away_team[i//2-1]=df_8th['away_team'][i]
    else:
        if i==1:
            if df_8th['result_predicted'][i]=='win':
                home_team[int(0.5*i+0.5)]=df_8th['home_team'][i]
            elif df_8th['result_predicted'][i]=='lose':
                home_team[int(0.5*i+0.5)]=df_8th['away_team'][i]
            else:
                if df_8th['xG_predicted'][i] >0:
                    home_team[int(0.5*i+0.5)]=df_8th['home_team'][i]
                else:
                    home_team[int(0.5*i+0.5)]=df_8th['away_team'][i]
        else:
            if df_8th['result_predicted'][i]=='win':
                away_team[int(0.5*i-0.5)]=df_8th['home_team'][i]
            elif df_8th['result_predicted'][i]=='lose':
                away_team[int(0.5*i-0.5)]=df_8th['away_team'][i]
            else:
                if df_8th['xG_predicted'][i] >0:
                    away_team[int(0.5*i-0.5)]=df_8th['home_team'][i]
                else:
                    away_team[int(0.5*i-0.5)]=df_8th['away_team'][i]
# print(home_team)
# print(away_team)
df_4th=pd.DataFrame({'home_team':home_team,'away_team':away_team})
# print(df_4th)

#%%
#Import ovr/att/mid/defe/privious_ranking/points/elo_ranking/elo_rating in 4th match

ovr_home,ovr_away=[[],[]]
att_home,att_away=[[],[]]
mid_home,mid_away=[[],[]]
defe_home,defe_away=[[],[]]
Previous_Rank_home,Previous_Rank_away=[[],[]]
Total_Points_home,Total_Points_away=[[],[]]
Elo_ranking_home,Elo_ranking_away=[[],[]]
Elo_rating_home,Elo_rating_away=[[],[]]

for i in range(len(df_4th)):
    for j in range(len(df_group_stage)):
            if df_4th['home_team'][i]==df_group_stage['home_team'][j]:
                ovr_home.append(df_group_stage['ovr_home'][j])
                att_home.append(df_group_stage['att_home'][j])
                mid_home.append(df_group_stage['mid_home'][j])
                defe_home.append(df_group_stage['defe_home'][j])
                Previous_Rank_home.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_home.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_home.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_home.append(df_group_stage['Elo_rating_home'][j])
                break
    for j in range(len(df_group_stage)):
            if df_4th['away_team'][i]==df_group_stage['home_team'][j]:
                ovr_away.append(df_group_stage['ovr_home'][j])
                att_away.append(df_group_stage['att_home'][j])
                mid_away.append(df_group_stage['mid_home'][j])
                defe_away.append(df_group_stage['defe_home'][j])
                Previous_Rank_away.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_away.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_away.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_away.append(df_group_stage['Elo_rating_home'][j])
                break

insert_titles=['ovr_home','ovr_away','att_home','att_away','mid_home','mid_away','defe_home','defe_away',
                'Previous_Rank_home','Previous_Rank_away','Total_Points_home','Total_Points_away',
                'Elo_ranking_home','Elo_ranking_away','Elo_rating_home','Elo_rating_away']
insert_items=[ovr_home,ovr_away,att_home,att_away,mid_home,mid_away,defe_home,defe_away,
                Previous_Rank_home,Previous_Rank_away,Total_Points_home,Total_Points_away,
                Elo_ranking_home,Elo_ranking_away,Elo_rating_home,Elo_rating_away]
for i in range(len(insert_items)):
    df_4th.insert(len(df_4th.columns),insert_titles[i],insert_items[i])
# print(df_4th.info())

#%%
#Inset 4th difference
dif_title=['ovr_dif','att_dif','mid_dif','defe_dif',
            'rank_dif','points_dif','Elo_ranking_dif','Elo_rating_dif']

for i in range(0,len(insert_titles),2):
    df_4th[dif_title[i//2]] = df_4th.apply(lambda x: x[insert_titles[i]] - x[insert_titles[i+1]], axis = 1)
# print(df_4th.info())

#%%
#Import 4_xG_prediction
xG_predicted4=xg(df_4th[['Elo_rating_dif']])
# print(xG_predicted4)

df_4th.insert(len(df_4th.columns),'xG_predicted',xG_predicted4)
# print(df_4th[['home_team','away_team','Elo_rating_dif','xG_predicted']])

#%%
#Predict 8th result

X4=df_4th.drop(labels=['home_team','away_team',
                              'ovr_home','ovr_away',
                              'att_home','att_away',
                              'mid_home','mid_away',
                              'defe_home','defe_away',
                              'Previous_Rank_home','Previous_Rank_away',
                              'Total_Points_home','Total_Points_away',
                              'Elo_ranking_home','Elo_ranking_away',
                              'Elo_rating_home','Elo_rating_away'],axis=1)
result_4 =RP(X4)

df_4th.insert(len(df_4th.columns),'result_predicted',result_4)
print(df_4th)

#%%
#Fillter champion/third round teams
home_team_champ=['t1']
away_team_champ=['t2']

home_team_third=['t3']
away_team_third=['t4']

for i in range(len(df_4th)):
    if i%2==0:
        if df_4th['result_predicted'][i]=='win':
            home_team_champ[0]=df_4th['home_team'][i]
            home_team_third[0]=df_4th['away_team'][i]
        elif df_4th['result_predicted'][i]=='lose':
            home_team_champ[0]=df_4th['away_team'][i]
            home_team_third[0]=df_4th['home_team'][i]
        else:
            if df_4th['xG_predicted'][i] >0:
                home_team_champ[0]=df_4th['home_team'][i]
                home_team_third[0]=df_4th['away_team'][i]
            else:
                home_team_champ[0]=df_4th['away_team'][i]
                home_team_third[0]=df_4th['home_team'][i]
    else:
        if df_4th['result_predicted'][i]=='win':
            away_team_champ[0]=df_4th['home_team'][i]
            away_team_third[0]=df_4th['away_team'][i]
        elif df_4th['result_predicted'][i]=='lose':
            away_team_champ[0]=df_4th['away_team'][i]
            away_team_third[0]=df_4th['home_team'][i]
        else:
            if df_4th['xG_predicted'][i] >0:
                away_team_champ[0]=df_4th['home_team'][i]
                away_team_third[0]=df_4th['away_team'][i]
            else:
                away_team_champ[0]=df_4th['away_team'][i]
                away_team_third[0]=df_4th['home_team'][i]

df_champion=pd.DataFrame({'home_team':home_team_champ,'away_team':away_team_champ})
df_third=pd.DataFrame({'home_team':home_team_third,'away_team':away_team_third})
# print(df_champion)
# print(df_third)

#%%
#Import ovr/att/mid/defe/privious_ranking/points/elo_ranking/elo_rating in third match

ovr_home,ovr_away=[[],[]]
att_home,att_away=[[],[]]
mid_home,mid_away=[[],[]]
defe_home,defe_away=[[],[]]
Previous_Rank_home,Previous_Rank_away=[[],[]]
Total_Points_home,Total_Points_away=[[],[]]
Elo_ranking_home,Elo_ranking_away=[[],[]]
Elo_rating_home,Elo_rating_away=[[],[]]

for i in range(len(df_third)):
    for j in range(len(df_group_stage)):
            if df_third['home_team'][i]==df_group_stage['home_team'][j]:
                ovr_home.append(df_group_stage['ovr_home'][j])
                att_home.append(df_group_stage['att_home'][j])
                mid_home.append(df_group_stage['mid_home'][j])
                defe_home.append(df_group_stage['defe_home'][j])
                Previous_Rank_home.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_home.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_home.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_home.append(df_group_stage['Elo_rating_home'][j])
                break
    for j in range(len(df_group_stage)):
            if df_third['away_team'][i]==df_group_stage['home_team'][j]:
                ovr_away.append(df_group_stage['ovr_home'][j])
                att_away.append(df_group_stage['att_home'][j])
                mid_away.append(df_group_stage['mid_home'][j])
                defe_away.append(df_group_stage['defe_home'][j])
                Previous_Rank_away.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_away.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_away.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_away.append(df_group_stage['Elo_rating_home'][j])
                break

insert_titles=['ovr_home','ovr_away','att_home','att_away','mid_home','mid_away','defe_home','defe_away',
                'Previous_Rank_home','Previous_Rank_away','Total_Points_home','Total_Points_away',
                'Elo_ranking_home','Elo_ranking_away','Elo_rating_home','Elo_rating_away']
insert_items=[ovr_home,ovr_away,att_home,att_away,mid_home,mid_away,defe_home,defe_away,
                Previous_Rank_home,Previous_Rank_away,Total_Points_home,Total_Points_away,
                Elo_ranking_home,Elo_ranking_away,Elo_rating_home,Elo_rating_away]
for i in range(len(insert_items)):
    df_third.insert(len(df_third.columns),insert_titles[i],insert_items[i])
# print(df_third.info())

#%%
#Inset third difference
dif_title=['ovr_dif','att_dif','mid_dif','defe_dif',
            'rank_dif','points_dif','Elo_ranking_dif','Elo_rating_dif']

for i in range(0,len(insert_titles),2):
    df_third[dif_title[i//2]] = df_third.apply(lambda x: x[insert_titles[i]] - x[insert_titles[i+1]], axis = 1)
# print(df_third.info())

#%%
#Import third_xG_prediction
xG_predictedThird=xg(df_third[['Elo_rating_dif']])
# print(xG_predictedThird)

df_third.insert(len(df_third.columns),'xG_predicted',xG_predictedThird)
# print(df_third[['home_team','away_team','Elo_rating_dif','xG_predicted']])

#%%
#Predict third result

Xthird=df_third.drop(labels=['home_team','away_team',
                              'ovr_home','ovr_away',
                              'att_home','att_away',
                              'mid_home','mid_away',
                              'defe_home','defe_away',
                              'Previous_Rank_home','Previous_Rank_away',
                              'Total_Points_home','Total_Points_away',
                              'Elo_ranking_home','Elo_ranking_away',
                              'Elo_rating_home','Elo_rating_away'],axis=1)
result_third =RP(Xthird)

df_third.insert(len(df_third.columns),'result_predicted',result_third)
print(df_third)

#%%
#Import ovr/att/mid/defe/privious_ranking/points/elo_ranking/elo_rating in champion match

ovr_home,ovr_away=[[],[]]
att_home,att_away=[[],[]]
mid_home,mid_away=[[],[]]
defe_home,defe_away=[[],[]]
Previous_Rank_home,Previous_Rank_away=[[],[]]
Total_Points_home,Total_Points_away=[[],[]]
Elo_ranking_home,Elo_ranking_away=[[],[]]
Elo_rating_home,Elo_rating_away=[[],[]]

for i in range(len(df_champion)):
    for j in range(len(df_group_stage)):
            if df_champion['home_team'][i]==df_group_stage['home_team'][j]:
                ovr_home.append(df_group_stage['ovr_home'][j])
                att_home.append(df_group_stage['att_home'][j])
                mid_home.append(df_group_stage['mid_home'][j])
                defe_home.append(df_group_stage['defe_home'][j])
                Previous_Rank_home.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_home.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_home.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_home.append(df_group_stage['Elo_rating_home'][j])
                break
    for j in range(len(df_group_stage)):
            if df_champion['away_team'][i]==df_group_stage['home_team'][j]:
                ovr_away.append(df_group_stage['ovr_home'][j])
                att_away.append(df_group_stage['att_home'][j])
                mid_away.append(df_group_stage['mid_home'][j])
                defe_away.append(df_group_stage['defe_home'][j])
                Previous_Rank_away.append(df_group_stage['Previous_Rank_home'][j])
                Total_Points_away.append(df_group_stage['Total_Points_home'][j])
                Elo_ranking_away.append(df_group_stage['Elo_ranking_home'][j])
                Elo_rating_away.append(df_group_stage['Elo_rating_home'][j])
                break

insert_titles=['ovr_home','ovr_away','att_home','att_away','mid_home','mid_away','defe_home','defe_away',
                'Previous_Rank_home','Previous_Rank_away','Total_Points_home','Total_Points_away',
                'Elo_ranking_home','Elo_ranking_away','Elo_rating_home','Elo_rating_away']
insert_items=[ovr_home,ovr_away,att_home,att_away,mid_home,mid_away,defe_home,defe_away,
                Previous_Rank_home,Previous_Rank_away,Total_Points_home,Total_Points_away,
                Elo_ranking_home,Elo_ranking_away,Elo_rating_home,Elo_rating_away]
for i in range(len(insert_items)):
    df_champion.insert(len(df_champion.columns),insert_titles[i],insert_items[i])
# print(df_champion.info())

#%%
#Inset champion difference
dif_title=['ovr_dif','att_dif','mid_dif','defe_dif',
            'rank_dif','points_dif','Elo_ranking_dif','Elo_rating_dif']

for i in range(0,len(insert_titles),2):
    df_champion[dif_title[i//2]] = df_champion.apply(lambda x: x[insert_titles[i]] - x[insert_titles[i+1]], axis = 1)
# print(df_champion.info())

#%%
#Import third_xG_prediction
xG_predictedChamp=xg(df_champion[['Elo_rating_dif']])
# print(xG_predictedChamp)

df_champion.insert(len(df_champion.columns),'xG_predicted',xG_predictedChamp)
# print(df_champion[['home_team','away_team','Elo_rating_dif','xG_predicted']])

#%%
#Predict champion result

Xchamp=df_champion.drop(labels=['home_team','away_team',
                              'ovr_home','ovr_away',
                              'att_home','att_away',
                              'mid_home','mid_away',
                              'defe_home','defe_away',
                              'Previous_Rank_home','Previous_Rank_away',
                              'Total_Points_home','Total_Points_away',
                              'Elo_ranking_home','Elo_ranking_away',
                              'Elo_rating_home','Elo_rating_away'],axis=1)
result_champ =RP(Xchamp)

df_champion.insert(len(df_champion.columns),'result_predicted',result_champ)
print(df_champion)