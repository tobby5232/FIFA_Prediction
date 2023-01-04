# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:06:56 2022

@author: ch406
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np


#%%
# Import xG data
df_xg = pd.read_csv('xG_rawData_ANSI.csv',encoding='ANSI')
df_xg=df_xg[['Date','Home','Away',
             'Score_Home','Score_Away',
             'xG_Home','xG_Away']]

for i in range(len(df_xg)):
    df_xg['Date'][i]=df_xg['Date'][i].split('/')[0]
df_xg['Date']=df_xg['Date'].astype('int32')

for i in range(len(df_xg)):
    df_xg["Home"][i]=df_xg["Home"][i].replace("N. Macedonia", "North Macedonia")
    df_xg["Home"][i]=df_xg["Home"][i].replace("Korea Republic", "South Korea")
    df_xg["Home"][i]=df_xg["Home"][i].replace("Turkiye", "Turkey")
    df_xg["Away"][i]=df_xg["Away"][i].replace("N. Macedonia", "North Macedonia")
    df_xg["Away"][i]=df_xg["Away"][i].replace("Korea Republic", "South Korea")
    df_xg["Away"][i]=df_xg["Away"][i].replace("Turkiye", "Turkey")
    
# print(df_xg.info())

#%%
# Import Elo_rating data
df_Elo = pd.read_csv('Elo_cleanData.csv')
# print(df_Elo.info())

#%%
#Merge df_xg & df_Elo
df_xG_final=df_xg.copy()

Elo_ranking_home,Elo_rating_home,avg_rank_home,avg_rating_home,Elo_ranking_away,Elo_rating_away,avg_rank_away,avg_rating_away=[[],[],[],[],
                                                                                                                               [],[],[],[]]
list_Elo_items=[Elo_ranking_home,Elo_rating_home,avg_rank_home,avg_rating_home,
                Elo_ranking_away,Elo_rating_away,avg_rank_away,avg_rating_away]
Elo_items=['Elo_ranking_home','Elo_rating_home','avg_rank_home','avg_rating_home',
           'Elo_ranking_away','Elo_rating_away','avg_rank_away','avg_rating_away']
Elo_items_single=['Elo_ranking','Elo_rating','avg_rank','avg_rating']

# Checking missing country
# df_xG_final.insert(len(df_xG_final.columns),'if_match','N')

for i in range(len(df_xG_final)):
    for j in range(len(df_Elo)):
        if df_xG_final['Home'][i]==df_Elo['Country_simp'][j] and df_xG_final['Date'][i]==df_Elo['year'][j]:
            # df_xG_final['if_match'][i]='Y'
            for k in range(len(Elo_items_single)):
                list_Elo_items[k].append(df_Elo[Elo_items_single[k]][j])
        if df_xG_final['Away'][i]==df_Elo['Country_simp'][j] and df_xG_final['Date'][i]==df_Elo['year'][j]:
            # df_xG_final['if_match'][i]='Y'
            for k in range(len(Elo_items_single)):
                list_Elo_items[k+len(Elo_items_single)].append(df_Elo[Elo_items_single[k]][j])
print(Elo_ranking_away)
            
for i in range(len(list_Elo_items)):
    df_xG_final.insert(len(df_xG_final.columns),Elo_items[i],list_Elo_items[i])
# print(df_xG_final.info())

#%%
#Add column for goal_dif,xG_dif,Elo_ranking_dif,Elo_rating_dif,avg_rank_dif,avg_rating_dif

goal_dif,xG_dif,Elo_ranking_dif,Elo_rating_dif,avg_rank_dif,avg_rating_dif=[[],[],[],
                                                                             [],[],[]]
dif_list=[goal_dif,xG_dif,Elo_ranking_dif,
          Elo_rating_dif,avg_rank_dif,avg_rating_dif]
dif_list_title=['goal_dif','xG_dif','Elo_ranking_dif',
                'Elo_rating_dif','avg_rank_dif','avg_rating_dif']

for i in range(len(df_xg)):
    goal_dif.append(df_xG_final['Score_Home'][i]-df_xG_final['Score_Away'][i])
    xG_dif.append(df_xG_final['xG_Home'][i]-df_xG_final['xG_Away'][i])
    Elo_ranking_dif.append(df_xG_final['Elo_ranking_home'][i]-df_xG_final['Elo_ranking_away'][i])
    Elo_rating_dif.append(df_xG_final['Elo_rating_home'][i]-df_xG_final['Elo_rating_away'][i])
    avg_rank_dif.append(df_xG_final['avg_rank_home'][i]-df_xG_final['avg_rank_away'][i])
    avg_rating_dif.append(df_xG_final['avg_rating_home'][i]-df_xG_final['avg_rating_away'][i])
    
for i in range(len(dif_list)):
    df_xG_final.insert(len(df_xG_final.columns),dif_list_title[i],dif_list[i])
# print(df_xG_final.info())


#%%
#Input FIFA_ranking & country_code
df_ranking = pd.read_csv('AllTime_ranking.csv')
df_country_code = pd.read_csv('Country_code.csv')

#Replace contary_code as full country name in df_ranking
df_ranking_merge = df_ranking.merge(df_country_code, how='inner', indicator=False)
df_ranking_merge=df_ranking_merge[['Country','Year','Previous_Rank','Total_Points','Country_Code']]
# print(df_ranking_merge.info())

#Check missing data
# df_ranking_out = df_ranking.merge(df_country_code, how='outer', 
#                                   indicator=True).loc[lambda x : x['_merge'] == 'left_only']
# print(df_ranking_out)

#%%
#Insert team ranking data to df_xG_final
Previous_Rank_home=[]
Total_Points_home=[]
Country_Code_home=[]

Previous_Rank_away=[]
Total_Points_away=[]
Country_Code_away=[]

insert_items=[Previous_Rank_home,Total_Points_home,Country_Code_home,
              Previous_Rank_away,Total_Points_away,Country_Code_away]
insert_items_title=['Previous_Rank_home','Total_Points_home','Country_Code_home',
                    'Previous_Rank_away','Total_Points_away','Country_Code_away']

for i in range(len(df_xG_final)):
    for j in range(len(df_ranking_merge)):
        if df_xG_final['Home'][i]==df_ranking_merge['Country'][j] and df_xG_final['Date'][i]==df_ranking_merge['Year'][j]:
            Previous_Rank_home.append(df_ranking_merge['Previous_Rank'][j])
            Total_Points_home.append(df_ranking_merge['Total_Points'][j])
            Country_Code_home.append(df_ranking_merge['Country_Code'][j])
        if df_xG_final['Away'][i]==df_ranking_merge['Country'][j] and df_xG_final['Date'][i]==df_ranking_merge['Year'][j]:
            Previous_Rank_away.append(df_ranking_merge['Previous_Rank'][j])
            Total_Points_away.append(df_ranking_merge['Total_Points'][j])
            Country_Code_away.append(df_ranking_merge['Country_Code'][j])
            
for k in range(len(insert_items)):
    df_xG_final.insert(len(df_xG_final.columns),insert_items_title[k],insert_items[k])
    if k==0 or k==1 or k==3 or k==4:
        df_xG_final[insert_items_title[k]]=df_xG_final[insert_items_title[k]].astype('float64')
print(df_xG_final.info())

#%%
#Input EA rating data

df_rating = pd.read_csv('AllTeamRating.csv')
df_rating=df_rating[['nation','overall','attack','mid','defence']]
for i in range(len(df_rating["nation"])):
    df_rating["nation"][i]=df_rating["nation"][i].replace("Côte d'Ivoire", "Ivory Coast")
    df_rating["nation"][i]=df_rating["nation"][i].replace("Korea Republic", "South Korea")
    
#Groupby different and average the rating
df_rating_group=df_rating.groupby(['nation']).agg(['mean'])
#Rating 縮減到小數後兩位
df_rating_group=df_rating_group.applymap(lambda x: '%.2f'%x)
# print(df_rating_group)

#%%
#Insert EA rating data to df_xG_final

ovr_home=[]
att_home=[]
mid_home=[]
defe_home=[]

ovr_away=[]
att_away=[]
mid_away=[]
defe_away=[]

Insert_column=[ovr_home,att_home,mid_home,defe_home,
                ovr_away,att_away,mid_away,defe_away]
Insert_title=['ovr_home','att_home','mid_home','defe_home',
                'ovr_away','att_away','mid_away','defe_away']

#Check nation different (Na value in list)
# df_OddAndResult.insert(11,'if_match','N')

for i in range(len(df_xG_final)):
    for j in range(len(df_rating_group)):
        if df_xG_final['Home'][i]==df_rating_group.index[j]:
            ovr_home.append(df_rating_group['overall']['mean'][j])
            att_home.append(df_rating_group['attack']['mean'][j])
            mid_home.append(df_rating_group['mid']['mean'][j])
            defe_home.append(df_rating_group['defence']['mean'][j])
            # df_xG_final['if_match'][i]='Y'
        if df_xG_final['Away'][i]==df_rating_group.index[j]:
            ovr_away.append(df_rating_group['overall']['mean'][j])
            att_away.append(df_rating_group['attack']['mean'][j])
            mid_away.append(df_rating_group['mid']['mean'][j])
            defe_away.append(df_rating_group['defence']['mean'][j])
            # df_xG_final['if_match'][i]='Y'
            
# print(df_xG_final[df_xG_final['if_match']=='N'])
            
for k in range(8):
    df_xG_final.insert(len(df_xG_final.columns),Insert_title[k],Insert_column[k])
    df_xG_final[Insert_title[k]]=df_xG_final[Insert_title[k]].astype('float64')
# print(df_xG_final.info()) 

#%%
#Input feature difference
insert_titles=['ovr_home','ovr_away','att_home','att_away','mid_home','mid_away','defe_home','defe_away',
                'Previous_Rank_home','Previous_Rank_away','Total_Points_home','Total_Points_away']

dif_title=['ovr_dif','att_dif','mid_dif','defe_dif',
            'rank_dif','points_dif']

for i in range(0,len(insert_titles),2):
    df_xG_final[dif_title[i//2]] = df_xG_final.apply(lambda x: x[insert_titles[i]] - x[insert_titles[i+1]], axis = 1)
print(df_xG_final.info())

#%%
df_xG_final.to_csv("xG_data.csv", index = None)

#%%
#Plot correlation matrix

df_xG_final_simp=df_xG_final[['xG_dif',
                              'rank_dif',
                              'points_dif',
                              'ovr_dif',
                              'att_dif',
                              'mid_dif',
                              'defe_dif',
                              'Elo_ranking_dif',
                              'Elo_rating_dif']]
plt.style.use('ggplot')
plt.figure(dpi=200)

correlation_matrix=df_xG_final_simp.corr()
print(correlation_matrix)

#Mask for triangle graph 
mask2 = np.zeros_like(correlation_matrix)
mask2[np.triu_indices_from(mask2)] = True

#Draw heatmap
heatmap=sns.heatmap(correlation_matrix, mask=mask2,annot=True,annot_kws={'size':6},fmt=".2f", vmin=-1, vmax=1,
                    xticklabels=True, yticklabels=True,cbar_kws={"shrink": .52})

heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize = 6) #Set y-axis label size
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90, fontsize = 6) #Set x-axis label size

heatmap.set_title('Correlation Heatmap with xG_diff', fontdict={'fontsize':8}, pad=8) #Set title size

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=7) #Set colorbar size

plt.show()