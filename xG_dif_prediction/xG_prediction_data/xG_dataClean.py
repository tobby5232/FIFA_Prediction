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
    
print(df_xg.info())

#%%
# Import Elo_rating data
df_Elo = pd.read_csv('Elo_cleanData.csv')
print(df_Elo.info())

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
print(df_xG_final.info())

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
print(df_xG_final.info())

#%%
df_xG_final.to_csv("xG_data.csv", index = None)

# #%%
# #Plot correlation matrix

# df_xG_final_simp=df_xG_final[['goal_dif','xG_dif','Elo_ranking_dif',
#                               'Elo_rating_dif','avg_rank_dif','avg_rating_dif']]
# plt.style.use('ggplot')
# plt.figure(dpi=200)

# correlation_matrix=df_xG_final_simp.corr()
# print(correlation_matrix)

# #Mask for triangle graph 
# mask2 = np.zeros_like(correlation_matrix)
# mask2[np.triu_indices_from(mask2)] = True

# #Draw heatmap
# heatmap=sns.heatmap(correlation_matrix, mask=mask2,annot=True,annot_kws={'size':6},fmt=".2f", vmin=-1, vmax=1,
#                     xticklabels=True, yticklabels=True,cbar_kws={"shrink": .52})

# heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize = 6) #Set y-axis label size
# heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90, fontsize = 6) #Set x-axis label size

# heatmap.set_title('Correlation Heatmap-simp', fontdict={'fontsize':8}, pad=8) #Set title size

# cbar = heatmap.collections[0].colorbar
# cbar.ax.tick_params(labelsize=7) #Set colorbar size

# plt.show()