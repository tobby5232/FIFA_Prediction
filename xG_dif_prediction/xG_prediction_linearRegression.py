# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 01:07:02 2022

@author: ch406
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np


#%%
# Import xG data
df_xG_final = pd.read_csv('xG_data.csv')

print(df_xG_final.info)

#%%
#Plot correlation matrix

df_xG_final_simp=df_xG_final[['goal_dif','xG_dif','Elo_ranking_dif',
                              'Elo_rating_dif','avg_rank_dif','avg_rating_dif']]

# df_xG_final_simp=df_xG_final[['xG_dif','rank_dif','points_dif',
#                               'ovr_dif','att_dif','mid_dif','defe_dif',
#                               'Elo_ranking_dif','Elo_rating_dif']]
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

heatmap.set_title('Correlation Heatmap-simp', fontdict={'fontsize':8}, pad=8) #Set title size

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=7) #Set colorbar size

plt.show()

#%%
#xG_predicted method

def xG_predicted_method (X):
    from sklearn.linear_model import LinearRegression
    linearMmodel = LinearRegression(fit_intercept=True)
    linearMmodel.fit(df_xG_final_simp[['Elo_rating_dif']], df_xG_final_simp['xG_dif'])
    xG_predicted = linearMmodel.predict(X)
    return xG_predicted

#%%
#Plot scatter for Elo_rating_dif & goal_dif(predict_goal_dif)


from sklearn.linear_model import LinearRegression

# 建立LinearRegression模型
linearMmodel = LinearRegression(fit_intercept=True)
# 使用訓練資料訓練模型
linearMmodel.fit(df_xG_final_simp[['Elo_rating_dif']], df_xG_final_simp['xG_dif'])
# 使用訓練資料預測
xG_predicted = linearMmodel.predict(df_xG_final_simp[['Elo_rating_dif']])

from sklearn import metrics
print('R2 score: ', linearMmodel.score(df_xG_final_simp[['Elo_rating_dif']], df_xG_final_simp['xG_dif']))
mse = metrics.mean_squared_error(df_xG_final_simp['xG_dif'], xG_predicted)
print('MSE score: ', mse)

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True,dpi=200)
plt.scatter(df_xG_final_simp['Elo_rating_dif'], df_xG_final_simp['xG_dif'],s=10, label='origin')
plt.plot(df_xG_final_simp['Elo_rating_dif'], xG_predicted,label='xG_predicted',color='black')
plt.xlabel('Elo_rating_dif')
plt.ylabel('xG_dif')
plt.legend()
plt.show()