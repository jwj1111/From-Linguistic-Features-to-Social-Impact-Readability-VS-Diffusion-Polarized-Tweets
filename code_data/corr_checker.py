import pandas as pd
import numpy as np
import pingouin as pg
import os

def load_data(data_path):
    '''
    data_path: path of all_with_readability.csv
    '''
    data_df=pd.read_csv(data_path,encoding='utf-8')
    return data_df

def calculate_spread(data_df):
    '''
    data_df: dataframe from load_data()
    '''
    metrics=['likes','replies','retweets','quotes']
    metrics_df=data_df[metrics].copy()
    y_min,y_max=0.02,1.0
    normalized_data_df=pd.DataFrame()
    for metric in metrics:
        x=metrics_df[metric]
        x_min,x_max=x.min(),x.max()
        if (x_max-x_min)==0:
            normalized_col=np.full(len(x),y_min)
        else:
            normalized_col=(y_max-y_min)*(x-x_min)/(x_max-x_min)+y_min #归一化到[0.02,1]
        normalized_data_df[metric]=normalized_col
    prob_df=normalized_data_df.apply(lambda x: x/x.sum(),axis=0) #概率矩阵
    num_rows=len(normalized_data_df)
    k=1/np.log(num_rows)
    entropy_values=-k*(prob_df*np.log(prob_df.replace(0,1e-10))).sum(axis=0) #每个指标的信息熵
    redundancy=1-entropy_values #信息冗余度
    weights=redundancy/redundancy.sum() #基于信息冗余度判断区别性指标
    spread_scores=np.dot(normalized_data_df,weights)
    data_df['spread']=spread_scores
    return data_df

def correlation_analysis(data_df):
    '''
    data_df: dataframe from calculate_spread()
    '''
    data_df=data_df[data_df['type']=='polarized']
    readability_metrics=['char_per_word','syll_per_word','complex_ratio','long_ratio','difficult_ratio']
    results=[]
    y=data_df['spread'].dropna()
    normal_y=pg.normality(y)['normal'].iloc[0]
    for metric in readability_metrics:
        x=data_df[metric].dropna()
        normal_x=pg.normality(x)['normal'].iloc[0]
        if normal_x and normal_y:
            stats=pg.corr(x,y,method='pearson')
            method='Pearson'
        else:
            stats=pg.corr(x,y,method='spearman')
            method='Spearman'
        results.append({
            'metric':metric,
            'correlation_method':method,
            'correlation':round(stats['r'].iloc[0],4),
            'p_value':round(stats['p-val'].iloc[0],4),
            'significance':"True" if stats['p-val'].iloc[0]<0.05 else "False",
            'normal_metric':str(normal_x),
            'normal_spread':str(normal_y),
        })
    results_df=pd.DataFrame(results)
    return results_df

def save_data(result_df,filename):
    '''
    result_df: df from correlation_analysis
    filename: name of the output csv file (xxx.csv)
    '''
    os.makedirs(r'report/correlation',exist_ok=True)
    result_df.to_csv(fr'''report/correlation/{filename}''',encoding='utf-8',index=False)
    print(fr'''data saved: report/correlation/{filename}''')

def check_corr(data_path,filename):
    '''
    data_path: path of all_with_readability.csv
    filename: name of the output csv file (xxx.csv)
    '''
    data_df=load_data(data_path)
    data_df1=calculate_spread(data_df)
    result_df=correlation_analysis(data_df1)
    save_data(result_df,filename)