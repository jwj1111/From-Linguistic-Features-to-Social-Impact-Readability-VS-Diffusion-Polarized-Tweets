import pandas as pd
import pingouin as pg
import os

def load_data(data_path):
    '''
    data_path: path of all_with_readability.csv
    '''
    data_df=pd.read_csv(data_path,encoding='utf-8')
    return data_df

def check_difference(data_df,alpha=0.05):
    '''
    data_df: data_df from load_data()
    alpha: threshold for p value (significance)
    '''
    metrics=['char_per_word','syll_per_word','complex_ratio','long_ratio','difficult_ratio']
    df1=data_df[data_df['type']=='polarized'][metrics]
    df2=data_df[data_df['type']=='neutral'][metrics]
    assert df1.columns.equals(df2.columns), "same column num required"
    results=[]
    for col in df1.columns:
        group1=df1[col].dropna()
        group2=df2[col].dropna()
        normal1=pg.normality(group1)['normal'].iloc[0]
        normal2=pg.normality(group2)['normal'].iloc[0]
        homoscedastic=pg.homoscedasticity([group1.to_numpy(),group2.to_numpy()])['equal_var'].iloc[0]
        if normal1 and normal2:
            if homoscedastic:
                stats=pg.ttest(group1,group2,correction=False)
                method="Independent t-test"
            else:
                stats=pg.ttest(group1,group2,correction=True)
                method="Welch's t-test"
            central_tendency="mean"
            pol_central=group1.mean()
            neu_central=group2.mean()
            effect_size=stats['cohen-d'].iloc[0]
            effect_type="Cohen's d"
        else:
            stats=pg.mwu(group1,group2,alternative='two-sided')
            method="Mann-Whitney U"
            central_tendency="median"
            pol_central=group1.median()
            neu_central=group2.median()
            effect_size=stats['RBC'].iloc[0]
            effect_type="Rank-biserial r"
        results.append({
            'metric':col,
            'method':method,
            'p':round(stats['p-val'].iloc[0],4),
            'significance':"True" if stats['p-val'].iloc[0]<alpha else "False",
            'effect_type':effect_type,
            'effect_size':round(effect_size,4),
            f'polarized_{central_tendency}':round(pol_central,4),
            f'neutral_{central_tendency}':round(neu_central,4),
            'normal_polarized':str(normal1),
            'normal_neutral':str(normal2),
            'homoscedastic':str(homoscedastic)
        })
    rslt_df=pd.DataFrame(results)
    return rslt_df

def save_data(rslt_df,filename):
    '''
    rslt_df: df from check_difference
    filename: name of the output csv file (xxx.csv)
    '''
    os.makedirs(r'report/difference',exist_ok=True)
    rslt_df.to_csv(fr'''report/difference/{filename}''',encoding='utf-8',index=False)
    print(fr'''data saved: report/difference/{filename}''')

def difference_check(data_path,filename):
    '''
    data_path: path of all_with_readability.csv
    filename: name of the output csv file (xxx.csv)
    '''
    data_df=load_data(data_path)
    rslt_df=check_difference(data_df,0.05)
    save_data(rslt_df,filename)