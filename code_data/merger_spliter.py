import pandas as pd
import os

def data_merge(directory_path='senti_results'):
    '''
    directory_path: directory senti_results
    '''
    output_file_name='all_data.csv'
    files=os.listdir(directory_path)
    csv_files=[file for file in files if file.endswith('.csv')]
    dataframes=[]
    for csv_file in csv_files:
        category=csv_file.split('.')[0]
        full_path=os.path.join(directory_path,csv_file)
        df=pd.read_csv(full_path,encoding='utf-8')
        df['from']=category
        dataframes.append(df)
    all_data_df=pd.concat(dataframes,axis=0)
    all_data_df.to_csv(output_file_name,encoding='utf-8',index=False)
    print(f'''all_data_df saved: {output_file_name}''')

def data_split(merged_info_readability_path=r'all_with_readability.csv'):
    '''
    merged_info_readability_path: filepath of merged_info_readability_csv (all_with_readability.csv by Default)
    '''
    df=pd.read_csv(merged_info_readability_path,encoding='utf-8')
    categories=df['from'].unique().tolist()
    os.makedirs('read_results',exist_ok=True)
    for category in categories:
        df_cat=df[df['from']==category]
        df_cat.to_csv(fr'''read_results/{category}.csv''',encoding='utf-8',index=False)
        print(f'''data saved: read_results/{category}.csv''')