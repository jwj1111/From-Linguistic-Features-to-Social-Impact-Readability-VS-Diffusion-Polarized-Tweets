import pandas as pd
import readability
import syntok.segmenter as segmenter
from tqdm import tqdm
import os

def load_data(data_path):
    '''
    data_path: the path of all_data.csv
    '''
    data_df=pd.read_csv(data_path,encoding='utf-8')
    return data_df

def tokenizing(text):
    '''
    text: text from data_df
    '''
    tokenized='\n\n'.join('\n'.join(' '.join(token.value.lower() for token in sentence) for sentence in paragraph) for paragraph in segmenter.analyze(text))
    return tokenized

def readability_cal(data_df):
    '''
    data_df: dataframe from all_data.csv
    '''
    text_list=data_df['text'].to_list()
    readability_metrics=[]
    for text in tqdm(text_list,total=len(text_list),desc='calculate readability...'):
        tokenized=tokenizing(text)
        try:
            readability_rslt=readability.getmeasures(tokenized,lang='en')
            readability_metric={}
            readability_metric['words']=readability_rslt['sentence info']['words']
            readability_metric['sylls']=readability_rslt['sentence info']['syllables']
            readability_metric['chars']=readability_rslt['sentence info']['characters']
            readability_metric['complex']=readability_rslt['sentence info']['complex_words']
            readability_metric['long']=readability_rslt['sentence info']['long_words']
            readability_metric['difficult']=readability_rslt['sentence info']['complex_words_dc']
            readability_metric['char_per_word']=readability_rslt['sentence info']['characters_per_word']
            readability_metric['syll_per_word']=readability_rslt['sentence info']['syll_per_word']
            readability_metric['complex_ratio']=readability_rslt['sentence info']['complex_words']/readability_rslt['sentence info']['words']
            readability_metric['long_ratio']=readability_rslt['sentence info']['long_words']/readability_rslt['sentence info']['words']
            readability_metric['difficult_ratio']=readability_rslt['sentence info']['complex_words_dc']/readability_rslt['sentence info']['words']
        except ValueError:
            readability_metric={'words':0,'sylls':0,'chars':0,'complex':0,'long':0,'difficult':0,'char_per_word':0,'syll_per_word':0,'complex_ratio':0,'long_ratio':0,'difficult_ratio':0}
        readability_metrics.append(readability_metric)
    readability_df=pd.DataFrame(readability_metrics)
    all_readability_df=pd.concat([data_df,readability_df],axis=1)
    return all_readability_df

def length_filter(all_readability_df):
    '''
    all_readability_df: data_df with readability_df
    '''
    length_filtered_df=all_readability_df.copy()
    length_filtered_df=length_filtered_df[length_filtered_df['words']>0]
    q1=length_filtered_df['words'].quantile(0.25)
    q3=length_filtered_df['words'].quantile(0.75)
    iqr=q3-q1
    lower_whisker=(q1-1.5*iqr)
    upper_whisker=q3+1.5*iqr
    low_num=lower_whisker if lower_whisker>10 else 10
    high_num=upper_whisker
    df_final=length_filtered_df[(length_filtered_df['words']>=low_num)&(length_filtered_df['words']<=high_num)]
    print(f'''shortest {df_final['words'].agg('min')} | longest {df_final['words'].agg('max')}''')
    os.makedirs('read_results',exist_ok=True)
    df_final.to_csv(r'''read_results/all_with_readability.csv''',encoding='utf-8',index=False)
    print(f'''result saved: read_results/all_with_readability.csv''')

def readability_calculate(data_path):
    '''
    data_path: the path of all_data.csv
    '''
    data_df=load_data(data_path)
    all_readability_df=readability_cal(data_df)
    length_filter(all_readability_df)