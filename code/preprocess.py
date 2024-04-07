from distutils.command.clean import clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def load_data():
    music_df = pd.read_csv('../data/song_emotions.csv')[['month','year', 'region', 'id', 'acoustic', 'dance', 'energy', 'instrumental', 'loudness', 'mode', 'tempo', 'valence', 'explicit', 'emotions']]
    unemployment_df = pd.read_csv('../data/unemployment.csv')[['Year', 'Period', 'Value']]
    unemployment_df = unemployment_df.rename(columns={'Year': 'year', 'Period': 'month', 'Value': 'unemployment'})
    unemployment_df['month'] = unemployment_df['month'].str.extract(r'(\d+)').astype(int)
    unemployment_df = unemployment_df.assign(region='United States')
    final_df = music_df.merge(unemployment_df, on=['year', 'month', 'region'], how='left')
    final_df.to_csv('../data/combined.csv')
    return final_df

def data_split(init_df):
    '''Normalizing and splitting data into training and testing sets'''

    init_df = init_df[['month','year', 'id', 'acoustic', 'dance', 'energy', 'instrumental', 'loudness', 'mode', 'tempo', 'valence', 'explicit', 'emotions', 'unemployment']]
    
    #init_df = init_df.dropna()
    #print(init_df.head())
    cols = init_df.columns
    #df_scaled = normalize(init_df) 

    #df_scaled = pd.DataFrame(df_scaled, columns=cols) 
    df_encoded = pd.get_dummies(init_df, columns=['emotions'], drop_first=True) #one hot encoding regime types and admin scale
    #print(df_scaled.head())
    y=init_df['emotions']
    X=init_df.drop(columns=['emotions'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1680)
    return X_train, X_test, y_train, y_test
