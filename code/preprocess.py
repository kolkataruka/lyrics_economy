import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

#To preprocess for analysis, after data has been all collected and is ready.


def load_data():
    '''
    Loads in the final music data (with emotions) and the unemployment data.
    Combines all the data into one csv file
    '''
    music_df = pd.read_csv('../data/song_emotions.csv')[['month','year', 'region', 'id', 'acoustic', 'dance', 'energy', 'instrumental', 'loudness', 'mode', 'tempo', 'valence', 'explicit', 'emotions', 'emotions_id', 'anger', 'love', 'sadness']]
    unemployment_df = pd.read_csv('../data/unemployment.csv')
    final_df = music_df.merge(unemployment_df, on=['year', 'month', 'region'], how='left')[['month','year', 'region', 'id', 'acoustic', 'dance', 'energy', 'instrumental', 'loudness', 'mode', 'tempo', 'valence', 'explicit', 'emotions', 'emotions_id', 'anger', 'love', 'sadness', 'unemployment']]
    final_df.to_csv('../data/combined.csv')
    return final_df

def data_split(init_df, ycol, norm):
    '''Normalizing, preprocessing and splitting data into training and testing sets'''

    init_df = init_df[['acoustic', 'dance', 'energy', 'instrumental', 'loudness', 'mode', 'tempo', 'valence', 'explicit', 'unemployment', 'region', ycol]]
    
    #One-hot encoding for fixed effects
    init_df = pd.get_dummies(init_df, columns=['region'], drop_first=True)
     
    
    if ycol == 'emotions_id':
        init_df = init_df[init_df['emotions_id'] <= 1]
        #print(len(init_df))
    #init_df = init_df.drop(columns=['region']).astype(float)
    
    y=init_df[ycol]
    X=init_df.drop(columns=[ycol])

    cols = X.columns
    if norm:
        new_df = normalize(X)
        X = pd.DataFrame(new_df, columns=cols, index=X.index)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1680)
    return X_train, X_test, y_train, y_test, cols
