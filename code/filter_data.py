import pandas as pd


'''This file is to be run only once. It filters out the downloaded data for only
the specified countries, extracts the top 5 songs of eachh month,
and cleans it for for futher preprocessing'''

countries = ['United States', 'Canada', 'United Kingdom']
init_df = pd.read_csv('../data/charts.csv')
filtered_df = init_df[init_df['region'].isin(countries)]
filtered_df = filtered_df[filtered_df['chart'] == 'top200']
filtered_df = filtered_df[(filtered_df['rank'] >= 1) & (filtered_df['rank'] <= 10)][['title', 'rank', 'date', 'artist', 'url', 'region']]
filtered_df['date'] = pd.to_datetime(filtered_df['date']).dt.date
filtered_df['id'] = filtered_df['url'].str.split('/').str[-1] #Creating an id column, with spotify track ID for each song
filtered_df['year'] = pd.to_datetime(filtered_df['date']).dt.year
filtered_df['month'] = pd.to_datetime(filtered_df['date']).dt.month
filtered_df.to_csv('../data/all_data.csv')
print(filtered_df.head())
print(len(filtered_df))
print(filtered_df.describe())

#Now combining all the data such that we only retain the top 5 songs from each month
song_counts = filtered_df.groupby(['month', 'year', 'region', 'id']).size().reset_index(name='count')
song_counts_sorted = song_counts.sort_values(by=['year','month', 'region', 'count'], ascending=[True, True, True, False])
top_songs_by_month = song_counts_sorted.groupby(['month', 'region', 'year']).head(5)
final_top_100_df = pd.merge(top_songs_by_month, filtered_df, on=['month', 'year', 'region', 'id'], how='inner').drop_duplicates(subset=['month', 'year', 'region', 'id'])
final_top_100_df.to_csv('../data/top_songs.csv')