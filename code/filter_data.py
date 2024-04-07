import pandas as pd

countries = ['United States']
init_df = pd.read_csv('../data/charts.csv')
filtered_df = init_df[init_df['region'].isin(countries)]
filtered_df = filtered_df[filtered_df['chart'] == 'top200']
filtered_df = filtered_df[(filtered_df['rank'] >= 1) & (filtered_df['rank'] <= 10)][['title', 'rank', 'date', 'artist', 'url', 'region']]
filtered_df['date'] = pd.to_datetime(filtered_df['date']).dt.date
filtered_df['id'] = filtered_df['url'].str.split('/').str[-1]
filtered_df['year'] = pd.to_datetime(filtered_df['date']).dt.year
filtered_df['month'] = pd.to_datetime(filtered_df['date']).dt.month
filtered_df.to_csv('../data/all_data.csv')
print(filtered_df.head())
print(len(filtered_df))
print(filtered_df.describe())
song_counts = filtered_df.groupby(['month', 'year', 'id']).size().reset_index(name='count')
song_counts_sorted = song_counts.sort_values(by=['year','month', 'count'], ascending=[True, True, False])
top_songs_by_month = song_counts_sorted.groupby(['month', 'year']).head(10)
final_top_100_df = pd.merge(top_songs_by_month, filtered_df, on=['month', 'year', 'id'], how='inner').drop_duplicates(subset=['month', 'year', 'id'])
final_top_100_df.to_csv('../data/top_songs.csv')