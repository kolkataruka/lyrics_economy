import pandas as pd

MONTH_DICT = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12
}

usa_df = pd.read_csv('../data/united_states.csv')[['Year', 'Period', 'Value']]
usa_df = usa_df.rename(columns={'Year': 'year', 'Period': 'month', 'Value': 'unemployment'})
usa_df['month'] = usa_df['month'].str.extract(r'(\d+)').astype(int)
usa_df = usa_df.assign(region='United States')

canada_df = pd.read_csv('../data/canada.csv')[['REF_DATE', 'GEO', 'VALUE']]
canada_df = canada_df.rename(columns={'VALUE': 'unemployment', 'GEO': 'region'})
canada_df['year'] = canada_df['REF_DATE'].apply(lambda x: x.split('-')[0])
canada_df['month'] = canada_df['REF_DATE'].apply(lambda x: x.split('-')[1])


uk_df = pd.read_csv('../data/united_kingdom.csv')[['Title', 'Unemployment rate (aged 16 and over, seasonally adjusted): %']]
uk_df = uk_df.rename(columns={'Unemployment rate (aged 16 and over, seasonally adjusted): %': 'unemployment'})
uk_df['year'] = uk_df['Title'].apply(lambda x: int(x.split(' ')[0]))
uk_df['month'] = uk_df['Title'].apply(lambda x: MONTH_DICT[x.split(' ')[1]])
uk_df = uk_df.assign(region='United Kingdom')

merged_df = pd.concat([usa_df, canada_df[['year', 'month', 'unemployment', 'region']]], ignore_index=True)
final_df = pd.concat([merged_df, uk_df[['year', 'month', 'unemployment', 'region']]], ignore_index=True)

final_df.to_csv('../data/unemployment.csv')
