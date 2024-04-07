# lyrics_economy
**Lyrics and Economics**
This is my text analysis project for ECON1680.

*File Descriptions*:

`code/filter_data.py`: This filters in the relevant data from the overall
list of popular songs around the world downloaded from Kaggle
`code/getdata.py`: Extracts the features and lyrics of each song using the 
Spotify API and RapidAPI
`code/cleaning_emotions.py`: Cleans the lyrics for text analysis, translates
those lyrics which are not in English, and then generates emotions for each lyric
`preprocess.py`: Actual preprocessing for our economic analysis. Reads in the 
song emotions data and joins with relevant economic data, and preprocesses. 
Splits the data into training and testing sets.
`main.py`: For calling, preprocessing and analysing the data

*How to Run*
The data needs to first be generated. First download the dataset from Kaggle.
Then run the following files in order:
1. `filter_data.py`
2. `getdata.py`
3. `cleaning_emotions.py`
This prepares all the music data we need. After this, to run the analysis,
simply run `main.py`

