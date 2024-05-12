# From Balance Sheets to Beats: Analysing Popular Lyrics across Economic Conditions


This is my text analysis project for ECON1680.

**Folders**:

*data*: Contains all the source data files, along with combined data used for analysis

*code*: Code for data generation, preprocess and training

*output*: Summary statistics, results, and graphs

**File Descriptions**:

- `code/filter_data.py`: This filters in the relevant data from the overall
list of popular songs around the world downloaded from Kaggle 

- `code/get_audio_lyrics.py`: Extracts the features and lyrics of each song using the 
Spotify API and RapidAPI

- `code/cleaning_emotions.py`: Cleans the lyrics for text analysis, translates
those lyrics which are not in English, and then extracts emotions for each lyric

- `code/process_econ.py`: Cleans and concatenates monthly unemployment data for 
the US, UK and Canada for analysis

- `code/preprocess.py`: Actual preprocessing for our economic analysis. Reads in the 
song emotions data and joins with relevant economic data, and preprocesses. 
Splits the data into training and testing sets.

- `code/main.py`: For calling, preprocessing and analysing the data. Has OLS and MLP 
models

**How to Run**

The data needs to first be generated. First download the dataset from Kaggle at 
https://www.kaggle.com/ds/1265407 .
Then run the following files in order:
1. `filter_data.py`
2. `get_audio_lyrics.py`
3. `cleaning_emotions.py`

Also run `process_econ.py` separately to combine unemployment data into one file

This prepares all the music data we need. After this, to run the analysis,
simply run `main.py`.

You can read the written paper here: *Econ1680_Draft_Project2.pdf*.

