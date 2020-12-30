import datetime
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

games_details = pd.read_csv('data/NBA-games-data/games_details.csv')
st.dataframe(games_details.head(5))

players = pd.read_csv('data/NBA-games-data/players.csv')
st.dataframe(players.head(5))

teams = pd.read_csv('data/NBA-games-data/teams.csv')
st.dataframe(teams.head(5))

ranking = pd.read_csv('data/NBA-games-data/ranking.csv')
st.dataframe(ranking.head(5))

games = pd.read_csv('data/NBA-games-data/games.csv')
st.dataframe(games.head(5))


def print_missing_values(df):
    df_null = pd.DataFrame(len(df) - df.notnull().sum(), columns=['Count'])
    df_null = df_null[df_null['Count'] > 0].sort_values(
        by='Count', ascending=False)
    df_null = df_null / len(df) * 100

    if len(df_null) == 0:
        st.markdown('No missing value.')
        return

    x = df_null.index.values
    height = [e[0] for e in df_null.values]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x, height, width=0.8)
    plt.xticks(x, x, rotation=60)
    plt.xlabel('Columns')
    plt.ylabel('Percentage')
    plt.title('Percentage of missing values in columns')
    st.pyplot(plt.show())


def dataset_overview(df, df_name):
    st.markdown(f'### {df_name} dataset overview')
    st.markdown(f'dataset shape : {df.shape}')
    st.markdown(f'#### Display 5 first rows')
    st.markdown(df.head())
    st.markdown('*****')
    st.markdown(f'#### Describe dataset')
    st.markdown(df.describe().T)
    st.markdown('*****')
    st.markdown(f'#### Missing values')
    print_missing_values(df)

dataset_overview(games_details, 'games_details')

st.title("Let's take a look at the data")
df = pd.read_csv('data/NBA2K/nba2k20-full.csv')
st.dataframe(df.head(5))
