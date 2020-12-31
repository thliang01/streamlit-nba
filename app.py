# -*- coding: utf-8 -*-
# Copyright 2020 Thomas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# More info: https://github.com/thliang01/streamlit-nba

import urllib
import os
import datetime
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
st.set_option('deprecation.showPyplotGlobalUse', False)

# ----- Load data -----

games_details = pd.read_csv('data/NBA-games-data/games_details.csv')
# st.dataframe(games_details.head(5))

players = pd.read_csv('data/NBA-games-data/players.csv')
# st.dataframe(players.head(5))

teams = pd.read_csv('data/NBA-games-data/teams.csv')
# st.dataframe(teams.head(5))

ranking = pd.read_csv('data/NBA-games-data/ranking.csv')
# st.dataframe(ranking.head(5))

games = pd.read_csv('data/NBA-games-data/games.csv')
# st.dataframe(games.head(5))

df = pd.read_csv('data/NBA2K/nba2k20-full.csv')
# st.write(df.head(5))

# ----- ----- -----

# Streamlit encourages well-structured code, like starting execution in a main() function.
# def main():
#     # Render the readme as markdown using st.markdown.
#     readme_text = st.markdown(get_file_content_as_string("instructions.md"))

st.title('NBA games Data Visualization')
"""
### Dataset with all NBA games from 2004 season to dec 2020
---
"""

st.header('Who are the players with most games played ?')


def plot_top(df, column, label_col=None, max_plot=5):
    top_df = df.sort_values(column, ascending=False).head(max_plot)

    height = top_df[column]
    x = top_df.index if label_col is None else top_df[label_col]

    gold, silver, bronze, other = ('#FFA400', '#bdc3c7', '#cd7f32', '#3498db')
    colors = [gold if i == 0 else silver if i == 1 else bronze if i ==
              2 else other for i in range(0, len(top_df))]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(x, height, color=colors)
    plt.xticks(x, x, rotation=60)
    plt.xlabel(label_col)
    plt.ylabel(column)
    plt.title(f'Top {max_plot} of {column}')
    plt.show()


players_name = games_details['PLAYER_NAME']
val_cnt = players_name.value_counts().to_frame().reset_index()
val_cnt.columns = ['PLAYER_NAME', 'Number of games']

st.pyplot(
    plot_top(
        val_cnt,
        column='Number of games',
        label_col='PLAYER_NAME',
        max_plot=10))

st.markdown('---')

st.header('Is most game played means most time played ?')


def convert_min(x):
    if pd.isna(x):
        return 0
    x = str(x).split(':')
    if len(x) < 2:
        return int(x[0])
    else:
        return int(x[0]) * 60 + int(x[1])


df_tmp = games_details[['PLAYER_NAME', 'MIN']]
df_tmp.loc[:, 'MIN'] = df_tmp['MIN'].apply(convert_min)
agg = df_tmp.groupby('PLAYER_NAME').agg('sum').reset_index()
agg.columns = ['PLAYER_NAME', 'Number of seconds played']

st.pyplot(
    plot_top(
        agg,
        column='Number of seconds played',
        label_col='PLAYER_NAME',
        max_plot=10))

st.markdown('And the answer is yes ! LeBron James is truly a living legend !')
st.markdown('---')


# Download a single file and make its content available as a string.


# @st.cache(show_spinner=False)
# def get_file_content_as_string(path):
#     url = 'https://github.com/thliang01/streamlit-nba/master/' + path
#     response = urllib.request.urlopen(url)
#     return response.read().decode("utf-8")


# if __name__ == "__main__":
#     main()
