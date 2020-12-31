import datetime
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

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

st.title('NBA games Data Visualization')
st.header('Dataset with all NBA games'
          ' from 2004 season to dec 2020')


st.title('Who are the players '
         'with most games played ?')

# * optional kwarg unsafe_allow_html = True


# if __name__ == "__main__":
#     main()
