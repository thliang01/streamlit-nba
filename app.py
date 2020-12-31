# -*- coding: utf-8 -*-
# Copyright 2020 thliang01
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

# Download a single file and make its content available as a string.


# @st.cache(show_spinner=False)
# def get_file_content_as_string(path):
#     url = 'https://github.com/thliang01/streamlit-nba/master/' + path
#     response = urllib.request.urlopen(url)
#     return response.read().decode("utf-8")


# if __name__ == "__main__":
#     main()
