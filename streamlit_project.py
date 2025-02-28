import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.stats import binned_statistic_2d
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon, Arrow, ArrowStyle,FancyArrowPatch, Circle,FancyArrow
from mplsoccer.pitch import Pitch, VerticalPitch
from matplotlib.colors import Normalize
from matplotlib import cm
from highlight_text import fig_text, ax_text


import warnings
warnings.filterwarnings("ignore")

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import matplotlib.image as mpimg
import matplotlib.patches as patches
from io import BytesIO
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.markers import MarkerStyle
from mplsoccer import Pitch, VerticalPitch, FontManager, Sbopen, add_image
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.patheffects import withStroke, Normal
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer.utils import FontManager
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.cluster import KMeans
import warnings
from highlight_text import ax_text, fig_text
from PIL import Image
from urllib.request import urlopen
import os
import time
from unidecode import unidecode
from scipy.spatial import ConvexHull
URL = 'https://github.com/googlefonts/BevanFont/blob/main/fonts/ttf/Bevan-Regular.ttf?raw=true'
fprop = FontManager(URL).prop
img_save_loc='D:\CoCoding\DJ\Football Analytics\Data Visualization\\MatchReports'

st.title("칼리아리 vs 유벤투스 기회 창출")
st.subheader("필터를 통해 기회 창출한 패스들을 보세요!")

# Load Data
df = pd.read_csv('CagJuv.csv')

# Streamlit UI
st.title("칼리아리 vs 유벤투스 기회 창출")
st.subheader("필터를 통해 기회 창출한 패스들을 보세요!")

# Column mapping
column_mapping = {
    'match_id':'matchId',
    'event_id': 'eventId',
    'expanded_minute': 'expandedMinute',
    'minute': 'minute',
    'second': 'second',
    'team_id': 'teamId',
    'player_id': 'playerId',
    'x': 'x',
    'y': 'y',
    'end_x': 'endX',
    'end_y': 'endY',
    'satisfied_events_types': 'satisfiedEventsTypes',
    'is_touch': 'isTouch',
    'blocked_x': 'blockedX',
    'blocked_y': 'blockedY',
    'goal_mouth_z': 'goalMouthZ',
    'goal_mouth_y': 'goalMouthY',
    'related_event_id': 'relatedEventId',
    'related_player_id': 'relatedPlayerId',
    'is_shot': 'isShot',
    'card_type': 'cardType',
    'is_goal': 'isGoal',
    'outcome_type': 'outcomeType',
    'period_display_name': 'period',
    'shirt_no': 'shirtNo',
    'is_first_eleven': 'isFirstEleven'
}
df = df.rename(columns=column_mapping)

# Remove set-piece passes
df = df[~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn', na=False)]

# Select Team & Player
team = st.selectbox('팀을 선택하세요', df['teamId'].sort_values().unique(), index=None)
player = st.selectbox('Select a player', df[df['teamId'] == team]['name'].sort_values().unique(), index=None)

# Function to filter data
def filter_data(df, team, name):
    if team:
        df = df[df['teamId'] == team]  
    if name:
        df = df[df['name'] == name]  
    return df  # Always return DataFrame (Never None)

filtered_df = filter_data(df, team, player)

# Ensure `filtered_df` is valid before plotting
if filtered_df is None or filtered_df.empty:
    st.warning("No data available for the selected player.")
    filtered_df = pd.DataFrame(columns=df.columns)  # Avoid errors

# Create Pitch
pitch = VerticalPitch(pitch_type='opta', pitch_color='#ffffff', line_color='#000000', line_zorder=0.1, linewidth=0.5)
fig, ax = pitch.draw(figsize=(10, 10))

# Function to plot passes
def plot_chances_created(df, ax, pitch):
    if df is None or df.empty:
        return  # Prevents errors when no data is available
    
    df = df[df['qualifiers'].str.contains('KeyPass|BigChanceCreated|IntentionalGoalAssist', na=False)]
    
    for x in df.to_dict(orient='records'):


        pitch.lines(
        xstart=float(x['x']),
        ystart=float(x['y']),
        xend=float(x['endX']),
        yend=float(x['endY']),
        lw=3, comet=True, color='#FD4890', ax=ax, alpha=0.5  # Blue for Key Passes
    )
        pitch.scatter(
            x=float(x['endX']),  
            y=float(x['endY']),
            s=55,
            color='#FD4890' if 'IntentionalGoalAssist' in x['qualifiers'] else 'white', 
            edgecolor='white' if 'IntentionalGoalAssist' in x['qualifiers'] else '#FD4890',  
            linewidth=0.1 if 'IntentionalGoalAssist' in x['qualifiers'] else 1, 
            zorder=2, 
            ax=ax
        )

# Plot Passes
plot_chances_created(filtered_df, ax, pitch)

# Annotate Pitch
ax.text(50, 102, "CHANCES CREATED", color='#000000', va='bottom', ha='center', fontsize=10)
ax.annotate(xy=(104, 48), text='Attack', ha='center', color='#000000', rotation=90, fontsize=13)
ax.annotate(
    xy=(102, 58), xytext=(102, 43), text='', ha='center',
    arrowprops=dict(arrowstyle='->, head_length=0.3, head_width=0.05', color='#000000', lw=0.5)
)

# Show Plot
st.pyplot(fig)








