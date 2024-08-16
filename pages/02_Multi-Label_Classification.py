import pandas as pd
import numpy as np
import streamlit as st
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.max_open_warning'] = 0
matplotlib.use('Agg')
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.colors as pc
from PIL import Image
import io
import ast
import random





#DATASET#####

#################################################################### Data load
df = pd.read_csv("df_streamlit.csv") #Load dataset
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') #Set date data as date datatype
df['month'] = pd.to_datetime(df['month'], format='%Y-%m') #Set date data as date datatype
df['labels'] = df['labels'].apply(ast.literal_eval) #Convert labels datatype as python list
df['labels'] = df['labels'].apply(sorted) #Sort each list of labels
df['labels_string'] = df['labels'].apply(lambda x: ', '.join(x)) #Create a column with labels as list of strings





#STREAMLIT

#################################################################### Configuration
st.set_page_config(layout="wide") #Set wide page layout


# Custom CSS
st.markdown("""
    <style>
    h1 {  /* Title font size*/
        font-size: 40px;
    }
    h2 {  /* Header font size*/
        font-size: 30px;
    }
    h5 {  /* Graph title font size*/
    font-size: 20px;
    }
    p {
        font-size: 15px;  /* Paragraph font size */
    }
    .st-b1 {
        font-size: 8.5px;  /* Arrondissement name font size */
    }
    </style>
    """, unsafe_allow_html=True)





#################################################################### Layout
st.title("Airbnb Guest Reviews in Paris") # Page title
st.header("Multi-Label Classification") #Page subtitle
st.markdown("***") #Breakline
st.write('''
This page presents the results of topic classification for guest reviews across different time periods in Paris. The classification model categorizes comments into five main topics: Apartment, Bed, Communication, Location, Neighborhood. Based on the content of each review, one or multiple relevant topics are assigned. You can customize your view by selecting time period of interest and specific area in Paris.
 ''')


#################################################################### Filters
col1, col2 = st.columns([2, 3], gap = 'medium') #Create two columns for period & arrondissement filters

with col1: #On the first column
    min_date = df['month'].min().to_pydatetime() #Set minimum date
    max_date = df['month'].max().to_pydatetime() #Set maximum date
    selected_min, selected_max = st.slider( #Create slider
        "Period", #Slider name
        value=(min_date, max_date), #Slider values
        min_value=min_date, #Minimum value
        max_value=max_date, #Maximum value
        format="YYYY-MM" #Format
    )

with col2: #On the second column
    arrondissement_all = ["1 - Louvre","2 - Bourse","3 - Temple","4 - Hôtel-de-Ville","5 - Panthéon",
                          "6 - Luxembourg","7 - Palais-Bourbon","8 - Élysée","9 - Opéra", "10 - Entrepôt",
                          "11 - Popincourt","12 - Reuilly","13 - Gobelins","14 - Observatoire","15 - Vaugirard",
                          "16 - Passy","17 - Batignolles-Monceau","18 - Buttes-Montmartre","19 - Buttes-Chaumont","20 - Ménilmontant"]
    container = st.container() #Create a container for arrondissement filter
    if 'selected' not in st.session_state: #Create session_state for selected arrondissement
        st.session_state.selected = arrondissement_all #Set default selection of arrondissement
    if st.button('Select all arrondissements'): #Create button for all selection
        st.session_state.selected = arrondissement_all #If the button is clicked, all arrondissement is selected
    arrondissement = container.multiselect("Arrondissement:", arrondissement_all, default=["1 - Louvre","2 - Bourse","3 - Temple","4 - Hôtel-de-Ville","5 - Panthéon"]) #Create arrondissement mutiselect filter
    st.session_state.selected = arrondissement #Update selection at each select action is made

mask = (df['date'] >= selected_min) & (df['date'] <= selected_max) & (df['arrondissement'].isin(arrondissement)) #Create a mask with the filter selection
df_filtered = df[mask] #Select filtered data

labels_counts = df_filtered['labels_string'].value_counts().reset_index() #Create dataset with count of each combination of topics
labels_counts['percent'] = (labels_counts['count'] / labels_counts['count'].sum() * 100).round(2).astype(str) + '%' #Calculate the percentage of each combination
labels_counts['label_percent'] = '<b>'+labels_counts['labels_string']+'<br>'+labels_counts['percent'] #Create a column with the name and the percentage of each combination

df_exploded = df_filtered.explode('labels') #Create dataset with single topic 
df_exploded['labels'] = df_exploded['labels'].str.strip() #Remove space before and after each topic extracted
labels_counts_exploded = df_exploded['labels'].value_counts().reset_index() #Count the single topic


#################################################################### Graphs

col1, col2 = st.columns([3, 2], gap="large") #Create two columns for two graphs

with col1: #On the first column
    fig = px.treemap(labels_counts, #Create a treemap 
                     path=['label_percent'], 
                     values='count', 
                     title='Multi-Topics of guest reviews', 
                     color='count', 
                     color_continuous_scale=['#FBD2C5','#00A699'])
    fig.update_traces(textposition='middle center', #Update text configuration in the treemap
                      insidetextfont=dict(size=15) 
                      )
    fig.update_layout(width=1000, #Update the dimension of the graph
                      height=800, 
                      coloraxis_showscale=False, #Hide colorbar
                      title_font_size=17, #Update title configuration
                      title_xanchor='center',
                      title_x = 0.5
                      )
    st.plotly_chart(fig, use_container_width=False) #Show the graph

with col2: #On the second column
    fig = px.histogram(labels_counts_exploded, #Create barplot
                       y='labels', 
                       x= 'count', 
                       title='Count of each topics', 
                       color_discrete_sequence=['#FF5A5F'])
    fig.update_layout(width = 500, #Update the dimension of the graph
                      height = 780,
                      title_font_size=17, #Update the title configuration
                      title_xanchor='center',
                      title_x = 0.6,
                      yaxis_title=None, #Update axis title configuration
                      xaxis_title=None,
                      yaxis_tickfont=dict(size=15), #Update axis ticks configuration
                      xaxis_tickfont=dict(size=15),
                      )
    fig.update_yaxes(tickfont=dict(color='white'),categoryorder='category descending') #Update y-axes configuration
    fig.update_xaxes(tickfont=dict(color='white')) #Update x-axes configuration
    st.plotly_chart(fig) #Show the plot

st.markdown("<h5 style='text-align: center;'>Guest Reviews</h5>", unsafe_allow_html=True) #Title
col1, col2, col3 = st.columns([4,2,1]) #Create columns for filters
with col1: #On the first column
    topic = st.multiselect("Select Topics", ['apartment', 'bed', 'communication', 'location', 'neighborhood'], default=['apartment', 'bed', 'communication', 'location', 'neighborhood']) #Create topic selector
with col3: #On the last column
    st.write('') #Blank line
    if st.button('Refresh comments'): #Create refresh button
        comments = df_filtered.loc[df_filtered['labels'].apply(lambda x: x == list(sorted(topic))),'comments_en'].sample(5) #If clicked, comments are refreshed

comments = df_filtered.loc[df_filtered['labels'].apply(lambda x: x == list(sorted(topic))),'comments_en'].sample(5) #Create 5 random comments which correspond to applied filters
with st.container(border=True): #Create a container for the comments
    for index in range(len(comments)): #Show each comments generated in plain text
        st.write(index+1, comments.iloc[index])

