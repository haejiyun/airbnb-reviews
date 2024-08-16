import pandas as pd
import numpy as np
import streamlit as st
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['figure.max_open_warning'] = 0
matplotlib.use('Agg')
import plotly.express as px
import datetime
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image
import re





#################################################################### Streamlit Configuration
#st.set_option('deprecation.showPyplotGlobalUse', False) #Remove depreciation warning on pyplot
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
    font-size: 18px;
    }
    p {
        font-size: 15px;  /* Paragraph font size */
    }
    .st-b1 {
        font-size: 9px;  /* Arrondissement name font size */
    }
    [data-testid="stSidebar"] {
        width: 240px;  /* Sidebar width */
        min-width: 240px;  /* Sidebar minimum width */
        max-width: 240px;  /* Sidebar maximum width */
    }
    </style>
    """, unsafe_allow_html=True)





#DATASET#####

#################################################################### Data load
df = pd.read_csv('df_streamlit.csv') #Load 

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') #Set date data as date datatype
df['month'] = pd.to_datetime(df['month'], format='%Y-%m') #Set date data as date datatype

quartiers_gdf = gpd.read_file('quartier_paris.geojson') #Load quartier location data
quartiers_gdf = quartiers_gdf[['l_qu','geometry']].rename(columns = {'l_qu':'quartier'}) #Select using columns and rename

arrondissement_gdf = gpd.read_file('neighbourhoods.geojson') #Load arrondissement location data
arrondissement_gdf = arrondissement_gdf[['neighbourhood','geometry']] #Select using columns and rename





#################################################################### Layout
st.title("Airbnb Guest Reviews in Paris") #Page title
st.header("Sentiment Analysis") #Page subtitle
st.markdown("***") #Breakline
st.write('''
This page displays sentiment analysis results for guest reviews of Paris zones across different time periods. Based on the content of each comment, a sentiment score from 1 to 5 is assigned, where 1 represents highly negative sentiment and 5 indicates positive sentiment. You can customize your view by selecting time period of interest and specific area in Paris.  
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


#################################################################### Graphs

# Chorolepleth
col1, col2 = st.columns([2, 1.5], gap = 'small') #Create two columns for two graphs

with col1: #On the first column
    col1_bis, col2_bis= st.columns([1,7]) #Create a sub-columns for sub-filteres
    with col1_bis: #On the first sub-column
        st.markdown("<p style='line-height:2.4;'>Show by :</p>", unsafe_allow_html=True)
        #st.write("Show by :") #Show the name of filter
    with col2_bis: #On the second sub-column
        zone = st.radio("", options=["Arrondissement", "Quartier"], horizontal=True, label_visibility="collapsed") #Create area division option filter
    if zone == "Quartier": #If quartier is selected
        df_filtered_zone = df_filtered[['quartier','sentiment']].groupby('quartier').mean().reset_index() #Create dataset grouped by quartier
        gdf_zone = gpd.GeoDataFrame(pd.merge(df_filtered_zone, quartiers_gdf, on='quartier'), geometry='geometry') #Add quartier geolocalisation in the dataset
    elif zone == "Arrondissement": #If arrondissement is selected
        df_filtered_zone = df_filtered[['arrondissement','neighbourhood','sentiment']].groupby(['arrondissement','neighbourhood']).mean().reset_index() #Create dataset grouped by arrondissement
        gdf_zone = gpd.GeoDataFrame(pd.merge(df_filtered_zone, arrondissement_gdf, on='neighbourhood'), geometry='geometry') #Add arrondissement geolocalisation in the dataset

    choropleth = px.choropleth_mapbox(gdf_zone, #Create the choropleth
                            geojson= gdf_zone.geometry, 
                            locations=gdf_zone.index,
                            color='sentiment',
                            color_continuous_scale=['#FF5A5F','#00A699'],
                            range_color=[0, 5],
                            mapbox_style="carto-positron",
                            zoom=11.1, 
                            center={"lat": 48.86, "lon": 2.345},
                            #opacity=0.8,
                            height=350,
                            width=1300)
    choropleth.update_traces(marker_line_width=0, #Update market configuration
                             marker_opacity=0.8)
    choropleth.update_layout(coloraxis_colorbar={'lenmode': 'pixels','len': 525,'yanchor':'bottom','y': 0}, #Update colorbar configuration
                             margin=dict(l=5, r=5, t=0, b=50)) #Update margins
    st.plotly_chart(choropleth) #Show the choropleth

# Wordcloud
#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))
#stop_words.add('br')
#stop_words.add('\n')
with col2: #On the second column
    comments_pos = df_filtered.loc[df_filtered['sentiment_binary'] == 'positive', 'comments_en'].astype(str).str.replace('\n', ' ').dropna() #Extract positive comments
    comments_pos = ' '.join(comments_pos) #Extract words
    comments_neg = df_filtered.loc[df_filtered['sentiment_binary'] == 'negative', 'comments_en'].astype(str).str.replace('\n', ' ').dropna() #Extract negative comments
    comments_neg = ' '.join(comments_neg) #Extract words
    mask_pos = np.array(Image.open('circle_up.png')) #Import wordcloud mask shape for positive comments
    mask_neg = np.array(Image.open('circle_down.png')) #Import wordcloud mask shape for negative comments
    wordcloud_pos = WordCloud(colormap=LinearSegmentedColormap.from_list("custom", ['#00A699', '#767676','#79CCCD']), mask = mask_pos, background_color=None, mode="RGBA", max_words=100).generate(comments_pos) #Create wordcloud for positive comments
    wordcloud_neg = WordCloud(colormap=LinearSegmentedColormap.from_list("custom", ['#FF5A5F', '#FBD2C5','#FFF6E6']), mask = mask_neg, background_color=None, mode="RGBA",  max_words=100).generate(comments_neg) #Create wordcloud for negative comments
    st.write("") #Blank line
    wordcloud = plt.figure( facecolor = 'none') #Set graph background transparent
    ax1 = wordcloud.add_axes([0, 0, 1, 1]) #Configure wordcloud position for positive comments
    ax2 = wordcloud.add_axes([0, -0.7, 1, 1]) #Configure wordcloud position for negative comments
    ax1.imshow(wordcloud_pos) #Show wordcloud of positive comments
    ax1.set_title('Positive', color = '#00A699', size=20) #Title
    ax1.axis('off') #Remove axis
    ax2.imshow(wordcloud_neg) #Show wordcloud of negative comments
    ax2.set_title('Negative', color='#FF5A5F', size=20, y=-0.1) #Title
    ax2.axis('off') #Remove axis
    st.pyplot(wordcloud) #Remove axis

# Linechart
st.markdown("<h5 style='text-align: center;'>Daily Average Sentiment Score</h5>", unsafe_allow_html=True) #Linechart title
df_sentiment = df_filtered.set_index('date').resample('D')['sentiment'].mean().reset_index() #Create monthly data
linechart = px.line(df_sentiment, x='date', y='sentiment') #Create linechart
linechart.update_xaxes(dtick='D1', #Update ticks configuration
                       tickangle=-90, 
                       tickfont=dict(size=8)) #
linechart.update_layout(yaxis_title="Score", #Update y-axis title
                        yaxis_title_font=dict(size=15), 
                        xaxis_title=None, #Update x-axis title
                        yaxis=dict(range=[0, 5.2]),
                        yaxis_tickfont=dict(size=13), #Update y-ticks configuration
                        xaxis_tickfont=dict(size=13)) #Update x-ticks configuration
linechart.update_layout(margin=dict(l=60, r=60, t=5, b=115)) #Update margin
st.plotly_chart(linechart) #Show the chart
