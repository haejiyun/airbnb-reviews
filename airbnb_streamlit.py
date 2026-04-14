#################################################################### LIBRARIES IMPORT
import pandas as pd
import numpy as np
import streamlit as st
import geopandas as gpd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['figure.max_open_warning'] = 0
matplotlib.use('Agg')
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.colors as pc
import datetime
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image
import re
import io
import ast
import random
import time





#################################################################### STREAMLIT CONFIGURATION
st.set_page_config(page_title = "Airbnb Guest Reviews") #Set title to show on the tab of browser

# CSS
def inject_css(): #Create CSS function
    st.markdown("""
        <style>
        .block-container { /*Page block width*/
        max-width: 950px;  
        }
        h1 { /*Title font size*/
        font-size: 40px;
        }
        h2 { /*Header font size*/
            font-size: 30px;
        }
        h5 { /*Graph title font size*/
            font-size: 18px;
        }
        p {  /*Paragraph font size */
            font-size: 16px;
        }
        .st-co { /*Sidebar elements font size*/
            font-size: 12px !important;
        }
        svg.st-dv.st-dw.st-eh.st-ei.st-ej.st-ek.st-el { /*Hide x in*/
            display: none !important;
        }
        .st-emotion-cache-rkczhd p { /*Button font size*/
            font-size: 12px;
        }
        .st-emotion-cache-p2sra5 p { /*Button font size*/
            font-size: 12px;
        }
        .st-ez { /*Radio margin*/
            margin-top: 3.2px;
        }
        [data-testid="stSidebar"] {/*Sidebar width*/
        width: 280px; 
        min-width : 280px;
        max-width : 280px;
        }
        </style>
    """, unsafe_allow_html=True)
inject_css() #Execute CSS function




#################################################################### Data load
df = pd.read_csv('df_streamlit.csv') #Load 

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') #Set date data as date datatype
df['month'] = pd.to_datetime(df['month'], format='%Y-%m') #Set month data as date datatype
df['labels_list'] = df['labels'].apply(ast.literal_eval) #Convert labels datatype as python list
df['labels_list'] = df['labels_list'].apply(sorted) #Sort each list of labels
df['labels_string'] = df['labels_list'].apply(lambda x: ', '.join(x)) #Create a column with labels as list of strings

quartiers_gdf = gpd.read_file('quartier_paris.geojson') #Load quartier location data
quartiers_gdf = quartiers_gdf[['l_qu','geometry']].rename(columns = {'l_qu':'quartier'}) #Select using columns and rename

arrondissement_gdf = gpd.read_file('neighbourhoods.geojson') #Load arrondissement location data
arrondissement_gdf = arrondissement_gdf[['neighbourhood','geometry']] #Select using columns and rename





#################################################################### PAGE TITLE & SIDEBAR FILTERS
st.title("Airbnb Guest Reviews in Paris") #Page title

st.sidebar.markdown('#### Select Time Period') #Time Filter title
min_date = df['month'].min().to_pydatetime() #Set minimum date
max_date = df['month'].max().to_pydatetime() #Set maximum date
selected_min, selected_max = st.sidebar.slider( #Create time filter slider
    "Select Time Period", #Slider name
    value=( df['date'].min().to_pydatetime(), df['date'].max().to_pydatetime()), #Slider values
    min_value=min_date, #Minimum value
    max_value=max_date, #Maximum value
    format="YYYY-MM-DD", #Format
    label_visibility="collapsed" #Hide slider name
)
st.sidebar.write("") #Blank line
st.sidebar.markdown('#### Select Arrondissement') #Arrondissement Filter title
arrondissement_all = ["1 - Louvre","2 - Bourse","3 - Temple","4 - Hôtel-de-Ville","5 - Panthéon", #exhaustive list of arrondissement
                      "6 - Luxembourg","7 - Palais-Bourbon","8 - Élysée","9 - Opéra", "10 - Entrepôt",
                      "11 - Popincourt","12 - Reuilly","13 - Gobelins","14 - Observatoire","15 - Vaugirard",
                      "16 - Passy","17 - Batignolles-Monceau","18 - Buttes-Montmartre","19 - Buttes-Chaumont","20 - Ménilmontant"]

if 'selected' not in st.session_state: #Create session_state for selected arrondissement
    st.session_state.selected = ["1 - Louvre", "2 - Bourse", "3 - Temple", "4 - Hôtel-de-Ville", "5 - Panthéon", "6 - Luxembourg", "7 - Palais-Bourbon"] #Default arrondissement
def change_option(): #Create session_state for multiselect
    st.session_state.selected = st.session_state.options
col1, col2 = st.sidebar.columns([0.55, 1], gap = 'small') #Create columns for buttons
with col1: #On first column
    if st.button('Select All'): #Create button for all selection
        st.session_state.selected = arrondissement_all #If the button is clicked, all arrondissement is selected
with col2: #On second column
    if st.button('Deselect All'): #Create button to unselect all
        st.session_state.selected = [] #If button is clicked, none arrondissement is selected
with st.sidebar.form("Options"): #In a form
    arrondissement = st.multiselect("Select Arrondissement", #Create arrondissement mutiselect filter
                                    arrondissement_all, #Available options
                                    default=st.session_state.selected, #Default selection
                                    label_visibility="collapsed", #Hide title
                                    key='options', #session_state.options
                                    max_selections = 20 #maximum options that can be selected
                                  ) 
    submitted = st.form_submit_button("Apply Selection", on_click=change_option) #Create a button to apply selection
st.session_state.selected = arrondissement #update selected arrondissement

mask = (df['date'] >= selected_min) & (df['date'] <= selected_max) & (df['arrondissement'].isin(arrondissement)) #Create a mask with the filter selection
df_filtered = df[mask] #Select filtered data








#################################################################### TAB PAGES
SentimentTab, ClassificationTab, ReadmeTab = st.tabs(["Sentiment Analysis", "Multi-Label Classification", "Readme"]) #Create tabs


with SentimentTab: ################################################# Sentiment Analysis
    st.write('''
    This page displays sentiment analysis results for guest reviews of Paris zones across different time periods. Based on the content of each comment, a sentiment score from 1 to 5 is assigned, where 1 represents highly negative sentiment and 5 indicates positive sentiment. You can customize your view by selecting time period of interest and specific area in Paris.  
             ''')
    st.write("")

    if not arrondissement or selected_min == selected_max : #Warning message when no data is selected
        st.success("Please select at least one arrondissement and one period.")
    else : 

        col1, col2 = st.columns([3, 1.5], gap = 'small') #Create two columns for two graphs
        
        with col1: #On the first column
            st.markdown("<h5 style='text-align: center; font-weight: normal;'>Average Sentiment Score by Area</h5>", unsafe_allow_html=True) #Choropleth title
            col1_bis, col2_bis= st.columns([1,8]) #Create a sub-columns for sub-filteres
            with col1_bis: #On the first sub-column write 'show by'
                st.markdown("<p style='font-size:12px; line-height:3.5;'>Show by :</p>", unsafe_allow_html=True) 
            with col2_bis: #On the second sub-column create radio with options 'quartier' and 'arrondissement'
                zone = st.radio("Show by", options=["Quartier","Arrondissement"], horizontal=True, label_visibility="collapsed") #Create area division option filter
            if zone == "Quartier": #If quartier is selected
                df_filtered_zone = df_filtered[['quartier','sentiment']].groupby('quartier').mean().reset_index() #Create dataset grouped by quartier
                gdf_zone = gpd.GeoDataFrame(pd.merge(df_filtered_zone, quartiers_gdf, on='quartier'), geometry='geometry') #Add quartier geolocalisation in the dataset
                hover_data = ['quartier', 'sentiment'] # Define hover data for quartier
            elif zone == "Arrondissement": #If arrondissement is selected
                df_filtered_zone = df_filtered[['arrondissement','neighbourhood','sentiment']].groupby(['arrondissement','neighbourhood']).mean().reset_index() #Create dataset grouped by arrondissement
                gdf_zone = gpd.GeoDataFrame(pd.merge(df_filtered_zone, arrondissement_gdf, on='neighbourhood'), geometry='geometry') #Add arrondissement geolocalisation in the dataset
                hover_data = ['arrondissement', 'sentiment'] # Define hover data for arrondissement
            choropleth = px.choropleth_mapbox(gdf_zone, #Create the choropleth
                                    geojson= gdf_zone.geometry,
                                    locations=gdf_zone.index, 
                                    color='sentiment',
                                    color_continuous_scale=['#FF5A5F','#00A699'],
                                    range_color=[0, 5],
                                    mapbox_style="carto-positron",
                                    zoom=10.4, 
                                    center={"lat": 48.86, "lon": 2.347},
                                    hover_data=hover_data)
            choropleth.update_traces(marker_line_width=0, #Update marker configuration
                                     marker_opacity=0.8)
            choropleth.update_layout(coloraxis_colorbar={'lenmode': 'pixels','len': 370,'yanchor':'bottom','y': 0}, #Update colorbar configuration
                                     margin=dict(l=0, r=0, t=0, b=0), 
                                     height=350,
                                    ) 
            st.plotly_chart(choropleth, use_container_width=True) #Show the choropleth
        
        with col2: #On the second column
            st.markdown("<h5 style='text-align: center; font-weight: normal;'>Common Words in Reviews</h5>", unsafe_allow_html=True) #Wordcloud title
            comments_pos = df_filtered.loc[df_filtered['sentiment_binary'] == 'positive', 'comments_en'].astype(str).str.replace('\n', ' ').dropna() #Extract positive comments
            comments_pos = ' '.join(comments_pos) #Extract words
            comments_neg = df_filtered.loc[df_filtered['sentiment_binary'] == 'negative', 'comments_en'].astype(str).str.replace('\n', ' ').dropna() #Extract negative comments
            comments_neg = ' '.join(comments_neg) #Extract words
            mask_pos = np.array(Image.open('oval_up.png')) #Import wordcloud mask shape for positive comments
            mask_neg = np.array(Image.open('oval_down.png')) #Import wordcloud mask shape for negative comments
            wordcloud_pos = WordCloud(colormap=LinearSegmentedColormap.from_list("custom", ['#00A699', '#767676','#79CCCD']), mask = mask_pos, background_color=None, mode="RGBA", max_words=100).generate(comments_pos) #Create wordcloud for positive comments
            wordcloud_neg = WordCloud(colormap=LinearSegmentedColormap.from_list("custom", ['#FF5A5F', '#FBD2C5','#FFF6E6']), mask = mask_neg, background_color=None, mode="RGBA",  max_words=100).generate(comments_neg) #Create wordcloud for negative comments
            st.write("") #Blank line
            wordcloud = plt.figure(facecolor = 'none', figsize=(5, 3.8)) #Set graph background transparent
            ax1 = wordcloud.add_axes([0, 0, 1, 1]) #Configure wordcloud position for positive comments
            ax2 = wordcloud.add_axes([0, -0.81, 1, 1]) #Configure wordcloud position for negative comments
            ax1.imshow(wordcloud_pos) #Show wordcloud of positive comments
            ax1.set_title('Positive', color = '#00A699', size=20) #Title
            ax1.axis('off') #Remove axis
            ax2.imshow(wordcloud_neg) #Show wordcloud of negative comments
            ax2.set_title('Negative', color='#FF5A5F', size=20, y=-0.1) #Title
            ax2.axis('off') #Remove axis
            st.pyplot(wordcloud) #Show the wordcloud
        
        # Linechart
        st.write("")
        st.markdown("<h5 style='text-align: center; font-weight: normal;'>Daily Moving Average of Sentiment Score in Paris</h5>", unsafe_allow_html=True) #Linechart title
        df_sentiment = df_filtered.set_index('date').resample('D')['sentiment'].mean().reset_index() #Create daily data
        df_sentiment['moving_avg'] = df_sentiment['sentiment'].rolling(window=7).mean() #Moving average over 7 days
        linechart = px.line(df_sentiment, x='date', y='moving_avg') #Create linechart
        linechart.update_xaxes(dtick='D1', #Update ticks configuration
                               tickangle=-90, 
                               tickfont=dict(size=3))
        linechart.update_layout(yaxis_title="Score", #Update axis & ticks configuration
                                yaxis_title_font=dict(size=15), 
                                xaxis_title=None, 
                                yaxis=dict(range=[0, 5.2]),
                                yaxis_tickfont=dict(size=13), 
                                xaxis_tickfont=dict(size=11) 
                               ) 
        linechart.update_layout(margin=dict(l=60, r=60, t=5, b=150)) #Update margin
        linechart.update_traces(line_color="#00A699") #update linecolor
        st.plotly_chart(linechart) #Show the chart



with ClassificationTab: ############################################ Multi-classification
    
    labels_counts = df_filtered['labels_string'].value_counts().reset_index() #Create dataset with count of each combination of topics
    labels_counts['percent'] = (labels_counts['count'] / labels_counts['count'].sum() * 100).round(2).astype(str) + '%' #Calculate the percentage of each combination
    labels_counts['label_percent'] = '<b>'+labels_counts['labels_string']+'<br>'+labels_counts['percent'] #Create a column with the name and the percentage of each combination
    labels_counts['label_type'] = labels_counts['labels_string'].apply(lambda x: 'multi-label' if len(x.split(',')) > 1 else 'single-label') #Create a column specifying 'multi-label' or 'single-label'
    
    df_exploded = df_filtered.explode('labels_list') #Create dataset with single topic 
    df_exploded['labels_list'] = df_exploded['labels_list'].str.strip() #Remove space before and after each topic extracted
    labels_counts_exploded = df_exploded['labels_list'].value_counts().reset_index() #Count the single topic

    st.write('''
    This page presents the results of topic classification for guest reviews across different time periods in Paris. The classification model categorizes comments into five main topics: Apartment, Bed, Communication, Location, and Neighborhood. For each review, one or multiple relevant topics are assigned. You can customize your view by selecting time period of interest and specific area in Paris.
     ''')
    st.write("")

    if not arrondissement or selected_min == selected_max : #Warning message when no data is selected
        st.success("Please select at least one arrondissement and one period.")
    else : 
    
        col1, col2 = st.columns([3, 1.8], gap="medium") #Create two columns for two graphs
        
        with col1: #On the first column
            fig = px.treemap(labels_counts, #Create a treemap 
                             path=[px.Constant('all'),'label_type','label_percent'], 
                             values='count', 
                             #title='Multi-Topics of guest reviews', 
                             color='count',
                             color_continuous_scale=['#FBD2C5','#00A699'])
            fig.update_traces(textposition='middle center', #Update text configuration in the treemap
                              insidetextfont=dict(size=15),
                              marker=dict(cornerradius=3))
            fig.update_layout(width=1000, #Update graph layout
                              height=380, 
                              coloraxis_showscale=False, 
                              margin=dict(t=0, l=0, r=0, b=0),
                              paper_bgcolor= 'slategrey'
                              )
            st.markdown("<h5 style='text-align: center; font-weight: normal;'>Key Themes in Guest Reviews</h5>", unsafe_allow_html=True) #Treemap title
            st.plotly_chart(fig, use_container_width=True) #Show the graph
        
        with col2: #On the second column
            fig = px.histogram(labels_counts_exploded, #Create barplot
                               y='labels_list', 
                               x= 'count', 
                               color_discrete_sequence=['#FF5A5F'])
            fig.update_layout(width = 500, #Update graph layout
                              height = 380,
                              yaxis_title=None,
                              xaxis_title=None,
                              yaxis_tickfont=dict(size=15),
                              xaxis_tickfont=dict(size=15),
                              margin=dict(t=0, l=0, r=0, b=0)
                              )
            fig.update_yaxes(tickfont=dict(color='white'),categoryorder='category descending') #Update y-axes configuration
            fig.update_xaxes(tickfont=dict(color='white')) #Update x-axes configuration
            st.markdown("<h5 style='text-align: center; font-weight: normal;'>Count of single-label</h5>", unsafe_allow_html=True) #Barchart title
            st.plotly_chart(fig) #Show the plot

        st.markdown("")#Blank line
        st.markdown("<h5 style='text-align: center; font-weight: normal;'>Random Guest Reviews</h5>", unsafe_allow_html=True) #Title
        col1, col2 = st.columns([6,0.8]) #Create columns for filters
        with col1: #On the first column
            topic = st.multiselect("Select Labels", ['apartment', 'bed', 'communication', 'location', 'neighborhood'], default=['apartment', 'bed', 'communication', 'location', 'neighborhood']) #Create topic selector
        with col2: #On the last column
            st.write('') #Blank line
            if st.button('Refresh comments'): #Create refresh button
                filtered_df = df_filtered.loc[df_filtered['labels_list'].apply(lambda x: x == list(sorted(topic))), 'comments_en'] #Select filtered comments
                if len(filtered_df) >= 5: #If number of filtered comments are more than 5
                    comments = filtered_df.sample(5) #Show five random comments
                else: #If number of filtered comments are less than 5
                    comments = filtered_df #Show all comments
        
        filtered_df = df_filtered.loc[df_filtered['labels_list'].apply(lambda x: x == list(sorted(topic))), 'comments_en'] #Select filtered comments
        if len(filtered_df) >= 5: #If number of filtered comments are more than 5
            comments = filtered_df.sample(5) #Create 5 random comments which correspond to applied filters
        else: #If number of filtered comments are less than 5
            comments = filtered_df #Show all comments
        
        with st.container(border=True): #Create a container for the comments
            for index in range(len(comments)): #Show each comments generated in plain text
                st.write(index+1, comments.iloc[index])



with ReadmeTab: #################################################### Readme

    df['listing_id'] = df['listing_id'].astype(str) #Set listing id as string
    df.drop_duplicates('listing_id',inplace = True)
    
    st.write("""
    The purpose of this project is to analyze Airbnb guest experiences in Paris across various time periods, leveraging advanced Natural Language Processing (NLP) techniques on guest reviews. By combining various NLP tasks, this study aims to provide meaningful insights and a holistic view of guest experiences.
    
    Furthermore, this project showcases the power of modern NLP techniques in understanding customer experiences in the hospitality sector. It demonstrates how pre-trained models, which have been developed on large datasets, can be effectively utilized to perform complex language tasks. These models are particularly valuable when working with limited or unlabeled data, offering significant advantages in terms of time efficiency and computational resource management.
    
    The project exemplifies how sophisticated NLP applications can be developed using readily available open-source libraries, making advanced text analysis accessible without requiring exceptional Python skills.
    """)

    st.subheader("Dataset") #Header
    st.write('''
             
    Four datasets are used in the project:
             
    From [Inside Airbnb](https://insideairbnb.com/get-the-data/), an investigative website that reports and visualizes scraped data on Airbnb:
    - *listings.csv* : detailed information about 84,397 Airbnb properties in Paris.
    - *reviews.csv* : 1,794,006 comments left by guests for Airbnb properties in Paris.  
    - *neighbourhoods.geojson* : geolocation data of 20 arrondissements in Paris.  
    
    From [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/quartiers-administratifs/), the official open data platform of the French government:
    - *quartier_paris.geojson* : geolocation data of 80 quartiers in Paris.
    
    ''')
    
    st.write("") #Blank line
    st.markdown('<span style="font-size: 20px;">**Source Data**</span>', unsafe_allow_html=True) #Subheader
    st.write('''
    Below is an extract of columns from each dataset that are used in this analysis:         
    ''')
    st.write("*listings.csv*") #Show the dataset name
    st.dataframe(data = df[['listing_id','latitude','longitude']].head().rename(columns ={"listing_id":"id"}), hide_index=True) #Show the dataset
    st.write("*reviews.csv*") #Show the dataset name
    st.dataframe(data = df[['listing_id','date','comments']].head(), hide_index=True) #Show the dataset
    st.write("*neighbourhoods.geojson*") #Show the dataset name
    st.dataframe(data = df[['arrondissement','geometry_arrondissement']].head().rename(columns ={"geometry_arrondissement":"geometry"}), hide_index=True) #Show the dataset
    st.write("*quartier_paris.geojson*") #Show the dataset name
    st.dataframe(data = df[['quartier','geometry_quartier']].head().rename(columns ={"geometry_quartier":"geometry"}), hide_index=True) #Show the dataset
    
    st.write("") #Blank line
    st.markdown('<span style="font-size: 20px;">**Final Data**</span>', unsafe_allow_html=True) #Subheader
    st.write("After comprehensive data processing, a consolidated dataset is constructed as shown below, serving as the foundation for the NLP analysis.") #Explain the base dataset
    st.dataframe(data = df[['listing_id','date','comments','arrondissement','geometry_arrondissement','quartier','geometry_quartier']].head(), hide_index = True, width = 1400) #Show the dataset
    
    
    st.write("") #Blank line
    st.subheader("Techniques") #Header
    st.write('''
    The NLP techniques utilized comprise a comprehensive analysis pipeline that includes:
    - Language detection
    - Translation of non-English reviews to English
    - Sentiment analysis using state-of-the-art transformer models
    - Multi-label classification to categorize review content
    ''')
    
    st.write("") #Blank line
    st.markdown('<span style="font-size: 20px;">**Language Detection**</span>', unsafe_allow_html=True) #Subheader
    st.write('''
    Airbnb in Paris attracts a diverse array of international travelers, resulting in reviews written in numerous languages. Identifying the languages of reviews is crucial as it enables appropriate text preprocessing for accurate analysis across all languages and provides valuable insights into the diversity of Airbnb guests in Paris through the distribution of languages used.
    
    For language detection, *langdetect* package is employed, which is based on Google's language-detection library. This tool is particularly well-suited for identifying the language of given texts with high accuracy. Its ease of implementation and seamless integration with other NLP tools made it an ideal choice for this project. *langdetect*'s capability to identify over 55 languages, coupled with its speed and reliability, makes it exceptionally suitable for processing large volumes of review data.
    
             ''')
    container = st.container(border=True) #Create container for code
    container.write('*code example*') #Container title
    code = '''
    from langdetect import detect
    
    detect(comments) #Detect language
    '''
    container.code(code, language='python') #Show code bloc in the container
    container.write(f"{df['comments'][4]} : {df['language'][4]}") #Show the output
    
    st.write("") #Blank line
    st.markdown('<span style="font-size: 20px;">**Translation**</span>', unsafe_allow_html=True) #Subheader
    st.write('''
    To translate the reviews, *GoogleTranslator* from the *deep_translator* package is used. This tool harnesses the power of the Google Translate service, renowned for its accuracy and efficiency in language translation. By employing *GoogleTranslator* to convert all non-English reviews to English, we establish a uniform language base, enabling consistent application of sentiment scoring and classification techniques across the entire dataset.
             ''')
    container = st.container(border=True) #Create container for code
    container.write("*code example*") #Container title
    code = '''
    from deep_translator import GoogleTranslator
    
    GoogleTranslator(source='fr', target='en').translate(comments) #Translate
    '''
    container.code(code, language='python') #Show code bloc in the container
    container.write(f"{df['comments'][4]} : {df['comments_en'][4]}") #Show the output
    
    st.write("") #Blank line
    st.markdown('<span style="font-size: 20px;">**Sentiment Score**</span>', unsafe_allow_html=True) #Subheader
    st.write('''
    For sentiment analysis, *nlptown/bert-base-multilingual-uncased-sentiment* model is implemented through the *transformer* packages from [Hugging Face](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment). This model is a fine-tuned version of the bert-base-multilingual-uncased model, specifically optimized for sentiment analysis on product reviews across six languages: English, Dutch, German, French, Spanish, and Italian. It predicts sentiment on a scale of 1 to 5 stars, providing nuanced insights beyond simple positive/negative classifications. Although applied to English translations in this project, the model can directly analyze text in any of the six supported languages. 
             ''')
    container = st.container(border=True) #Create container for code
    container.write("*code example*") #Container title
    code ='''
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') #Initialize tokenizer with pretrained model
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') #Initialize sentiment calculator with pretrained model
    
    tokens = tokenizer.encode(comments, max_length=512, return_tensors='pt') #Apply tokenizer
    
    model(tokens) #Apply sentiment calculator on tokens
    '''
    container.code(code, language='python') #Show code bloc in the container
    container.write(f"{df['comments_en'][4]} : {df['sentiment'][4]}") #Show the output
    
    st.write("") #Blank line
    st.markdown('<span style="font-size: 20px;">**Multi-Label Classification**</span>', unsafe_allow_html=True) #Subheader
    st.write('''
    For multi-label classification of review, the zero-shot classification model *facebook/bart-large-mnli* is implemented through the [Hugging Face](https://huggingface.co/facebook/bart-large-mnli) pipeline. This approach employs the BART model fine-tuned on a large-scale dataset designed to train and evaluate models on understanding relationships between sentences and making inferences across a wide range of text genres. The model is applied to classify comments into five topics: 'apartment', 'bed', 'communication', 'location, and 'neighborhood'. With the multi-label option, the model returns a score for each topic. For the final classification, we retained topics with scores higher than 0.9, or the highest-scoring topic when no score exceeded 0.9. 
             ''')
    container = st.container(border=True) #Create container for code
    container.write("*code example*") #Container title
    code = '''
    from transformers import pipeline
    
    model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") #Initialize zero-shot model
    
    model(comments, candidate_labels=['communication', 'apartment', 'neighborhood', 'location', 'bed'], multi_label = True) #Apply multi-label zero-shot model
    '''
    container.code(code, language='python') #Show code bloc in the container
    container.write(f"{df['label_classification'][4]}") #Show the output
    
    st.write("") #Blank line
    st.subheader("Output") #Header
    st.write('''
    The analysis results were transformed into an interactive Streamlit application, featuring two main pages.
    
    Sentiment Analysis Page :
    - A map of Paris displaying sentiment scores across arrondissements and quartiers
    - Positive and negative word clouds visualizing frequently used terms
    - A chart showing the evolution of sentiment scores over time
    
    Multi-label Classification Page :
    - Distribution chart of the final topic classifications
    - Frequency chart showing how often each topic appears in reviews
    - A section displaying 5 random reviews corresponding to user-selected topics
    
    Both pages include filters for time period and arrondissement selection, allowing for dynamic exploration of the data. Users can interact with the results and filter information based on specific time periods or areas of Paris. Due to the extensive size of the dataset, the Streamlit transformation is currently limited to the results from Q1 2024. 
             ''')
    
    st.write("") #Blank line
    st.subheader("Code Source") #Header
    "[Full code](https://github.com/haejiyun/01.nlp/blob/main/airbnb-reviews/airbnb_nlp.py)" #Github code link
    "[Author's page](https://haejiyun.github.io/)" #Github page link
