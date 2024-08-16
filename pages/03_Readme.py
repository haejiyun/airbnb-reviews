import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st





#DATASET#####

#################################################################### Data load
df = pd.read_csv('df_streamlit.csv') #Load dataset
df['listing_id'] = df['listing_id'].astype(str) #Set listing id as string
df.drop_duplicates('listing_id',inplace = True)




#STREAMLIT#####

#################################################################### Configuration
st.set_page_config(layout="wide") #Set wide page layout


# Custom CSS
st.markdown("""
    <style>
    h1 {  /* Title */
        font-size: 40px;
    }
    h2 {  /* Header */
        font-size: 30px;
    }
    p {
        font-size: 15px;  /* Adjust paragraph font size */
    }
    </style>
    """, unsafe_allow_html=True)





#################################################################### Layout
st.title("Airbnb Guest Reviews in Paris") #Page title
st.header("Readme") #Page subtitle
st.markdown("***") #Breakline
st.write("""
The purpose of this project is to analyze Airbnb guest experiences in Paris across various time periods, leveraging advanced Natural Language Processing (NLP) techniques on guest reviews. By combining various NLP tasks, this study aims to provide meaningful insights and a holistic view of guest experiences.

Furthermore, this project showcases the power of modern NLP techniques in understanding customer experiences in the hospitality sector. It demonstrates how pre-trained models, which have been developed on large datasets, can be effectively utilized to perform complex language tasks. These models are particularly valuable when working with limited or unlabeled data, offering significant advantages in terms of time efficiency and computational resource management.

The project exemplifies how sophisticated NLP applications can be developed using readily available open-source libraries, making advanced text analysis accessible without requiring exceptional Python skills.
""")

#%%############################################################## STREAMLIT

st.subheader("Dataset") #Header
st.write('''
         
Four datasets are used in the project:
         
From [Inside Airbnb](https://insideairbnb.com/get-the-data/), an investigative website that reports and visualizes scraped data on Airbnb:
- *listings.csv* : detailed information about 84,397 Airbnb properties in Paris.  
- *reviews.csv* : 1,794,006 comments left by guests for Airbnb properties in Paris.  
- *neighbourhoods.geojson* : geolocation data of 20 arrondissements in Paris.  

From [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/quartiers-administratifs/), the official open data platform of the French government:
- *quartier_paris.geojson* : geolocation data of 80 quartiers in Paris.

Below is an extract of columns from each dataset that are used in this analysis:         
''')

#col1, col2, col3, col4 = st.columns([1.3,3.5,1.7,1.5]) #Divide four columns with different width to show four dataset on the same line
#with col1: #On the first column
st.write("*listings.csv*") #Show the dataset name
st.dataframe(data = df[['listing_id','latitude','longitude']].head().rename(columns ={"listing_id":"id"}), hide_index=True) #Show the dataset
#with col2: #On the second column
st.write("*reviews.csv*") #Show the dataset name
st.table(data = df[['listing_id','date','comments']].head()) #Show the dataset
#with col3: #On the third column
st.write("*neighbourhoods.geojson*") #Show the dataset name
st.dataframe(data = df[['arrondissement','geometry_arrondissement']].sample(10).rename(columns ={"geometry_arrondissement":"geometry"}), hide_index=True) #Show the dataset
#with col4: #On the fourth column
st.write("*quartier_paris.geojson*") #Show the dataset name
st.dataframe(data = df[['quartier','geometry_quartier']].sample(10).rename(columns ={"geometry_quartier":"geometry"}), hide_index=True) #Show the dataset

st.write("After comprehensive data processing, a consolidated dataset is constructed as shown below, serving as the foundation for the NLP analysis.") #Explain the base dataset
st.dataframe(data = df[['listing_id','date','comments','arrondissement','geometry_arrondissement','quartier','geometry_quartier']].sample(5), hide_index = True, width = 1400) #Show the dataset


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

For language detection, I employed the *langdetect* package, which is based on Google's language-detection library. This tool is particularly well-suited for identifying the language of given texts with high accuracy. Its ease of implementation and seamless integration with other NLP tools made it an ideal choice for this project. *langdetect*'s capability to identify over 55 languages, coupled with its speed and reliability, makes it exceptionally suitable for processing large volumes of review data.

         ''')
container = st.container(border=True) #Create container for code
container.write('*code example*') #Container title
code = '''
from langdetect import detect

detect(comments) #Detect language
'''
container.code(code, language='python') #Show code bloc in the container
container.write(f"{df['comments'][0]} : {df['language'][0]}") #Show the output

st.write("") #Blank line
st.markdown('<span style="font-size: 20px;">**Translation**</span>', unsafe_allow_html=True) #Subheader
st.write('''
To translate the reviews, I used the *GoogleTranslator* from the *deep_translator* package. This tool harnesses the power of the Google Translate service, renowned for its accuracy and efficiency in language translation. By employing *GoogleTranslator* to convert all non-English reviews to English, we establish a uniform language base, enabling consistent application of sentiment scoring and classification techniques across the entire dataset.
         ''')
container = st.container(border=True) #Create container for code
container.write("*code example*") #Container title
code = '''
from deep_translator import GoogleTranslator

GoogleTranslator(source='fr', target='en').translate(comments) #Translate
'''
container.code(code, language='python') #Show code bloc in the container
container.write(f"{df['comments'][0]} : {df['comments_en'][0]}") #Show the output

st.write("") #Blank line
st.markdown('<span style="font-size: 20px;">**Sentiment Score**</span>', unsafe_allow_html=True) #Subheader
st.write('''
For sentiment analysis, I employed the *nlptown/bert-base-multilingual-uncased-sentiment* model, implemented through the *transformer* packages from [Hugging Face](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment). This model is a fine-tuned version of the bert-base-multilingual-uncased model, specifically optimized for sentiment analysis on product reviews across six languages: English, Dutch, German, French, Spanish, and Italian. It predicts sentiment on a scale of 1 to 5 stars, providing nuanced insights beyond simple positive/negative classifications. Although applied to English translations in this project, the model can directly analyze text in any of the six supported languages. 
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
container.write(f"{df['comments_en'][0]} : {df['sentiment'][0]}") #Show the output

st.write("") #Blank line
st.markdown('<span style="font-size: 20px;">**Multi-Label Classification**</span>', unsafe_allow_html=True) #Subheader
st.write('''
For multi-label classification of review, I used the zero-shot classification model *facebook/bart-large-mnli* implemented through the [Hugging Face](https://huggingface.co/facebook/bart-large-mnli) pipeline. This approach employs the BART model fine-tuned on a large-scale dataset designed to train and evaluate models on understanding relationships between sentences and making inferences across a wide range of text genres. I applied the model to classify comments into five topics: 'apartment', 'bed', 'communication', 'location, and 'neighborhood'. With the multi-label option, the model returns a score for each topic. For the final classification, I retained topics with scores higher than 0.9, or the highest-scoring topic when no score exceeded 0.9. 
         ''')
container = st.container(border=True) #Create container for code
container.write("*code example*") #Container title
code = '''
from transformers import pipeline

model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") #Initialize zero-shot model

model(comments, candidate_labels=['communication', 'apartment', 'neighborhood', 'location', 'bed'], multi_label = True) #Apply multi-label zero-shot model
'''
container.code(code, language='python') #Show code bloc in the container
container.write(f"{df['label_classification'][0]}") #Show the output

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

Both pages include filters for time period and arrondissement selection, allowing for dynamic exploration of the data. Users can interact with the results and filter information based on specific time periods or areas of Paris. Due to the extensive size of the dataset, the Streamlit transformation is currently limited to the results from 2024. 
         ''')

st.write("") #Blank line
st.subheader("Code Source") #Header
"[Github code](https://github.com/haejiyun/airbnb-reviews/tree/main)" #Github code link
"[Author's Linkedin](https://www.linkedin.com/in/haejiyun/)" #Linkedin link
