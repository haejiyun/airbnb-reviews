import pandas as pd 
import numpy as np 
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import time
import os
from langdetect import detect, LangDetectException
from shapely.geometry import Point, Polygon
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from deep_translator import GoogleTranslator
from tqdm import tqdm





#%%############################################################## IMPORTING DATA
reviews = pd.read_csv("reviews.csv") #Airbnb reviews
listings = pd.read_csv("listings.csv") #Airbnb listings
neighborhoods_gdf = gpd.read_file('neighbourhoods.geojson') #Arrondissement geojson
quartiers_gdf = gpd.read_file('quartier_paris.geojson') #Quartier geojson





#%%############################################################## DATA CLEANING

#Quartiers dataset - select name and geometry info
quartiers_gdf = quartiers_gdf[['l_qu','geometry']].rename(columns = {'l_qu':'quartier'})

#Neighborhoods dataset 
neighborhoods_gdf = neighborhoods_gdf[['neighbourhood', 'geometry']] # Select name and geometry info
neighborhoods_gdf['arrondissement'] = neighborhoods_gdf['neighbourhood'].replace( # Replace name by arrondissement#
    {"Louvre": "1 - Louvre",
     "Bourse": "2 - Bourse",
     "Temple": "3 - Temple",
     "Hôtel-de-Ville": "4 - Hôtel-de-Ville",
     "Panthéon": "5 - Panthéon",
     "Luxembourg": "6 - Luxembourg",
     "Palais-Bourbon": "7 - Palais-Bourbon",
     "Élysée": "8 - Élysée",
     "Opéra": "9 - Opéra",
     "Entrepôt": "10 - Entrepôt",
     "Popincourt": "11 - Popincourt",
     "Reuilly": "12 - Reuilly",
     "Gobelins": "13 - Gobelins",
     "Observatoire": "14 - Observatoire",
     "Vaugirard": "15 - Vaugirard",
     "Passy": "16 - Passy",
     "Batignolles-Monceau": "17 - Batignolles-Monceau",
     "Buttes-Montmartre": "18 - Buttes-Montmartre",
     "Buttes-Chaumont": "19 - Buttes-Chaumont",
     "Ménilmontant":  "20 - Ménilmontant"
    }
    )

#Listings dataset
#Find corresponding arrondissement and quartier of each listing
listings = listings[['id','longitude','latitude']] # Select id and geometry info
geometry = [Point(xy) for xy in zip(listings['longitude'], listings['latitude'])] #Create geometry points
listings_gdf = gpd.GeoDataFrame(listings, geometry=geometry, crs="EPSG:4326") #Find geometry points for each listing
#Find listing's arrondissement geometry
listings_neighborhoods = gpd.sjoin(listings_gdf, neighborhoods_gdf, how='left', predicate='within') #Find each listings' arrondissement
listings_neighborhoods = listings_neighborhoods.merge(neighborhoods_gdf[['neighbourhood','geometry']], on='neighbourhood', how='left').rename(columns = {'geometry_y':'geometry_arrondissement'}) #Add arrondissement geometry, rename columns geometry -> geometry_arrondissement
listings_neighborhoods = listings_neighborhoods[['id', 'neighbourhood', 'arrondissement', 'geometry_arrondissement']] #Select id, arrondissement, geometry info
#Find listing's quartier geometry
listings_quartiers = gpd.sjoin(listings_gdf, quartiers_gdf, how='left', predicate='within') #Find each listings' quartier
listings_quartiers = listings_quartiers.merge(quartiers_gdf, on = 'quartier', how = 'left').rename(columns = {'geometry_y':'geometry_quartier'}) #Add quartier geometry, rename columns geometry -> geometry_quartier
listings_quartiers = listings_quartiers[['id', 'quartier', 'geometry_quartier']] #Select id, quartier, geometry info 
#Combine listings' arrondissement & quartier info
listings_neighborhoods_quartiers = listings_neighborhoods.merge(listings_quartiers, how = 'right', on = 'id') # Combine neighborhood and quartier

#Reviews dataset
reviews.dropna(subset='comments', inplace = True) #Remove rows without comments
reviews['date'] = pd.to_datetime(reviews['date'], format='%Y-%m-%d') #Change date datatype : object -> datetime
#reviews = reviews[reviews['date']>='2021-01-01'] #Select reviews from 2021
reviews['month'] = reviews['date'].dt.to_period('M').astype(str) #Create a column with YYYY-mm (without date)
reviews.drop_duplicates(subset=['listing_id','date','reviewer_name','comments'], inplace = True) #Drop duplicates
reviews = reviews[['listing_id','date','month','comments']] #Select listing_id, date, comments
#Add 'arrondissement' and 'quartier' information of each comments
df = reviews.merge(listings_neighborhoods_quartiers, how = 'left', left_on = 'listing_id', right_on = 'id').reset_index(drop=True) # Combine reviews and listing data
df.to_csv('df.csv', index = False) 





#%%############################################################## NLP - creation of functions

#Language Detection option 1 : without error details returned
def detect_language(text): #Create function to detect the language
    try: #Error handler
        if isinstance(text, str): #Check if the input is string
            return detect(text) #if yes, apply detect() function
        else: 
            return "unknown" #if not, return 'unknown'
    except LangDetectException: #for problem encountered
        return "unknown" #return 'unknown'

#Language Detection option 2 : with error details
def detect_language_error_detail(text):
    try:
        if isinstance(text, str):
            return detect(text)
        else:
            return "unknown"
    except LangDetectException as e: #for problem encountered
        return f"unknown, (Error: {str(e)})" #return 'unknown' and the problem details

#Translation option 1 : skip when text is longer than allowed number of characters
def translate(text, dest='en', retries=3, max_length=4500): #Create function to translate in english for text of maximum 4500 characters with 3 tries
    if pd.isna(text) or (isinstance(text, str) and text.strip() == ''): #if text is empty
        return text #return the empty text without translating
    if len(text) > max_length: #if text is longer than maximum length
        return "[TEXT TOO LONG - SKIPPED]" #skip translation and return 'text too long'
    for _ in range(retries): #number of retries
        try: #Error handler
            translated = GoogleTranslator(source='auto', target=dest).translate(text) #apply googletranslator to translate in english by detecting original language automatically
            return translated #return translated value
        except Exception as e: #if exception is raised
            print(f"\nTranslation error: {e}, retrying in 3 seconds...") #print error details
            time.sleep(3) #delay 3 seconds to retry
    return "[TRANSLATION FAILED]" #return 'failed' in case of translation failure after 3 tries

#Translation option 2: without skipping when text is too long
def translate_no_skip(text, dest='en', retries=3, max_length=4500):
    if pd.isna(text) or (isinstance(text, str) and text.strip() == ''):
        return text
    if len(text) > max_length: #if text is longer than maximum length
        text = text[:max_length] #truncated the text at the maximum length
        print(f"Text truncated to {max_length} characters.") #print the truncation
    for _ in range(retries):
        try:
            translated = GoogleTranslator(source='auto', target=dest).translate(text)
            return translated
        except Exception as e:
            print(f"\nTranslation error: {e}, retrying in 3 seconds...")
            time.sleep(3)
    return "[TRANSLATION FAILED]"

#Sentiment Score
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') #Create tokenizer
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') #Create sentiment classification model

def sentiment_score(text): #Create sentiment analysis function
    tokens = tokenizer.encode(text, max_length=512, truncation=True, return_tensors='pt') #Encode the text to integer input ids using tokenizer
    with torch.no_grad(): #Disable gradient calculation (not necessary in this case, but good practice to use when doing prediction with pytorch tensors)
        result = model(tokens) #Apply model to input ids
    return int(torch.argmax(result.logits))+1 #Select the index with highest value of prediction and add 1 in order to have the score between 1-5 instead of 0-4

#Saving file in progress
def save_progress(df, filename): #Create function to save progress file
    if not os.path.exists(filename): #if file does not exist
        df.to_csv(filename, mode='w', index=False) #create file and save
    else: #if file exists
        df.to_csv(filename, mode='a', header=False, index=False) #append to the existing file without header
    print(f"Progress saved to {filename}") #print the saving status

#Zero-shot-classification
model_zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") #Initialize zero-shot model

def zero_shot(text):#Create zero-shot function
    return model_zs(text, candidate_labels=labels, multi_label = True) #which returns the result of multi-label classification

def extract_label(output): #Define a function to extract predicted label
    labels = output['labels'] #Seperate labels
    scores = output['scores'] #and scores
    best_labels = [label for label, score in zip(labels, scores) if score >= 0.9] #Retrieve labels having a score higher or equal to 0.9
    return best_labels if best_labels else [labels[0]] #If no label is higher or equals to 0.9, return the label with the highest score





#%%############################################################## LANGUAGE DETECTION

# Detect language
df['language'] = df['comments'].apply(detect_language) #Apply language detection function to comments

# Show unknown language
df.loc[df['language'] == 'unknown','comments'].value_counts() #Show comments detected as unknown

# Remove rows with unknown language
df = df[~(df['language'] == 'unknown')] #Remove unknown language





#%%############################################################## TRANSLATION

#Create a column to store english translation
df['comments_en'] = None 

# Translation Option 1
save_interval = 1000 #Set saving progress interval
total_rows = len(df) #Assign total rows of the dataset
 
for start in range(0, total_rows, save_interval): #From the start of each saving interval
    end = min(start + save_interval, total_rows) #to the end of each saving interval
    #Translation
    tqdm.pandas(desc=f"Processing rows {start}to {end}") #Activate progress bar
    df.loc[start:end, 'comments_en'] = df.loc[start:end, 'comments'].progress_apply(translate) #Apply translate function
    #Saving
    save_progress(df.iloc[start:end], f'translation_progress.csv') #Save the progress

df.to_csv('df_translation.csv', index = False)#Save the final translation
skipped_count = (df['comments_en'] == "[TEXT TOO LONG - SKIPPED]").sum() #count number of translation skipped 
failed_count = (df['comments_en'] == "[TRANSLATION FAILED]").sum() #count number of translation failed
print(f"\nAll rows processed with {skipped_count} skipped and {failed_count} failed translation") #Show the completion

df['comments_en'] = df['comments_en'].str.replace('<br/>', '', regex=False) #Replace html syntax 'br' by space



#%%############################################################## SENTIMENT SCORE
save_interval = 1000 #Set saving progress interval
total_rows = len(df) #Assign total rows of the dataset

#Calculate sentiment score
for start in range(0, total_rows, save_interval): #From the start of each saving interval
    end = min(start + save_interval, total_rows) #to the end of each saving interval
    #Sentiment score
    tqdm.pandas(desc=f"Processing rows {start}-{end}") #Activate progress bar
    df.loc[start:end, 'sentiment'] = df.loc[start:end, 'comments_en'].progress_apply(sentiment_score) #Apply sentiment_score function
    #Saving
    save_progress(df.iloc[start:end], f'sentiment_progress.csv') #Save the progress

df['sentiment_binary'] = ['positive' if i>3 else 'negative' for i in df['sentiment']] #Assign negative/positive

df.to_csv('df_sentiment.csv', index = False) #Save the final file
print("\nAll rows processed!") #Show the completion





#%%############################################################## ZERO-SHOT CLASSIFICATION
#Create labels to attribute
labels = ['bed', 'apartment', 'neighborhood', 'location', 'communication'] 

#Create a column to store zero-shot classification
df['label_classification'] = None

save_interval = 1000 #Set saving progress interval
total_rows = len(df) #Assign total rows of the dataset

#Apply zero-shot classification
for start in range(0, total_rows, save_interval): #From the start of each saving interval
    end = min(start + save_interval, total_rows) #to the end of each saving interval
    # Zero-shot classification
    tqdm.pandas(desc=f"Processing rows {start}-{end}") #Activate progress bar
    df.loc[start:end, 'label_classification'] = df.loc[start:end, 'comments_en'].progress_apply(zero_shot) #Apply zero-shot function
    # Saving
    save_progress(df.iloc[start:end], f'zeroshot_progress.csv') #Save the progress

df['labels'] = df['label_classification'].apply(extract_label) #Select the best labels

df.to_csv('df_final.csv', index = False) #Save the final file
print("\nCompleted!") #Notify the completion





#%%############################################################## Streamlit Dataset
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df_streamlit = df[df['date']>='2024-01-01'] #set period to show in streamlit
listing_location = listings[['id', 'latitude','longitude']].rename(columns = {'id':'listing_id'}) #Extract listings' longitude and latitude
df_streamlit = df_streamlit.merge(listing_location, how = 'left', on = 'listing_id') #Add longitude, latitude
df_streamlit.to_csv('df_streamlit.csv', index = False) #save file for streamlit
