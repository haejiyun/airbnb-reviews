The purpose of this project is to analyze Airbnb guest experiences in Paris across various time periods, leveraging advanced Natural Language Processing (NLP) techniques on guest reviews. By combining various NLP tasks, this study aims to provide meaningful insights and a holistic view of guest experiences.

Furthermore, this project showcases the power of modern NLP techniques in understanding customer experiences in the hospitality sector. It demonstrates how pre-trained models, which have been developed on large datasets, can be effectively utilized to perform complex language tasks. These models are particularly valuable when working with limited or unlabeled data, offering significant advantages in terms of time efficiency and computational resource management.
<br/>

**Dataset**<br/>

Four datasets are used in the project:
         
From [Inside Airbnb](https://insideairbnb.com/get-the-data/), an investigative website that reports and visualizes scraped data on Airbnb:
- *listings.csv* : detailed information about 84,397 Airbnb properties in Paris.  
- *reviews.csv* : 1,794,006 comments left by guests for Airbnb properties in Paris.  
- *neighbourhoods.geojson* : geolocation data of 20 arrondissements in Paris.  

From [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/quartiers-administratifs/), the official open data platform of the French government:
- *quartier_paris.geojson* : geolocation data of 80 quartiers in Paris.  

**Techniques**  


The NLP techniques utilized comprise a comprehensive analysis pipeline that includes:
- Language detection
- Translation of non-English reviews to English
- Sentiment analysis using state-of-the-art transformer models
- Multi-label classification to categorize review content

**Language Detection**  

Airbnb in Paris attracts a diverse array of international travelers, resulting in reviews written in numerous languages. Identifying the languages of reviews is crucial as it enables appropriate text preprocessing for accurate analysis across all languages and provides valuable insights into the diversity of Airbnb guests in Paris through the distribution of languages used.

For language detection, I employed the *langdetect* package, which is based on Google's language-detection library. This tool is particularly well-suited for identifying the language of given texts with high accuracy. Its ease of implementation and seamless integration with other NLP tools made it an ideal choice for this project. *langdetect*'s capability to identify over 55 languages, coupled with its speed and reliability, makes it exceptionally suitable for processing large volumes of review data.
<br/>
<br/>
**Translation**<br/>

To translate the reviews, I used the *GoogleTranslator* from the *deep_translator* package. This tool harnesses the power of the Google Translate service, renowned for its accuracy and efficiency in language translation. By employing *GoogleTranslator* to convert all non-English reviews to English, we establish a uniform language base, enabling consistent application of sentiment scoring and classification techniques across the entire dataset.
<br/>
<br/>
**Sentiment Score**<br/>

For sentiment analysis, I employed the *nlptown/bert-base-multilingual-uncased-sentiment* model, implemented through the *transformer* packages from [Hugging Face](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment). This model is a fine-tuned version of the bert-base-multilingual-uncased model, specifically optimized for sentiment analysis on product reviews across six languages: English, Dutch, German, French, Spanish, and Italian. It predicts sentiment on a scale of 1 to 5 stars, providing nuanced insights beyond simple positive/negative classifications. Although applied to English translations in this project, the model can directly analyze text in any of the six supported languages. 
<br/>
<br/>
**Multi-Label Classification**<br/>

For multi-label classification of review, I used the zero-shot classification model *facebook/bart-large-mnli* implemented through the [Hugging Face](https://huggingface.co/facebook/bart-large-mnli) pipeline. This approach employs the BART model fine-tuned on a large-scale dataset designed to train and evaluate models on understanding relationships between sentences and making inferences across a wide range of text genres. I applied the model to classify comments into five topics: 'apartment', 'bed', 'communication', 'location, and 'neighborhood'. With the multi-label option, the model returns a score for each topic. For the final classification, I retained topics with scores higher than 0.9, or the highest-scoring topic when no score exceeded 0.9. 
<br/>
<br/>
**Output**<br/>

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
