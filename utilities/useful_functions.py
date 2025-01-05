import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import emoji
import re

from wordcloud import WordCloud
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException




def extract_website_name(url:str) -> str:
    """
    Extract the website name from a URL.
    
    Args:
        url (str): The input URL
        
    Returns:
        str: The extracted website name
    """

    # Remove protocol prefixes if they exist
    url = url.lower().replace('https://', '').replace('http://', '')
    
    # Split by '/' and take the first part (domain)
    domain = url.split('/')[0]
    
    # Split domain by '.' and take the part that represents the website name
    website_name = domain.split('.')[0]
    
    # Handle cases where the website name comes after 'www.'
    if website_name == 'www':
        website_name = domain.split('.')[1]
    
    return website_name



def create_boxplot(df, numerical_col, category_col):
    """
    Creates a box plot using seaborn.

    Args:
    df: pandas DataFrame containing the data.
    numerical_col: string, name of the numerical column.
    category_col: string, name of the categorical column.
    """

    sns.boxplot(x=category_col, y=numerical_col, data=df)
    plt.xlabel(category_col)
    plt.ylabel(numerical_col)
    plt.title(f"Box Plot of {numerical_col} by {category_col}")
    plt.show()




def create_wordcloud(df, text_column: str, stop_words: set, width = 1000, height=600, output_image_path=None):
    """
    Generate a word cloud from a text column in a TSV dataset.
    
    Args:
        text_column (str): Column name containing text data.
        output_image_path (str): Optional path to save the word cloud image.
        language (str): Language for stopwords (default is 'uk' for Ukrainian).
    """
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the dataset.")
    

    # Combine all text from the specified column
    text_data = ' '.join(df[text_column].dropna())
    
    
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        stopwords=stop_words,
        collocations=True
    ).generate(text_data)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    if output_image_path:
        wordcloud.to_file(output_image_path)
        print(f"Word cloud saved to {output_image_path}")




def detect_language_safe(text):
    """
    It detects a language of a string using Google's langdetect
    """

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"




def count_emojis(text):
    """
    Calculate the number of emojis in a given text.
    
    Args:
        text (str): Input text.
        
    Returns:
        int: Count of emojis in the text.
    """
    if not isinstance(text, str):
        return 0  # Return 0 for non-string inputs
    
    # Extract emojis from text
    emojis = [char for char in text if char in emoji.EMOJI_DATA]
    return len(emojis)



def count_special_characters(text):
    """
    Calculate the number of special characters in a given text.
    
    Args:
        text (str): Input text.
        
    Returns:
        int: Count of special characters in the text.
    """
    if not isinstance(text, str):
        return 0  # Return 0 for non-string inputs
    
    # Define special characters (excluding alphanumeric and whitespace)
    special_chars = re.findall(r"[^a-zA-Z0-9\s\u0400-\u04FF]", text)
    return len(special_chars)



