import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
import string

# Define the clean function
def clean(text, 
          lowercase=True, 
          remove_brackets=True, 
          remove_urls=True, 
          remove_html=True, 
          remove_punctuation=True, 
          remove_numbers=True, 
          remove_stopwords=True, 
          stem_or_lem="stem", 
          language="english", 
          custom_stopwords=None, 
          tokenizer=str.split):
    nltk.download('stopwords', quiet=True)
    if stem_or_lem == "stem":
        stemmer = nltk.SnowballStemmer(language)
        process_word = stemmer.stem
    else:
        nltk.download('wordnet', quiet=True)
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        process_word = lemmatizer.lemmatize
    
    stopword_set = set(stopwords.words(language))
    if custom_stopwords:
        stopword_set.update(custom_stopwords)
    
    if lowercase:
        text = str(text).lower()
    if remove_brackets:
        text = re.sub(r'\[.*?\]', '', text)
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    if remove_html:
        text = re.sub(r'<.*?>+', '', text)
    if remove_punctuation:
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    if remove_numbers:
        text = re.sub(r'\w*\d\w*', '', text)
    
    words = tokenizer(text)
    if remove_stopwords:
        words = [word for word in words if word and word not in stopword_set]
    if stem_or_lem in ["stem", "lemmatize"]:
        words = [process_word(word) for word in words]
    
    return " ".join(words)

def clean_dataframe_column(df, column_name, **kwargs):
    if not isinstance(df, pd.DataFrame) or column_name not in df.columns:
        raise ValueError("Invalid DataFrame or column name")
    df[column_name] = df[column_name].apply(clean, **kwargs)
    return df

# Prompt user for inputs
print("Welcome! Let's configure the text cleaning process.")
print("Enter 'y' for yes, 'n' for no, or press Enter for default (yes).")

# Get user preferences
lowercase = input("Convert to lowercase? (y/n) [default: y] ") != "n"
remove_brackets = input("Remove content in brackets? (y/n) [default: y] ") != "n"
remove_urls = input("Remove URLs? (y/n) [default: y] ") != "n"
remove_html = input("Remove HTML tags? (y/n) [default: y] ") != "n"
remove_punctuation = input("Remove punctuation? (y/n) [default: y] ") != "n"
remove_numbers = input("Remove numbers? (y/n) [default: y] ") != "n"
remove_stopwords = input("Remove stopwords? (y/n) [default: y] ") != "n"

stem_or_lem = input("Use stemming or lemmatization? (stem/lem) [default: stem] ") or "stem"
language = input("Language (e.g., english, spanish)? [default: english] ") or "english"

custom_stopwords_input = input("Add custom stopwords (comma-separated, e.g., not,very)? [default: none] ")
custom_stopwords = custom_stopwords_input.split(",") if custom_stopwords_input else None

tokenizer_input = input("Use custom tokenizer? (y/n) [default: n] ") != "n"
tokenizer = str.split  # Default
if tokenizer_input == "y":
    tokenizer_option = input("Enter tokenizer (e.g., nltk.word_tokenize)? [default: str.split] ") or "str.split"
    if tokenizer_option == "nltk.word_tokenize":
        from nltk.tokenize import word_tokenize
        tokenizer = word_tokenize

# Example usage with user inputs
sample_text = "Running to www.example.com in 2023! Great product [Score: 9/10]"
print("\nCleaning sample text:", sample_text)
cleaned_text = clean(sample_text, 
                     lowercase=lowercase,
                     remove_brackets=remove_brackets,
                     remove_urls=remove_urls,
                     remove_html=remove_html,
                     remove_punctuation=remove_punctuation,
                     remove_numbers=remove_numbers,
                     remove_stopwords=remove_stopwords,
                     stem_or_lem=stem_or_lem,
                     language=language,
                     custom_stopwords=custom_stopwords,
                     tokenizer=tokenizer)
print("Cleaned text:", cleaned_text)

# Optional: Process a DataFrame if provided
use_df = input("\nProcess a DataFrame? (y/n) [default: n] ") != "n"
if use_df:
    df_data = input("Enter DataFrame data (e.g., 'review1,review2' for column 'review'): ") or "Great!,Bad service"
    df = pd.DataFrame({"review": df_data.split(",")})
    cleaned_df = clean_dataframe_column(df, "review", 
                                       lowercase=lowercase,
                                       remove_brackets=remove_brackets,
                                       remove_urls=remove_urls,
                                       remove_html=remove_html,
                                       remove_punctuation=remove_punctuation,
                                       remove_numbers=remove_numbers,
                                       remove_stopwords=remove_stopwords,
                                       stem_or_lem=stem_or_lem,
                                       language=language,
                                       custom_stopwords=custom_stopwords,
                                       tokenizer=tokenizer)
    print("Cleaned DataFrame:\n", cleaned_df)