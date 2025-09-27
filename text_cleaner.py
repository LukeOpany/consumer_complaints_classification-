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
    import nltk
    import re
    from nltk.corpus import stopwords
    import string
    
    # Download stopwords if not present
    nltk.download('stopwords', quiet=True)
    
    # Initialize stemmer or lemmatizer based on choice
    if stem_or_lem == "stem":
        stemmer = nltk.SnowballStemmer(language)
        process_word = stemmer.stem
    else:  # lemmatize
        nltk.download('wordnet', quiet=True)
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        process_word = lemmatizer.lemmatize
    
    # Get stopwords set, with option to extend with custom ones
    stopword_set = set(stopwords.words(language))
    if custom_stopwords:
        stopword_set.update(custom_stopwords)
    
    # Apply transformations
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
    
    # Tokenize text
    words = tokenizer(text)
    
    # Filter stopwords and process words
    if remove_stopwords:
        words = [word for word in words if word and word not in stopword_set]
    if stem_or_lem in ["stem", "lemmatize"]:
        words = [process_word(word) for word in words]
    
    # Join back into string
    return " ".join(words)

# Example usage with DataFrame
def clean_dataframe_column(df, column_name, **kwargs):
    import pandas as pd
    if not isinstance(df, pd.DataFrame) or column_name not in df.columns:
        raise ValueError("Invalid DataFrame or column name")
    df[column_name] = df[column_name].apply(clean, **kwargs)
    return df