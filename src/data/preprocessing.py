import re
import string

from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords=True, lower_case=True, remove_punctuations=True, remove_digits=True, remove_extraspaces=True,
                 stemming=False, lemmatization=False):
        self.remove_stopwords = remove_stopwords
        self.lower_case = lower_case
        self.remove_punctuations = remove_punctuations
        self.remove_digits = remove_digits
        self.remove_extraspaces = remove_extraspaces
        self.stemming = stemming
        self.lemmatization = lemmatization
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.iloc[:, 0].copy()
        if self.remove_stopwords:
            stop_words = set(stopwords.words('english'))
            X_ = X_.apply(lambda x: " ".join([item for item in x.split() if item not in stop_words]))
            
        if self.lower_case:
            X_ = X_.str.lower()
            
        if self.remove_punctuations:
            X_ = X_.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
            
        if self.remove_digits:
            X_ = X_.str.replace('\d+', '', regex=True)
            
        if self.remove_extraspaces:
            X_ = X_.apply(lambda x: re.sub(' +', ' ', x))
                        
        if self.stemming:
            stemmer = PorterStemmer()
            X_ = X_.apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
                        
        if self.lemmatization:
            lemmatizer = WordNetLemmatizer()
            X_ = X_.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
        
        return X_