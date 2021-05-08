# import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import preprocess_text
from nltk.corpus import stopwords
import pandas as pd
import nltk

try :
    STOPWORDS = stopwords.words('english') # load STOPWORDS
except :
    nltk.download('stopwords')
    STOPWORDS = stopwords.words('english')  # load STOPWORDS

def load_process_data(path):

    df = pd.read_csv(path,index_col=[0]) # load dataset

    X_train , y_train = df['tweet'], df['class']

    X_train = X_train.map(preprocess_text) # preprocess text

    vectorizer = TfidfVectorizer(max_features=10000,  # vectorizer
                                 min_df=2,ngram_range=(1,2),stop_words=STOPWORDS)

    X_train = vectorizer.fit_transform(X_train.values.tolist()) # apply TfidfVectorizer on training data

    return vectorizer,X_train,y_train
