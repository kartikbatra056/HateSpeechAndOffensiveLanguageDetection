# import libraries
from sklearn.metrics import f1_score,classification_report
from src.utils import get_logger,preprocess_text
import pandas as pd
import unittest
import logging
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        data_path = 'data/processed/test.csv'  # loading test data

        weight_path = 'src/weight'

        df = pd.read_csv(data_path,index_col=[0])

        vectorizer = joblib.load(os.path.join(weight_path,'vectorizer.pkl'))

        cls.model = joblib.load(os.path.join(weight_path,'model.pkl'))

        cls.X_test , cls.y_test = vectorizer.transform(df['tweet'].map(preprocess_text)) , df['class']

    def test_score(self):

        y_pred = self.__class__.model.predict(self.__class__.X_test)

        score = f1_score(self.__class__.y_test,y_pred,average='macro')

        self.assertGreater(score,0.75)
