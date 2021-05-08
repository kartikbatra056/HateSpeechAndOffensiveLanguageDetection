# Import libraies
from sklearn.model_selection import train_test_split
from src.utils import get_logger
import pandas as pd
import os

SEED = 24

path = 'data/raw/labeled_data.csv'  # path to raw dataset

save_path = 'data/processed' # path to save split dataset

log_path = 'logs'

if not os.path.exists(save_path): # check if directory already exists
        os.mkdir(save_path)

if not os.path.exists(log_path): # check if directory already exists
        os.mkdir(log_path)

logger = get_logger(log_path)

logger.info('Loading data...')

df = pd.read_csv(path,index_col=[0]) # loading dataset

logger.info('Splitting data into training and testing set.')
train_df,test_df = train_test_split(df,test_size=0.25,stratify=df['class'],random_state=SEED) # split training and testing dataset

logger.info('Saving splitted data...')
train_df.to_csv(os.path.join(save_path,'train.csv'))    # save training dataset

test_df.to_csv(os.path.join(save_path,'test.csv'))      # save testing dataset
