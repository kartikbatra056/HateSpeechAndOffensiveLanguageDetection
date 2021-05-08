from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from build_features import load_process_data
from sklearn.model_selection import cross_val_score
from src.utils import get_class_weights,save_weights,get_logger
import warnings
import numpy as np
import argparse
import os
warnings.filterwarnings('ignore')

def main(path,args,logger,save_path=None,cv=5):

        logger.info('Loading and Processing data...')

        vectorizer, X_train ,y_train = load_process_data(path)

        logger.info('Calculating class weights')

        class_weight = get_class_weights(y_train)

        logger.info('Training model...')

        model, train_score = train(X_train, args, y_train, class_weight)

        logger.info(f'Training f1 score is:{train_score}')

        logger.info(f'Cross validating model with cv = {cv}')

        score_list, mean_score = cross_validate(model,X_train,y_train,cv=cv)

        logger.info(f'Mean Cross validation f1 score is:{mean_score}')

        logger.info(f'List of cross val f1 score is:{score_list}')

        if save_path is not None:
            logger.info('Saving model...')
            save_weights(model,vectorizer,save_path)

def train(X_train, args, y_train, class_weight, SEED=24):
        '''
        Perform training over text data

        Parameters
        ----------
        X_train : input features for training
        y_train : labels for training set
        args : a dict object containing Parameters for training
        class_weight : dict mapping weight for each label
        SEED : to make prediction deterministic

        Returns
        ---------
        model : trained model
        score : F1-score for training data
        '''

        model = SGDClassifier(penalty=args.regression_type,learning_rate=args.scheduler,eta0=args.learning_rate,
                              alpha=args.regularization,class_weight=class_weight,random_state=SEED)

        model.fit(X_train,y_train)

        y_pred = model.predict(X_train)

        score = f1_score(y_train,y_pred,average='macro')

        return model,score


def cross_validate(model,X_train,y_train,cv=5):
        '''
        Perform cross validation over training set

        Parameters
        ----------
        model : trained model for cross validation
        X_train : input features
        y_train : labels
        cv : number of cross validation set

        Returns
        ---------
        mean_score : mean F1-score for cross validation
        score_list : list of F1-score for cross validation
        '''
        score_list = cross_val_score(model,X_train,y_train,cv=cv,scoring='f1_macro')

        mean_score = np.mean(score_list)

        return score_list,mean_score

if __name__=='__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('path',type=str,help='Path to training data')
        parser.add_argument('--regression_type','-rt',type=str,default='elasticnet')
        parser.add_argument('--scheduler','-sc',type=str,default='adaptive')
        parser.add_argument('--learning-rate','-lr',type=float,default=0.1)
        parser.add_argument('--regularization','-reg',type=float,default=0.0003)
        parser.add_argument('--save_weight','-s',action='store_true')

        args = parser.parse_args()

        if args.save_weight:

            SAVE_PATH = 'src/weight'

            if not os.path.exists(SAVE_PATH):
                os.mkdir(SAVE_PATH)
        else:
            SAVE_PATH = None

        PATH = args.path

        log_path = 'logs'

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        logger = get_logger(log_path)

        main(PATH,args,logger,SAVE_PATH)
