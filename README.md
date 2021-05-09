
# Objective

Billions of people have started using different social networking platform for their purpose.
Social networking platform has given us opportunity to connect with people with same interest and collaborate with them easily. 
Likely said it has various benefits but one major problem arising nowadays is the spread of hate speech and use of offensive
language against a person or a community.
Use of such kind of language can harm people's sentiments or beliefs.
Spread of such information could be controlled if we can detect it properly.
Machine learning can help us detect such hateful and offensive language.

# Dataset 

Dataset for the project was downloaded from kaggle : [Hate Speech and Offensive Language dataset](https://www.kaggle.com/mrmorj/hate-speech-and-offensive-language-dataset)
```
It is a twitter dataset which can be used for Multiclass Text Classification task to Classify text into three categories :
                  1. Hate Speech 
                  2. Offensive Language 
                  3. Neither 

The text data is really messy and contains text that can be considered racist, sexist, homophobic, or generally offensive.
Which is actually good for our project as data resembles with real world scenario.
```

# Directory Structure 

Following folder structure is inspired from [Cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/#directory-structure)
I have just slightly modified the structure for my purpose. 

```
.
├── app
│   ├── app.py                                  # flask code that works as backend  
│   ├── static                                  # folder containing css files for web page
│   │   └── style.css       
│   └── templates                               # folder containing html file for web page
│       └── base.html       
├── data
│   ├── processed                               # folder containing preprocessed dataset (in this case just splitted the dataset into train-test) 
│   │   ├── test.csv
│   │   └── train.csv
│   └── raw                                     # folder containing raw dataset 
│       └── labeled_data.csv
├── evaluate                                    # folder containing file to run evaluation test
│   └── evaluation.py
├── logs                                        # logs generated while training 
│   ├── error.err
│   └── info.log
├── notebooks                                   # folder containing notebooks for initial exploration
│   └── 01-initial-exploration.ipynb
└── src                                         # folder containing main model files
    ├── __init__.py
    ├── data                                    # file to process dataset 
    │   └── make_dataset.py
    ├── models                                  # file containing main model 
    │   ├── __init__.py
    │   ├── build_features.py
    │   └── train_model.py
    ├── utils.py                                # file containing utility functions for training and inference 
    └── weight                                  # folder containing saved weights of model and other artifacts
        ├── model.pkl                           # Saved a trained model as pickle  
        └── vectorizer.pkl                      # tfidf vectorizer saved as pickle format to use while inference and evaluation 
        
```          

# Getting Started

Here I've mentioned steps to running over linux OS or WSL2(Bash). If you prefer Windows command line please make necessary minor changes accordingly.

## Initial Steps
* Install Python3
* Clone repository ```git clone https://github.com/kartikbatra056/HateSpeechAndOffensiveLanguageDetection.git``` 
* Enter cloned directory ```cd HateSpeechAndOffensiveLanguageDetection```
* Set ```export PYTHONPATH=.                      # prevents from getting module not found error```
* Create a virtual environment ```python3 -m venv env```
* Activate virtual environment ```source ./env/bin/activate```
* Install dependencies ```pip3 install -r requirements.txt```

## Create or process dataset

You can also skip this as it just splits dataset in ```data/raw``` directory into training and test set storing it into ```data/processed```  

* Generate or process dataset ```python3 src/data/make_dataset.py```        

## Train model
You can train model as follow 
```

# first lets know the parameter to be passed while training so run following command

python3 src/models/train_model.py -h 

# you will get output as below

positional arguments:
  path                  Path to training data

optional arguments:
  -h, --help            show this help message and exit
  --regression_type REGRESSION_TYPE, -rt REGRESSION_TYPE
  --scheduler SCHEDULER, -sc SCHEDULER
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
  --regularization REGULARIZATION, -reg REGULARIZATION
  --save_weight, -s
  
 # Next to train model 
 
 python3 src/models/train_model.py path
 
 # all logs generated will be stored into logs directory
```
## Evaluate model
You need to evaluate model before passing it to deployment. For which I have written a simple evaluation test using ```unittest``` module in python which tests model's F1_score on test set is higher than a certain limit for following model which is 0.75 which means if model has ```F1_score (macro) > 0.75``` then model has passed the test and ready for deployment. In real world scenario we have to re-train model due to change in data with time this test helps verfiy that our model is better than the previous model. Though I have considered a single criterion (F1_score) you can use another criterion like accuracy,F1_score(weighted).      

```
# To evaluate model you just need to run following command 
python3 -m unittest evaluate/evaluation.py
```

## Deploy model

* Deploy the app ```python3 app/app.py```
* Open web browser and go to ```http://localhost:5000```

# Methods used 

1. Exploratory Data Analysis
2. Data cleaning and preprocessing 
3. Feature extraction using ```TFidfVectorizer```
4. Model Building 
5. Model Deployment

# Technologies

* Python 
* scikit-learn
* nltk
* regex
* flask 
* HTML
* CSS 

# Website

You can find the webiste running live at [HerokuApp](https://hateandoffensivelangdetector.herokuapp.com/).


Just Copied a bit of text from Eminem's Song ```Without me``` (Just Eminem Fan) you can see the results below. 

![Deployed model](https://github.com/kartikbatra056/HateSpeechAndOffensiveLanguageDetection/blob/master/model.JPG)

# Contact

* Contact on [linkedin](https://www.linkedin.com/in/kartik-batra-ba3380174/)
* Mail me [kartikbatra16012001@gmail.com](mailto:kartikbatra16012001@gmail.com)
