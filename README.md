
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

# Folder Structure 

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
