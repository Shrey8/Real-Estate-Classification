# Script to deploy model locally

# import Flask and jsonify
from flask import Flask, jsonify, request
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import re
import numpy as np
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import gensim
from collections import Counter
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)
from numpy import dot
from numpy.linalg import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import pickle


# Goal is to build an API that will tell us loan probabilites when it receives the information. Will use Flask
app = Flask(__name__)
api = Api(app)

# Load Model
loaded_model = pickle.load(open('hist_gradient_boosting_finalized_model.sav', 'rb'))

# Import custom built made classes so pickle object can communicate with them 

#### Numerical Pipeline
# Grab Numerical Data
def numFeat(data):
    cat_feats = data.dtypes[data.dtypes == 'object'].index.tolist()
    num_feats = data.dtypes[~data.dtypes.index.isin(cat_feats)].index.tolist()
    return data[num_feats]

# Create above function into a FunctionTransformer
keep_num = FunctionTransformer(numFeat)

# Create Feature Transformer on select columns (only numerical in our case)
class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, data, **transform_params):
        price_difference = np.abs(data['price_id1'] - data['price_id2'])
        bedroom_difference = np.abs(data['bedrooms_id1'] - data['bedrooms_id2'])
        bathroom_difference = np.abs(data['bathrooms_id1'] - data['bathrooms_id2'])
        area_difference = np.abs(data['totalArea_id1'] - data['totalArea_id2'])
        apartment_dummy_difference = np.abs(data['apartment_dummy_1'] - data['apartment_dummy_2'])
        house_dummy_difference = np.abs(data['house_dummy_1'] - data['house_dummy_2'])
        plot_dummy_difference = np.abs(data['plot_dummy_1'] - data['plot_dummy_2'])
        investment_dummy_difference = np.abs(data['investment_dummy_1'] - data['investment_dummy_2'])
        other_dummy_difference = np.abs(data['other_dummy_1'] - data['other_dummy_2'])
        
        features = pd.DataFrame()
        features['price_difference'] = price_difference
        features['bedroom_difference'] = bedroom_difference
        features['bathroom_difference'] = bathroom_difference
        features['area_difference'] = area_difference
        features['apartment_dummy_difference'] = apartment_dummy_difference
        features['house_dummy_difference'] = house_dummy_difference
        features['plot_dummy_difference'] = plot_dummy_difference
        features['investment_dummy_difference'] = investment_dummy_difference
        features['other_dummy_difference'] = other_dummy_difference
        
        #data = features

        return features       
    
    
    def fit(self, data, y=None, **fit_params):
        return self
    
    
##### Grab String Data for Word2Vec
# Grab String Data
def catFeat(data):
    cat_feats = data.dtypes[data.dtypes == 'object'].index.tolist()
    #num_feats = data.dtypes[~data.dtypes.index.isin(cat_feats)].index.tolist()
    return data[cat_feats]

# Create above function into a FunctionTransformer
keep_cat = FunctionTransformer(catFeat)

class Word2VecTransformer():
    def __init__(self, columns=None):
        self.columns = columns
        self.Word2VecTitle = Word2Vec.load("/Users/Shrey/LHL_Notes/Final_Project/casaData/TestWord2vecTitle.model")
        self.Word2VecDescription = Word2Vec.load("/Users/Shrey/LHL_Notes/Final_Project/casaData/TestWord2vecDescription.model")

    def transform(self, df, **transform_params):
        
        #Title Columns
    
        title_1 = []
        for i in df['title_id1']:
            title_1.append(re.sub(r'\W+', ' ', i.lower()))
            
        # Tokenize words in title_1
        title_1 = [nltk.word_tokenize(sentence) for sentence in title_1]
        
        # Remove stopwords
        for i in range(len(title_1)):
            title_1[i] = [word for word in title_1[i] if word not in stopwords_dict]
        
        # Clean string data for title_id2 column
        title_2 = []
        for i in df['title_id2']:
            title_2.append(re.sub(r'\W+', ' ', i.lower()))
            
        # Tokenize words in title_2
        title_2 = [nltk.word_tokenize(sentence) for sentence in title_2]
        
        # Remove stopwords
        for i in range(len(title_2)):
            title_2[i] = [word for word in title_2[i] if word not in stopwords_dict]
            
        title = title_1 + title_2
        
        self.Word2VecTitle.build_vocab(title, update=True)
        
        self.Word2VecTitle.train(title, total_examples=self.Word2VecTitle.corpus_count ,epochs=1)
        
        title_1_vector_sums = []
        for i in range(len(title_1)):
            vec = []
            for word in title_1[i]:
                vec.append(self.Word2VecTitle.wv[word])
            if len(vec) > 0:
                title_1_vector_sums.append(sum(vec)/len(vec))
            else:
                title_1_vector_sums.append(sum(vec)/(len(vec)+1))
            
        title_2_vector_sums = []
        for i in range(len(title_2)):
            vec = []
            for word in title_2[i]:
                vec.append(self.Word2VecTitle.wv[word])
            if len(vec) > 0:
                title_2_vector_sums.append(sum(vec)/len(vec))
            else:
                title_2_vector_sums.append(sum(vec)/(len(vec)+1))
                
        test_t1np = np.asarray(title_1_vector_sums)
        
        test_t2np = np.asarray(title_2_vector_sums)
        
        
        # Description Columns 
        description_id1 = []
        for i in df['description_id1']:
            description_id1.append(re.sub(r'\W+', ' ', i ).lower())
            
        # Tokenize description_id1
        description_id1 = [nltk.word_tokenize(sentence) for sentence in description_id1]
        
        # Remove Stopwords from description_id1
        for i in range(len(description_id1)):
            description_id1[i] = [word for word in description_id1[i] if word not in stopwords_dict]
        
        # Clean string data for description_id2 column
        description_id2 = []
        for i in df['description_id2']:
            description_id2.append(re.sub(r'\W+', ' ', i ).lower())
            
        # Tokenize description_id2
        description_id2 = [nltk.word_tokenize(sentence) for sentence in description_id2]
         
        # Remove Stopwords from description_id2
        for i in range(len(description_id2)):
            description_id2[i] = [word for word in description_id2[i] if word not in stopwords_dict]
        
        # Combine tokenized columns
        description = description_id1 + description_id2
        
        self.Word2VecDescription.build_vocab(description, update=True)
        
        self.Word2VecDescription.train(description, total_examples=self.Word2VecDescription.corpus_count ,epochs=1)
        
        description_1_vector_sums = []
        for i in range(len(description_id1)):
            vec = []
            for word in description_id1[i]:
                vec.append(self.Word2VecDescription.wv[word])
            if len(vec) > 0:
                description_1_vector_sums.append(sum(vec)/len(vec))
            else:
                description_1_vector_sums.append(np.ones(100))
                
                
        description_2_vector_sums = []
        for i in range(len(description_id2)):
            vec = []
            for word in description_id2[i]:
                vec.append(self.Word2VecDescription.wv[word])
            if len(vec) > 0:
                description_2_vector_sums.append(sum(vec)/len(vec))
            else:
                description_2_vector_sums.append(np.ones(100))
                
        test_d1np = np.asarray(description_1_vector_sums)
        
        test_d2np = np.asarray(description_2_vector_sums)    
        
        ## Title Cosine Similarities 

        #Calculate description cosine similarity
        test_description_cos_similarity = []
        for i in range(len(test_d1np)):
                test_description_cos_similarity.append(np.dot(test_d1np[i],test_d2np[i])/(norm(test_d1np[i])*norm(test_d2np[i])))
                
        # Calculate test cosine similarity
        test_title_cos_similarity = []
        for i in range(len(test_t1np)):
                test_title_cos_similarity.append(np.dot(test_t1np[i],test_t2np[i])/(norm(test_t1np[i])*norm(test_t2np[i])))
                
                
        features = pd.DataFrame()
        features['description_cos_similarity'] = test_description_cos_similarity
        features['title_cos_similarity'] = test_title_cos_similarity
        
        return features
    
    def fit(self, data, y=None, **fit_params):
        return self
    

num_pipeline = Pipeline([
    ("num_feats", keep_num),
    ("new_features" , SelectColumnsTransformer())])

cat_pipeline = Pipeline([
    ("cat_feats", keep_cat),
    ("word_2_vec", Word2VecTransformer())])

all_features = FeatureUnion([
    ('numeric_features', num_pipeline),
    ('categorical_features', cat_pipeline)])

main_pipeline = Pipeline([
    ('all_features', all_features),
    ('modeling', loaded_model)
])


        
# Now, we need to create an endpoint where we can communicate with our ML model. This time, we are going to use POST request.
class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        test = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        convert_dict = {'price_id1': float,
                         'bedrooms_id1': float,
                         'bathrooms_id1': float,
                         'totalArea_id1': float,
                         'price_id2': float,
                         'bedrooms_id2': float,
                         'bathrooms_id2': float,
                         'totalArea_id2': float,
                         'apartment_dummy_1': float,
                         'house_dummy_1': float,
                         'plot_dummy_1': float,
                         'investment_dummy_1': float,
                         'other_dummy_1': float,
                         'apartment_dummy_2': float,
                         'house_dummy_2': float,
                         'plot_dummy_2': float,
                         'investment_dummy_2': float,
                         'other_dummy_2': float,
                         'title_id1': str,
                         'title_id2': str,
                         'description_id1': str,
                         'description_id2': str}
        test = test.astype(convert_dict) 
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = main_pipeline.predict(test)
        #status = 'First value probability of not getting a loan and second value is probability of getting a loan'
        # we cannot send numpt array as a result
        return res.tolist() 

    
# assign endpoint
api.add_resource(Scoring, '/scoring')


# The last thing to do is to create an application run when the file api.py is run directly (not imported as a module from another script).

if __name__ == '__main__':
    app.run(debug=True)

# Test JSON values below in Postman to get response 
# {"price_id1": 100000,
#  "bedrooms_id1": 4,
#  "bathrooms_id1": 3,
#  "totalArea_id1": 3500,
#  "price_id2": 990000,
#  "bedrooms_id2": 4,
#  "bathrooms_id2": 3,
#  "totalArea_id2": 3500,
#  "apartment_dummy_1": 0,
#  "house_dummy_1": 1,
#  "plot_dummy_1": 0,
#  "investment_dummy_1": 0,
#  "other_dummy_1": 0,
#  "apartment_dummy_2": 0,
#  "house_dummy_2": 1,
#  "plot_dummy_2": 0,
#  "investment_dummy_2": 0,
#  "other_dummy_2": 0,
#  "title_id1": "Beautiful Place in Madrid",
#  "title_id2": "Beautiful Place in downtown Madrid",
#  "description_id1": "4 bedroom apartment in Madrid",
#  "description_id2": "4 bedroom apartment in downtown Madrid"}