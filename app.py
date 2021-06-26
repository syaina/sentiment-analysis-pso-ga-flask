from flask import *
app = Flask(__name__)

# ===================================== import libraries =================================================
import pandas as pd 
import numpy as np

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pyswarms as ps
from feature_selection_ga import FeatureSelectionGA

import pickle
import os
import codecs

import re

def pickle_load (path, filename):
  loaded_pickle = pickle.load(open(path+filename, 'rb'))
  return loaded_pickle
# =========================================================================================================

SAVED_FOLDER = './saved-files'
app.config['SAVED_FOLDER'] = SAVED_FOLDER

raw_df = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'dataset_30-05-21.csv'))
replace_word = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'replace_word_list.csv'))
w2v_file = codecs.open(os.path.join(app.config['SAVED_FOLDER'], 'w2v_sastrawi_200-3-25_50.txt'), encoding='utf-8')
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=False)
dataset_load = pickle_load(app.config['SAVED_FOLDER'], '/smote_200-3-25_50.smote')
pso_load = pickle_load(app.config['SAVED_FOLDER'], '/pso-1_10_10.pso')
ga_load = pickle_load(app.config['SAVED_FOLDER'], '/ga-2_20_10.ga')

svm_model = pickle_load(app.config['SAVED_FOLDER'], '/svm')
pso_model_load = pickle_load(app.config['SAVED_FOLDER'], '/pso')
ga_model_load = pickle_load(app.config['SAVED_FOLDER'], '/ga')
pso_model = pso_model_load[0]
sf_pso = pso_model_load[1]
ga_model = ga_model_load[0]
sf_ga = ga_model_load[1]

fit_sc = pickle_load(app.config['SAVED_FOLDER'], '/fit_sc')

# =========================================================================================================
def build_word_vector(tokens, size, w2v_model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += w2v_model.wv[word].reshape((1,size))
            count += 1
        except KeyError: 
            continue
    
    if count != 0:
        vec /= count
    return vec

# =========================================================================================================
def preprocessing(dataset):
    sentiment = dataset[['full_text']]
    df = sentiment.copy()
    df['cleansing'] = df['full_text'].str.replace('@[^\s]+','')
    df['cleansing'] = df['cleansing'].str.replace('(#[A-Za-z0-9]+)','')
    df['cleansing'] = df['cleansing'].str.replace('http\s+','')
    df['cleansing'] = df['cleansing'].str.replace('(\w*\d\w*)','')
    df['cleansing'] = df['cleansing'].str.replace('&amp;',' ')
    df['cleansing'] = df['cleansing'].str.replace('[^A-Za-z\s\/]',' ')
    df['cleansing'] = df['cleansing'].str.replace('[^\w\s]',' ')
    df['cleansing'] = df['cleansing'].str.replace('\s+',' ')

    df['case_folding'] = df['cleansing'].apply(lambda x: x.lower())
   
    replace_word_dict = {}
    for i in range(replace_word.shape[0]):
        replace_word_dict[replace_word['before'][i]] = replace_word['after'][i]

    df['normalize_text'] = df['case_folding'].apply(lambda x : ' '.join(replace_word_dict.get(i, i) for i in x.split()), 1)
    df['tokenization'] = df['normalize_text'].apply(lambda x: x.split())
    
    factory = StopWordRemoverFactory()
    sastrawi_stopwords = factory.get_stop_words()
    df['stopword_removal'] = df['tokenization'].apply(lambda x: [word for word in x if word not in sastrawi_stopwords])
    
    df_cleansing = df[['full_text', 'cleansing']]
    df_case_folding = df[['cleansing', 'case_folding']]
    df_normalize_text = df[['case_folding', 'normalize_text']]
    df_tokenization = df[['normalize_text', 'tokenization']]
    df_stopword_removal = df[['tokenization', 'stopword_removal']]

    return df_cleansing, df_case_folding, df_normalize_text, df_tokenization, df_stopword_removal, df

def get_pengujian(filename):
    df_pengujian = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], filename))
    pengujian = df_pengujian.to_html(classes='table table-sm table-responsive table-bordered table-fixed table-with-num')

    return pengujian

def get_X_y():
    preprocessing_result = preprocessing(raw_df)
    df = preprocessing_result[-1]
    X = np.concatenate([build_word_vector(z, 200, w2v_model) for z in map(lambda x: x, df['sastrawi_stopword_removal'])])
    sc = StandardScaler()
    fit_sc = sc.fit(X)
    X = sc.transform(X)

    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df['sentiment'])

    df_train = pd.DataFrame(X)
    df_train['target'] = pd.DataFrame(y)

    return X, y, df_train

def get_X_y_smote():
    X_smote = dataset_load[0][:50]
    y_smote = dataset_load[1][:50]

    df_train_smote = pd.DataFrame(X_smote)
    df_train_smote['target'] = pd.DataFrame(y_smote)

    return X_smote, y_smote, df_train_smote

def get_training_result(filename):
    df = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], filename))
    result = df.to_html(classes='table table-sm table-responsive table-bordered table-fixed')

    return result

def get_replace_word_dict():
    replace_word_dict = {}
    for i in range(replace_word.shape[0]):
        replace_word_dict[replace_word['before'][i]] = replace_word['after'][i]
    return replace_word_dict
 
def cleansing(sentence): 
    sentence = re.sub('@[^\s]+','', sentence)
    sentence = re.sub('(#[A-Za-z0-9]+)','', sentence)
    sentence = re.sub('http\s+','', sentence)
    sentence = re.sub('(\w*\d\w*)','', sentence)
    sentence = re.sub('&amp;',' ', sentence)
    sentence = re.sub('[^A-Za-z\s\/]',' ', sentence)
    sentence = re.sub('[^\w\s]',' ', sentence)
    sentence = re.sub('\s+',' ', sentence)
    return sentence

def preprocess(sentence):
    replace_word_dict = get_replace_word_dict()
    sentiment = ''
    sentence = cleansing(sentence)
    sentence = sentence.lower().split()
    for word in sentence:
        sentiment += ' ' + replace_word_dict.get(word, word)
    sentence = sentiment.split()
    return sentence

def svm_predict(sentence):
    sentence = preprocess(sentence)
    vect_sentence = np.concatenate([build_word_vector(sentence, 200, w2v_model)])
    vect_sentence_norm = fit_sc.transform(vect_sentence)
    predict = svm_model.predict(vect_sentence_norm.reshape(-1,200))
    return predict[0]

def pso_predict(sentence):
    sentence = preprocess(sentence)
    vect_sentence = np.concatenate([build_word_vector(sentence, 200, w2v_model)])
    vect_sentence_norm = fit_sc.transform(vect_sentence)
    vect_sentence_norm_fs = vect_sentence_norm[:, sf_pso==1]
    predict = pso_model.predict(vect_sentence_norm_fs.reshape(-1,138))
    return predict[0]

def ga_predict(sentence):
    sentence = preprocess(sentence)
    vect_sentence = np.concatenate([build_word_vector(sentence, 200, w2v_model)])
    vect_sentence_norm = fit_sc.transform(vect_sentence)
    vect_sentence_norm_fs = vect_sentence_norm[:, sf_ga==1]
    predict = ga_model.predict(vect_sentence_norm_fs.reshape(-1,124))
    return predict[0]
    

# =======================================================================================================

@app.route("/")
def show_index():
    data_html = raw_df.to_html(classes='table table-sm table-responsive table-bordered dataset')
    row = raw_df.shape[0]
    col = raw_df.shape[1]

    return render_template('index.html', 
        table=data_html, 
        col=col, 
        row=row)

@app.route("/pengolahan-data")
def show_pengolahan_data():
    preprocessing_result = preprocessing(raw_df)
    df_cleansing = preprocessing_result[0].to_html(classes='table table-sm table-responsive table-bordered table-fixed')
    df_case_folding = preprocessing_result[1].to_html(classes='table table-sm table-responsive table-bordered table-fixed')
    df_normalize_text = preprocessing_result[2].to_html(classes='table table-sm table-responsive table-bordered table-fixed')
    df_tokenization = preprocessing_result[3].to_html(classes='table table-sm table-responsive table-bordered table-fixed')
    df_stopword_removal = preprocessing_result[4].to_html(classes='table table-sm table-responsive table-bordered table-fixed')
    
    dimensi_w2v = get_pengujian('pengujian-dimensi_w2v.csv')
    window_w2v = get_pengujian('pengujian-window_w2v.csv')
    epoch_w2v = get_pengujian('pengujian-epoch_w2v.csv')

    get_data_train = get_X_y_smote()
    data_train = get_data_train[-1].to_html(classes='table table-sm table-responsive table-bordered')

    return render_template('pengolahan-data.html', 
        cleansing=df_cleansing, 
        case_folding=df_case_folding,
        normalize_text=df_normalize_text,
        tokenization=df_tokenization,
        stopword_removal=df_stopword_removal,
        dimensi_w2v=dimensi_w2v,
        window_w2v=window_w2v,
        epoch_w2v=epoch_w2v,
        data_train=data_train)
    
@app.route("/fitur-seleksi")
def show_fitur_seleksi():
    particle_pso = get_pengujian('pengujian-particle_pso.csv')
    populasi_ga = get_pengujian('pengujian-populasi_ga.csv')

    return render_template('fitur-seleksi.html',
        particle_pso=particle_pso,
        populasi_ga=populasi_ga
        )

@app.route("/training-evaluasi")
def show_training_evaluasi():
    gamma_svm = get_pengujian('pengujian-gamma_svm.csv')
    c_svm = get_pengujian('pengujian-c_svm.csv')
    fold_pso = get_training_result('fold_pso.csv')
    fold_ga = get_training_result('fold_ga.csv')

    return render_template('training-evaluasi.html',
        gamma_svm=gamma_svm,
        c_svm=c_svm,
        fold_pso=fold_pso,
        fold_ga=fold_ga)

@app.route("/prediksi-sentimen")
def show_prediksi_sentimen():
    return render_template('prediksi-sentimen.html')

