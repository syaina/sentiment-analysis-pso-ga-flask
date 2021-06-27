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

X = dataset_load[0][:100]
y = dataset_load[1][:100]
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

def get_X_y_smote():
    X_smote = dataset_load[0][:100]
    y_smote = dataset_load[1][:100]

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
# SIMULASI FITUR SELEKS

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
kernel = 'rbf'
C = 4
GAMMA = 0.01
model = SVC(kernel=kernel, C=C, gamma=GAMMA)

def kfold_train(X, y):
  cv_acc = []
  
  for train, test in kfold.split(X, y): 
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    model_fit = model.fit(X_train, y_train)
    y_pred = model_fit.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_acc.append(acc)
    
  return np.array(cv_acc).mean()

def kfold_training(X, y):
    cv_cr = []

    for train, test in kfold.split(X, y): 
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        model_fit = model.fit(X_train, y_train)
        y_pred = model_fit.predict(X_test)
        cr = classification_report(y_test, y_pred, output_dict=True)
        cv_cr.append(cr)

    return cv_cr

def get_score(cr):
    recall_0 = []
    precision_0 = []
    f1_0 = []

    recall_1 = []
    precision_1 = []
    f1_1 = []

    recall_2 = []
    precision_2 = []
    f1_2 = []

    recall = []
    precision = []
    f1 = []
    accuracy = []

    for i in range(len(cr)):
        recall_0.append(cr[i]['0']['recall'])
        precision_0.append(cr[i]['0']['precision'])
        f1_0.append(cr[i]['0']['f1-score'])

        recall_1.append(cr[i]['1']['recall'])
        precision_1.append(cr[i]['1']['precision'])
        f1_1.append(cr[i]['1']['f1-score'])

        recall_2.append(cr[i]['2']['recall'])
        precision_2.append(cr[i]['2']['precision'])
        f1_2.append(cr[i]['2']['f1-score'])

        recall.append(recall_0)
        recall.append(recall_1)
        recall.append(recall_2)

        precision.append(precision_0)
        precision.append(precision_1)
        precision.append(precision_2)

        f1.append(f1_0)
        f1.append(f1_1)
        f1.append(f1_2)
    
        accuracy.append(cr[i]['accuracy'])

    accuracy = round((np.array(accuracy).mean() * 100) ,2)
    recall = round((np.array(recall).mean() * 100), 2)
    precision = round((np.array(precision).mean() * 100), 2)
    f1 = round((np.array(f1).mean() * 100), 2)

    return accuracy, recall, precision, f1

def f_per_particle(m, alpha):
    total_features = X.shape[1]
    if np.count_nonzero(m) == 0: 
        X_subset = X
    else:
        X_subset = X[:,m==1]

    P = kfold_train(X_subset, y)
    fitness = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return fitness

def f(x, alpha=0.9):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def running_pso(n_particles):
    options = { 'c1': 2, 
                'c2': 2, 
                'w': 0.9, 
                'k': 3, 
                'p': 1}

    dimensions = X.shape[1]
    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensions, options=options)
    cost, pos = optimizer.optimize(f, iters=10, verbose=2)
    return pos, cost

class CustomFitnessFunctionClass:
    def __init__(self,n_total_features,n_splits = 5, alpha=0.01, *args,**kwargs):
        
        self.n_splits = n_splits
        self.alpha = alpha
        self.n_total_features = n_total_features

    def calculate_fitness(self,model,x,y):
        alpha = self.alpha
        total_features = self.n_total_features

        P = kfold_train(x,y)
        fit = (alpha*(1.0 - P) + (1.0 - alpha)*(1.0 - (x.shape[1])/total_features))
        fitness = 1 - fit
        return fitness

def running_ga(n_pop):
    alpha = 0.9
    ff = CustomFitnessFunctionClass(n_total_features=X.shape[1], n_splits=10, alpha=alpha)
    fsga = FeatureSelectionGA(model, X, y, verbose=1, ff_obj=ff)
    pop = fsga.generate(n_pop=n_pop, ngen=10, mutxpb=0.2)

    if (len(pop) > 1):
      pos = np.array(pop[0])
    else : 
      pos = np.array(pop)
    X_subset = X[:, pos==1]
    P = kfold_train(X_subset,y)
    fit = (alpha*(1.0 - P) + (1.0 - alpha)*(1.0 - (X_subset.shape[1])/X.shape[1]))
    fitness = 1 - fit

    return pos, fitness
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
    
@app.route("/fitur-seleksi", methods=['GET', 'POST'])
def show_fitur_seleksi():
    particle_pso = get_pengujian('pengujian-particle_pso.csv')
    populasi_ga = get_pengujian('pengujian-populasi_ga.csv')

    if request.method == 'POST':
        fs = request.form['fs']

        if(fs == 'fs-pso'):
            pso_result = True
            ga_result = False
            n_pop = None

            particles = request.form['particle']
            n_particles = np.array(particles).astype(np.int)
            pso_result = running_pso(n_particles)
            pos = pso_result[0]
            cost = round(pso_result[1],3)

            train = kfold_training(X[:, pos==1],y)
            score = get_score(train)
            accuracy = score[0]
            recall = score[1]
            precision = score[2]
            f1 = score[3]

            return render_template('fitur-seleksi.html',
                particle_pso=particle_pso,
                populasi_ga=populasi_ga,
                pso_result=pso_result,
                ga_result=ga_result,
                pos=pos,
                cost=cost,
                n_particles=n_particles,
                n_pop=n_pop,
                accuracy=accuracy,
                recall=recall,
                precision=precision,
                f1=f1
                )

        elif(fs == 'fs-ga'):
            ga_result = True
            pso_result = False
            n_particles = None

            populasi = request.form['populasi']
            n_pop = np.array(populasi).astype(np.int)
            ga_result = running_ga(n_pop)
            pos = ga_result[0]
            cost = round(ga_result[1],3)

            train = kfold_training(X[:, pos==1],y)
            score = get_score(train)
            accuracy = score[0]
            recall = score[1]
            precision = score[2]
            f1 = score[3]

            return render_template('fitur-seleksi-ga.html',
                particle_pso=particle_pso,
                populasi_ga=populasi_ga,
                pso_result=pso_result,
                ga_result=ga_result,
                pos=pos,
                cost=cost,
                n_particles=n_particles,
                n_pop=n_pop,
                accuracy=accuracy,
                recall=recall,
                precision=precision,
                f1=f1
                )

    else:
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

@app.route("/prediksi-sentimen", methods=['GET', 'POST'])
def show_prediksi_sentimen():
    if request.method == 'POST':
        sentiment = request.form['sentiment']
        model = request.form['model']

        if(model == 'svm'):
            predict = svm_predict(sentiment)
        elif(model == 'pso'):
            predict = pso_predict(sentiment)
        elif(model == 'ga'):
            predict = ga_predict(sentiment)

        return render_template('prediksi-sentimen.html', 
            sentiment=sentiment,
            model=model,
            predict=predict)
    
    else:
        return render_template('prediksi-sentimen.html')

