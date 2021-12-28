from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
from tensorflow import keras
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, Embedding, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn import preprocessing
import nltk     
nltk.download('stopwords') 
nltk.download('punkt')  
nltk.download('words')                     
#import matplotlib.pyplot as plt           
import random
import numpy as np
import pandas as pd
import re                                  
import string                             
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer 
from nltk import pos_tag, word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

model = tf.keras.models.load_model("D:/ana/sdkpps/model.h5")
df1 = pd.read_csv('D:/ana/sdkpps/Text_Preprocessing_Depres_Suicide_990_Terbaru_Fix_3.csv').astype(str)
df1.columns = ['Label', 'Post']
df1.head()
token = Tokenizer(2772)
token.fit_on_texts(df1['Post'])

def ubah_text_lower(text):
    return text.lower()
    
stop = pd.read_csv("D:/ana/sdkpps/stopwords.txt", names= ["stopwords"], header = None)
stop_words_corpus = list(stopwords.words('indonesian'))
stop_from_list = stop['stopwords'].to_list()
def hapus_stopwords(post):
  filtered_words = [w for w in post if w not in stop_words_corpus]
  filtered_words = [w for w in filtered_words if w not in stop_from_list]
  return filtered_words

def preprocess_filtered(post):
    #Remove 'b
    posted = re.sub(r"b'", ' ', str(post))
    # Remove urls dan trash
    posted = re.sub(r"http\S+|\\x\S+|www\S+|https\S+|dot|com", ' ', str(posted), flags=re.MULTILINE)
    # Remove user @ references and '#' from post
    posted = re.sub(r'\@\w+|\#|\d+', ' ', str(post))
    return posted

def tokenkan(text):
    return word_tokenize(text)

normalizad_word = pd.read_excel("D:/ana/sdkpps/normalisasi.xlsx")
normalizad_word_dict = {}
for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

def hapus_punct_2(text):
  text_nopunct = ''
  text_nopunct = re.sub('['+string.punctuation+']', '', text)
  return text_nopunct

def hapus_punct(text):
    PUNCT_TO_REMOVE = string.punctuation
    punct = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    return punct

def text_preprocessing(text):
  df = pd.DataFrame()
  df['text_lower'] = text.apply(ubah_text_lower)
  df['filtered'] = df['text_lower'].apply(preprocess_filtered)
  df['token'] = df['filtered'].apply(tokenkan)
  df['normalized'] = df['token'].apply(normalized_term)
  df['normalized'] = df['normalized'].apply(hapus_stopwords)

  # create stemmer
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()

  # stemmed
  def stemmed_wrapper(term):
      return stemmer.stem(term)

  term_dict = {}

  for document in df['normalized']:
      for term in document:
          if term not in term_dict:
              term_dict[term] = ' '
              
  #print(len(term_dict))
  print("------------------------")

  for term in term_dict:
      term_dict[term] = stemmed_wrapper(term)
      #print(term,":" ,term_dict[term])
      
  #print(term_dict)
  #print("------------------------")


  # apply stemmed term to dataframe
  def stemmed_term(document):
      return [term_dict[term] for term in document]

  df['stemmed'] = df['normalized'].swifter.apply(stemmed_term)
  print("------------------------")
  #print(df['stemmed'])

  df['stemmed'] = df['stemmed'].astype(str)
  df['stemmed'] = df['stemmed'].apply(hapus_punct_2)
  return df['stemmed']

def home(request):
    return render(request, 'index.html')

def classify(request):
    #Get the text from client side if we not get any text then set the text = default
    djtext = request.GET.get('text', 'default')

    #if we get any text ie. text != default
    if djtext != "default":
        datax = {'Postingan':[djtext]}
        dataf = pd.DataFrame(datax)
        dataf = text_preprocessing(dataf['Postingan'])

        X_sample = token.texts_to_sequences(dataf)
        X_sample = pad_sequences(X_sample, 100)
        y_sample = model.predict(X_sample)
        hasil = np.argmax(y_sample)

        if(hasil == 0):
          predicted = 'tidak mengidap gangguan psikologis'
        else:
          predicted = 'mengidap gangguan psikologis'

    if djtext == "default":
        predicted = "Anda belum memasukan text postingan sosmed. Ayo coba lagi"

    #now our data will sent back to the client side but it can only be send in form of Json
    params = {'Category': predicted} #We created a Json and set the value od this dict to predicted
    return render(request, 'result.html', params) #And we are returning a new file result.html and the predicted class

