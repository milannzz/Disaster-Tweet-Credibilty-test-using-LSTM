# Importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import LSTM, Dense, Embedding,SpatialDropout1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping

# Importing Data
df = pd.read_csv(r"data\train.csv")
X = df.text
Y = df.target
df_test = pd.read_csv(r"data\test.csv")
X_test = df_test.text

# Preprocessing
def Preprocessing(tweets):
    from nltk.corpus import stopwords
    corpus = []
    stopwords = set(stopwords.words('english'))
    for tweet in tweets:
        tweet = tweet.lower()
        tweet = re.sub(r"http\S+", "",tweet)
        tweet = re.sub(r'[0-9\.]+', '',tweet)
        tweet = re.sub('[^a-zA-Z]'," ",tweet)
        tweet = re.sub(' +', ' ',tweet)
        tweet = tweet.split()
        tweet = [word for word in tweet if not word in stopwords]
        tweet = ' '.join(tweet)
        corpus.append(tweet)
    return corpus

X = Preprocessing(X)
X_test = Preprocessing(X_test)

# Tokenizetion
max_words = 15000
max_len = 250
Embedding_dim = 100
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

word_index = tok.word_index
print('Found %s unique tokens.' % len(word_index))

model = Sequential()
model.add(Embedding(max_words, Embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(256,activation = "relu"))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

model.fit(sequences_matrix,Y,batch_size=64,epochs=5,validation_split=0.1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

#perdiction
y_pred = model.predict(test_sequences_matrix)

for i in range(len(y_pred)):
    if y_pred[i] > 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

#submission
submission = df_test
submission = submission.drop(["keyword","text","location"],axis=1)
submission["target"] = y_pred
submission.to_csv('submission.csv',index = False ,float_format ='%.0f')


