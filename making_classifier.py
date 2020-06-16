import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
import numpy as np
import pandas as pd
import nltk
import tensorflow_hub as hub #text encoding

from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing import sequence 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
nltk.download("stopwords")
nltk.download('punkt')


embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"


# Read in csv for training data
def get_training(file_name):

    return_data = []
    print("Get Training Data")    
    data = pd.read_csv(file_name, sep=',', header=None)
    data[0] = data[0].replace(r'[~`!@#$%^&*()_+-={[}}|\:;"\'<,>.?/]', '', regex=True)
    
    
    # code from https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    stop_words = set(stopwords.words('english')) 
    
    for sentence in data[0]:
        new_sentence = ''
        word_tokens = word_tokenize(sentence) 
          
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
          
        filtered_sentence = [] 
          
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w) 
      
        for word in filtered_sentence:
            new_sentence += word + " "

        return_data.append(new_sentence)
        
    data[0] = return_data
   
    return data
  #  [print(string) for string in data[0]]

# def testing():

    # dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
    # dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing dataset
    # y_train = dftrain.pop('survived') # pop column "survived" - output
    # y_eval = dfeval.pop('survived')   # pop colum "survived" - output
    
    # for index, value in enumerate(y_train):
        # y_train[index] = value + np.random.randint(5)
        
    # for index, value in enumerate(y_eval):
        # y_eval[index] = value + np.random.randint(5)
        
    # print(y_train)
    
    # CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       # 'embark_town', 'alone']
                       
    # NUMERIC_COLUMNS = ['age', 'fare']

    # feature_columns = []
    # for feature_name in CATEGORICAL_COLUMNS:
        # vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
        # feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    # for feature_name in NUMERIC_COLUMNS:
        # feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    # train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
    # eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

    # linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=10) # explicitly state the number of classes. 
    # linear_est.train(train_input_fn)  # train
    
    # result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

    # pred_dicts = list(linear_est.predict(eval_input_fn))
    # print(pred_dicts[0]['probabilities'])

# from https://colab.research.google.com/drive/1ysEKrw_LE2jMndo1snrZUh5w87LQsCxk#forceEdit=true&sandboxMode=true&scrollTo=Onu8leY4Cn9z
def encode_text(text):
    word_index = imdb.get_word_index() # get the index for all the vocabulary in the imdb dataset
    VOCAB_SIZE = 88584

    MAXLEN = 50
    BATCH_SIZE = 100
    
    tokens = keras.preprocessing.text.text_to_word_sequence(text) # break string to individual works
    tokens = [word_index[word] if word in word_index else 0 for word in tokens] # convert each words into index 
    return sequence.pad_sequences([tokens], MAXLEN)[0] # pad to the same length (10 for now)    
       
      
# multivariant linear regression: y = m1 * x1 + m2 * x2 + m3 *x3 + .... + b 
def multivariant_linear_regression():

    vocab_size = 88584
    embedding_dim = 16
    max_length = 50
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"

    data = get_training("total_product_info.csv")
    training_feature = data[0][:3500]
    training_label = 1/np.array(data[1][:3500]) 
  
    test_feature = data[0][3500::]
    test_label = 1/np.array(data[1][3500::])
    
    # tokenizer
    # tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    # tokenizer.fit_on_texts(training_feature)
    # word_index = tokenizer.word_index
    
    # sequences = tokenizer.texts_to_sequences(training_feature)
    # padded = pad_sequences(sequences, maxlen = max_length, padding=padding_type, truncating=trunc_type)
    
    # test_sequences = tokenizer.texts_to_sequences(test_feature)
    # testing_padded = pad_sequences(test_sequences, maxlen = max_length, padding=padding_type, truncating=trunc_type)

    padded_training = 1/np.array([encode_text(text) for text in training_feature])
    padded_test = 1/np.array([encode_text(text) for text in test_feature])

    #create model 

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), 
        tf.keras.layers.Flatten(),
    #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    model.compile(loss='mean_squared_error',optimizer='adam')
        
    num_epochs = 500
    model.fit(padded_training, training_label, epochs=num_epochs, validation_data=(padded_test, test_label), callbacks=[callback], verbose=1)
    
    
    predict_value = np.array([padded_training[0]])
    print(training_label[0])
    
    
    fake_predict = model.predict(predict_value)
    print(fake_predict[0][0])

    model.save('my_model.h5')
    
    
# load model     
def load_model(string_to_perdict):


    model = keras.models.load_model('my_model.h5')
    
    predict_data = np.array([encode_text(string_to_perdict)])
    
    fake_predict = model.predict(predict_data)
    print(fake_predict[0][0])
    
def train_additional_items(file_name, split):
    model = keras.models.load_model('my_model.h5')
    
    data = get_training(file_name)
    training_feature = data[0][:split] 
    training_label = np.array(data[1][:split])
  
    test_feature = data[0][split::]
    test_label = np.array(data[1][split::])
    
    #encode text
    padded_training = np.array([encode_text(text) for text in training_feature])
    padded_test = np.array([encode_text(text) for text in test_feature])
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)   
    num_epochs = 1000
    model.fit(padded_training, training_label, epochs=num_epochs, validation_data=(padded_test, test_label), callbacks=[callback], verbose=1)
    
    model.save('my_model.h5')
    

if __name__ == '__main__':
    print(tf.__version__)
    
    load_model("College Vegetarian Cookbook: Quick Plant-Based Recipes Every College Student Will Love. Delicious and Healthy Meals for Busy People on a Budget (Vegetarian Cookbook)")
    load_model("Keto Meal Prep Cookbook For Beginners: 600 Easy, Simple & Basic Ketogenic Diet Recipes (Keto Cookbook)")
    load_model("This is a test")
    load_model("dfasdf dsfaf dfsasd dsfadsf dfsadf dfadsf ddgfgdf hghf 4546 213546 215464 21654621 546546sdfa ")
    load_model("Peanutbutter cheesecake for children lunch dinner breakfast")
    load_model("vegetarian food for college students")

    #train_additional_items('total_product_info.csv', 3500)
    #multivariant_linear_regression()

    print("Testing Script for making classifier")