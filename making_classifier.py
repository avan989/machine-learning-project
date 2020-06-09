import tensorflow as tf
import numpy as np
import pandas as pd

# Read in csv for training data
def get_training():

    print("Get Training Data")
    file_name = "total_product_info.csv"
    
    data = pd.read_csv(file_name, sep=',', header=None)
    
    print(data)  



if __name__ == '__main__':
    layers = tf.keras.layers

    print(tf.__version__)

    get_training()
    print("Testing Script for making classifier")