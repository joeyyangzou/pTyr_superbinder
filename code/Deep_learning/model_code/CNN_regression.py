#GPU memory
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from tensorflow.keras import layers, Sequential, losses, optimizers
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from sklearn.metrics import r2_score
from tensorflow.keras.constraints import max_norm
from tensorflow import  keras
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from bioencoder.encoder import *
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler,scale


physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.config.list_physical_devices('GPU'))
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)  # Note: This is for tensorflow 2.0 version, which differs from version 1.0.

standard_file_path=""#File path for data standardization processing. This file should generally include all data used for model training and testing.
model_save_path=""#Model save path
data_path=""#Training dataset and independent test set save path
result_save_path=""#Prediction result save path

StandardScalerIN=pd.read_csv(standard_file_path,sep='\t')
StandardScaler_range=StandardScalerIN['value']
StandardScaler_range=StandardScaler_range.values.reshape(-1, 1)
mm = StandardScaler()
mm=mm.fit(StandardScaler_range)
value_transform=mm.transform(StandardScaler_range)

mm_minmax = MinMaxScaler(feature_range=(-1, 1))
mm_minmax=mm_minmax.fit(value_transform)
def process(inputfile_path):
    data = pd.read_csv(inputfile_path, sep='\t', header=0)
    label = data['value']
    inputfile = data['sequence']
    sequence_list = []
    line_number = 0
    for line in inputfile:
        line_number = line_number + 1

    label_reshape = label.values.reshape(-1, 1)

    label_MinMaxScaler = mm.transform(label_reshape)
    label_MinMaxScaler=mm_minmax.transform(label_MinMaxScaler)


    return inputfile, line_number, label,label_MinMaxScaler
'''Perform one-hot encoding on the processed sequence list ↓ '''
def One_Hot(sequence,line_number):
    AA=['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R']
    AA2=['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    encodings = []
    for seq_line in sequence:
        code = []
        seq_line=list(seq_line)
        for aa in seq_line:
            if aa == 'X':
                for aa1 in AA2:
                    tag = 0.05 if aa == aa1 else 0
                    code.append(tag)
            else:
                for aa1 in AA:
                    tag = 1 if aa == aa1 else 0
                    code.append(tag)
        encodings.append(code)
    np.array(encodings)
    encodings=np.reshape(encodings,(line_number,8,20))
    return encodings
'''Build the network structure and compile it ↓ '''

def build_network():
    # First create a list containing multiple network layers
    conv_layers = [

        layers.Conv1D(filters=128, kernel_size=1, padding='same', activation=tf.nn.relu,input_shape=( 8, 20)),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=9, padding='same', activation=tf.nn.relu),
        layers.MaxPooling1D(2, data_format="channels_first"),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu),
        layers.MaxPooling1D(pool_size=2, strides=1),
        layers.Dropout(0.5),
    ]

    fc_layers = [
        layers.Dense(64, activation=tf.nn.relu),
        layers.MaxPooling1D(2),
        layers.Dense(32, activation=tf.nn.relu),
        layers.MaxPooling1D(2),
        layers.Dense(8, activation=tf.nn.relu),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation=tf.nn.tanh)
    ]

    conv_layers.extend(fc_layers)
    network = Sequential(conv_layers)
    network.build(input_shape=[None, 8, 20])
    network.compile(optimizer=optimizers.Adam(), loss='mean_squared_error', metrics=['mae'])  # optimizers.RMSprop(), the loss function here is mean squared error, used for regression prediction. The coefficient of determination R2 is often used in linear regression to represent the percentage of dependent variable variance described by the regression line. If R2 = 1, it means the model perfectly predicts the target variable. (lr=0.01, rho=0.9, epsilon=1e-06)
    network.summary()
    return network


'''Convert data types ↓ '''
def coeff_determination(y_true, y_pred):
    SS_res =K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    return x, y

'''The evaluate function below is used to evaluate the trained model ↓ '''
def evaluate(X, Y,Y_train_noNormalization,X_vali, Y_vali,Y_vali_noNormalization, X_TEST, Y_TEST,Y_test_noNormalization, batch_size=128, epochs=100,line_number_train=0,line_number_vali=0,line_number_test=0):
    classes = sorted([0, 1])
    print(Y)
    X_train, y_train = X, Y
    '''Convert the corresponding dataset to one-hot encoding format'''

    ############One_Hot#####################
    X_train = One_Hot(X_train, line_number_train)  # Training set after one-hot encoding
    X_vali = One_Hot(X_vali, line_number_vali)  # Validation set after one-hot encoding
    X_test = One_Hot(X_TEST, line_number_test)  # Test set after one-hot encoding
    ############One_Hot#####################


    X_train_t = X_train
    X_vali_t = X_vali
    X_test_t = X_test

    X_test_t = tf.cast(X_test_t, dtype=tf.float32)

    '''The following six commands are similar in pairs, first binding X and Y through tf.data format, then performing data type transformation and batch processing in the next line ↓ '''
    # Build training set object, random shuffle, preprocessing, batch processing
    train_db = tf.data.Dataset.from_tensor_slices((X_train_t,y_train))  # First convert pandas dataframe format to tf.data format dataset, this is to prepare for the next step of data shuffling. After converting to tf.data format, the sequence and label values are bound together, and we can process them synchronously
    train_db = train_db.shuffle(len(X)).map(preprocess).batch(batch_size)  # The map function is used for one-click preprocessing of sequences. This command: first randomize the data, then use map function for preprocessing, and use batch function to specify batch size
    # Build validation set object, preprocessing, batch processing
    vali_db = tf.data.Dataset.from_tensor_slices((X_vali_t, Y_vali))
    vali_db = vali_db.shuffle(len(X_vali_t)).map(preprocess).batch(batch_size)
    # Build test set object, preprocessing, batch processing
    test_db = tf.data.Dataset.from_tensor_slices((X_test_t, Y_TEST))
    test_db = test_db.shuffle(len(X_test_t)).map(preprocess).batch(batch_size)
    '''Call the built neural network, train it, and output the test results of the independent test set'''
    network = build_network()
    model_save_file_path=model_save_path
    checkpoint = ModelCheckpoint(model_save_file_path, monitor='val_mae', verbose=1, save_best_only=True, mode='auto',period=50, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    callbacks_list = [early_stopping, checkpoint]

    history = network.fit(train_db, epochs=epochs, validation_data=vali_db,  verbose=1,callbacks = [callbacks_list])  # ,callbacks = [early_stopping]
    print("Independent test:", network.evaluate(test_db))
    tmp_result = np.zeros((len(Y_test_noNormalization), len(classes)))
    predict = network.predict(X_test_t, batch_size=batch_size)
    print(predict)

    predict=mm_minmax.inverse_transform(predict)
    predict = mm.inverse_transform(predict)
    print()
    R_squire = r2_score(Y_test_noNormalization, predict)
    tmp_result[:, 0], tmp_result[:, 1] = Y_test_noNormalization, predict[:, 0]
    network.save(model_save_file_path, save_format='tf')
    return tmp_result, history, R_squire

'''The main function below is mainly used to call the above functions (main function)'''

#Custom function to save results
def save_predict_result(data, output):
    with open(output, 'w') as f:
        f.write('value'+'\t'+'predict'+'\n')
        for i in data:
            f.write('%f\t%f\n' % (i[0], float(i[1])))
    return None


#Run main function
def random_dataset_number(data,label,sample_num):
    '''Randomly generate a set of numbers, with random.seed(1) to keep the numbers used in each loop the same. Here we choose to directly input the sample number; another method is to input the percentage of values: 1、Balance positive and negative samples'''
    import random
    random.seed(1)  # This line of code: if the number in parentheses is 0, the array generated each time is different; if it is 1, the array generated each time is the same
    sample_list = [i for i in range(len(data))]  # [0, 1, 2, 3, 4, 5, 6, 7]
    sample_list = random.sample(sample_list, sample_num)  # Randomly selected [3, 4, 2, 0]
    sample_data = [data[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
    sample_label = [label[i] for i in sample_list]  # [3, 4, 2, 0]

    '''Save list'''
    name1 = ['sequence']
    sequence = pd.DataFrame(columns=name1, data=sample_data)  # Define the table header and data, this line saves the data column
    name2 = ['label']
    label = pd.DataFrame(columns=name2, data=sample_label)  # Define the table header and data, this line saves the label column
    return sample_data,sample_label,sequence,label
def random_dataset_percent(data,label,percent):
    '''Randomly generate a set of numbers, with random.seed(1) to keep the numbers used in each loop the same. Here we choose the percentage of the independent test set: 2、Custom function to extract independent test set'''

    sample_num = int(percent * len(data))  # Assume 20% of the data is used as the independent test set
    random.seed(2)  # This line of code: if the number in parentheses is 0, the array generated each time is different; if it is non-0, the array generated each time is the same, ensuring the independent test set extracted each time is the same
    sample_list_all = [i for i in range(len(data))]  # [0, 1, 2, 3, 4, 5, 6, 7]
    sample_list = random.sample(sample_list_all, sample_num)  # Randomly selected [3, 4, 2, 0]
    sample_difference = list(set(sample_list_all).difference(set(sample_list)))
    sample_test_data = [data[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
    sample_test_label = [label[i] for i in sample_list]  # [3, 4, 2, 0]
    sample_train_data = [data[i] for i in sample_difference]  # ['d', 'e', 'c', 'a']
    sample_train_label = [label[i] for i in sample_difference]  # [3, 4, 2, 0]
    '''Save list'''
    name = ['sequence']
    sequence = pd.DataFrame(columns=name, data=sample_test_data)  # Define the table header and data, this line saves the data column
    name = ['label']
    label = pd.DataFrame(columns=name, data=sample_test_label)  # Define the table header and data, this line saves the label column
    '''Save train_vali list'''
    name = ['sequence']
    sequence_train = pd.DataFrame(columns=name, data=sample_train_data)  # Define the table header and data, this line saves the data column
    name = ['label']
    label_train = pd.DataFrame(columns=name, data=sample_train_label)  # Define the table header and data, this line saves the label column
    return sample_test_data,sample_test_label,sample_train_data,sample_train_label,sequence,label,sequence_train,label_train
'''Process using bioinformatics data↓'''
def main():
    os.chdir(data_path)
    epoch = 1000
    X_train,line_number_train,Y_train,Y_train_MinMaxScaler=process("train")
    X_TEST, line_number_test,Y_TEST,Y_TEST_MinMaxScaler=process("test")
    ###########"""After data standardization, it is normalized to (-1, 1)."""#########################


    x_vali, y_vali = X_TEST, Y_TEST_MinMaxScaler
    line_number_vali=line_number_test
    os.chdir(result_save_path)
    ind_res_test, history,R_squire_test = evaluate(X_train, Y_train_MinMaxScaler,Y_train,x_vali, y_vali,y_vali, X_TEST,Y_TEST_MinMaxScaler,Y_TEST,epochs=epoch, batch_size=128,line_number_train=line_number_train,line_number_vali=line_number_vali,line_number_test=line_number_test) #,R_squire_valiind_res_vali,
    save_predict_result(ind_res_test, 'regression_predict.txt')
    acc = history.history['mae']
    val_acc = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print('R_squire_test:' + str(R_squire_test))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training mae')
    plt.plot(val_acc, label='Validation mae')
    plt.title('Training and Validation mae')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('regression_predict.pdf')

if __name__ == '__main__':
    main()
'''1、Balance positive and negative samples
   2、First split the independent test set by proportion (extract positive and negative samples separately, then merge), add a command to check if the test set file exists
   3、After extracting the independent test set, randomly split the remaining positive and negative samples into training set and validation set by proportion, ensuring the split training set and validation set remain the same during training
   4、Use the existing functions in the original code to combine label and sequence into one file (ensure one-to-one correspondence), then randomize and shuffle
   5、Training set: Validation set: Test set = 6:2:2'''
