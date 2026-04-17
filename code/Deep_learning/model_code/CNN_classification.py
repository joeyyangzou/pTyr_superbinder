#GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tkinter import  filedialog
import  numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from matplotlib.ticker import NullFormatter
import pandas as pd
import os
from tensorflow.keras import layers, Sequential, losses, optimizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import roc_curve, auc
from numpy import interp, math
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.config.list_physical_devices('GPU'))

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

model_save_path=""#Binary classification model save path
metrics_path=""#Confusion matrix path
data_path=""#Training set and independent test set storage path
result_path=""#Model prediction results and loss curve storage path

'''Process the original sequence and generate a list for one-hot encoding ↓ '''
def process(inputfile_path):
    data = pd.read_csv(inputfile_path, sep='\t', header=0)
    lable = data['label']
    inputfile = data['sequence']
    sequence_list = []
    line_number = 0
    for line in inputfile:
        line_number = line_number + 1
    return inputfile, line_number, lable
'''Perform one-hot encoding on the processed sequence list ↓ '''
def One_Hot(sequence,line_number):
    AA=['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R']
    encodings = []
    for seq_line in sequence:
        code = []
        seq_line=list(seq_line)
        print(seq_line)
        for aa in seq_line:
                for aa1 in AA:
                    tag = 1 if aa == aa1 else 0
                    code.append(tag)
        encodings.append(code)
    np.array(encodings)
    encodings=np.reshape(encodings,(line_number,8,20))
    return encodings
'''Build the network structure and compile it ↓ '''


def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):  # Calculate performance metrics at threshold 0.5
    my_metrics = {  # First declare a dictionary with corresponding KEY values
        'SN': 'NA',
        'SP': 'NA',
        'ACC': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:  # If positive sample
            if scores[i] >= cutoff:  # Threshold is 0.5, if score is greater than 0.5
                tp = tp + 1  # tp+1  Predicted positive and actual positive
            else:
                fn = fn + 1  # Actual positive but predicted negative
        else:  # If negative sample
            if scores[i] < cutoff:  # Score below threshold, predicted negative and actual negative
                tn = tn + 1  # tn+1
            else:
                fp = fp + 1  # Score above threshold, predicted positive but actual negative

    my_metrics['SN'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'  # SN: Sensitivity
    my_metrics['SP'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'  # SP: Specificity
    my_metrics['ACC'] = (tp + tn) / (tp + fn + tn + fp)  # ACC: Accuracy
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ( tp + fp) * ( tp + fn) * ( tn + fp) * ( tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'  # Precision
    my_metrics['Recall'] = my_metrics['SN']  # Recall
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    return my_metrics
def drow_roc(y_true, pred):
    fpr, tpr, thresholds = roc_curve(y_true, pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=1, alpha=0.7, label='ROC (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def build_network():
    # 先创建包含多网络层的列表
    conv_layers = [
        layers.Conv1D(filters=128, kernel_size=1, padding='same', activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=9, padding='same', activation=tf.nn.relu),
        layers.MaxPooling1D(2,1),
       layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu),
        layers.MaxPooling1D(pool_size=2, strides=1),
        layers.Dropout(0.7),
    ]

    fc_layers = [
        layers.Dense(64, activation=tf.nn.relu),
        layers.MaxPooling1D(2,1),
        layers.Dense(32, activation=tf.nn.relu),
        layers.MaxPooling1D(2,1),
        layers.Dense(8, activation=tf.nn.relu),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation=tf.nn.sigmoid)
    ]

    conv_layers.extend(fc_layers)
    network = Sequential(conv_layers)
    network.build(input_shape=[ None,8, 20])
    network.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    network.summary()
    return network


'''Convert data types ↓ '''
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y

'''The evaluate function below is used to evaluate the trained model ↓ '''
mean_fpr = np.linspace(0, 1, 100)
def evaluate(X, Y,X_vali, Y_vali, X_TEST, Y_TEST, batch_size=512, epochs=100,line_number_train=0,line_number_vali=0,line_number_test=0):
    classes = sorted([0,1])# #set() function creates an unordered collection of unique elements, can perform relationship tests, remove duplicate data, and calculate intersection, difference, union, etc.
    print(Y)
    X_train, y_train = X, Y
    '''Convert the corresponding dataset to one-hot encoding format'''
    X_train = One_Hot(X_train,line_number_train) #Training set after one-hot encoding
    X_vali = One_Hot(X_vali, line_number_vali) #Validation set after one-hot encoding
    X_test = One_Hot(X_TEST,line_number_test) #Test set after one-hot encoding

    X_train_t = X_train
    X_vali_t = X_vali
    X_test_t = X_test
    X_test_t = tf.cast(X_test_t, dtype=tf.float32)
    '''The following six commands are similar in pairs, first binding X and Y through tf.data format, then performing data type transformation and batch processing in the next line ↓ '''
    # Build training set object, random shuffle, preprocessing, batch processing
    train_db = tf.data.Dataset.from_tensor_slices((X_train_t, y_train)) #First convert pandas dataframe format to tf.data format dataset, this is to prepare for the next step of data shuffling. After converting to tf.data format, the sequence and label values are bound together, and we can process them synchronously
    train_db = train_db.shuffle(len(X)).map(preprocess).batch(batch_size) #The map function is used for one-click preprocessing of sequences. This command: first randomize the data, then use map function for preprocessing, and use batch function to specify batch size
    # Build validation set object, preprocessing, batch processing
    vali_db = tf.data.Dataset.from_tensor_slices((X_vali, Y_vali))
    vali_db = vali_db.shuffle(len(X_vali_t)).map(preprocess).batch(batch_size)
    # Build test set object, preprocessing, batch processing
    test_db = tf.data.Dataset.from_tensor_slices((X_test_t, Y_TEST))
    test_db = test_db.shuffle(len(X_test_t)).map(preprocess).batch(batch_size)
    '''Call the built neural network, train it, and output the test results of the independent test set'''
    network = build_network()

    checkpoint = ModelCheckpoint( model_save_path,monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto',period=20, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    callbacks_list = [checkpoint,early_stopping]
    history = network.fit(train_db,  validation_data=vali_db,epochs=epochs,verbose=1,callbacks = [callbacks_list])#,callbacks = [early_stopping]
    print("Independent test:", network.evaluate(test_db))
    tmp_result = np.zeros((len(Y_TEST), len(classes)))
    predict=network.predict(X_test_t, batch_size=batch_size)
    tmp_result[:, 0], tmp_result[:, 1] = Y_TEST, predict[:, 0]
    matrix=[]
    drow_roc(Y_TEST,predict)
    matrix.append(calculate_metrics(Y_TEST, predict))
    df = pd.DataFrame(matrix).to_csv(metrics_path)
    network.save(model_save_path, save_format='tf')
    return tmp_result, history, Y_TEST

'''The main function below is mainly used to call the above functions (main function)'''
def main():
    os.chdir(data_path)
    epoch = 200
    '''0、Extract information from positive and negative samples, including sequence and label'''
    positive_sequence, positive_linenumber, positive_label = process('positive')
    negative_sequence, negative_linenumber, negative_label = process('negative')
    '''1、Balance positive and negative samples, and combine the corresponding data'''
    # print('line_number:'+str(positive_sequence))
    # print('chang du:'+str(len(positive_sequence)))
    # positive_sequence_balance,positive_label_balance,p_sequence_balance,p_label_balance= random_dataset_number(positive_sequence, positive_label, sample_num=positive_linenumber) #sample_num is the number of positive and negative samples to extract: 6102
    # negative_sequence_balance,negative_label_balance,n_sequence_balance,n_label_balance= random_dataset_number(negative_sequence, negative_label, sample_num=positive_linenumber) #sample_num is the number of positive and negative samples to extract: 6102
    # sequence_balance = pd.concat([p_sequence_balance, n_sequence_balance])  # Combine sequence data from positive and negative samples with 'sequence' header
    # label_balance = pd.concat([p_label_balance, n_label_balance])  # Combine sequence data from positive and negative samples with 'label' header
    # balance_merge = pd.concat([sequence_balance, label_balance],axis=1,names=['sequence','label'])  # Combine sequence data and label data
    # balance_merge.to_csv('balance.csv', encoding='gbk',sep='\t')  # Save test set data to folder
    # balance_sequence, balance_linenumber, balance_label = process('balance.csv')
    # print('balance_sequence:'+str(balance_sequence))
    '''2、Divide the independent test set, first check if the independent test set exists in the folder'''
    if os.path.isfile('test_+10.csv'):
        print('Test set file exists')
        independent_test_sequence, independent_test_linenumber, independent_test_label=process('test_+10.csv')
        train_vali_sequence, train_vali_linenumber, train_vali_label = process('train+vali_+10.csv')
    #'''The steps in else: 1、First extract sequence and label of positive and negative samples for independent test set according to proportion. 2、Combine the extracted sequence and label of positive and negative samples into one file. 3、Combine sequence and label into one file and save'''
    else:
        print('Training set and test set not divided!!!','\n')
        positive_sequence_test, positive_label_test,positive_sequence_train, positive_label_train,p_sequence_test,p_label_test,p_sequence_train_vali,p_label_test_train_vali = random_dataset_percent(positive_sequence, positive_label,percent=0.2)  # Independent test set positive sample extraction ratio is 0.2, sequence_test, label_test are data with headers, the other two sequence and label have no headers
        negative_sequence_test, negative_label_test, negative_sequence_train, negative_label_train,n_sequence_test,n_label_test,n_sequence_train_vali,n_label_test_train_vali = random_dataset_percent(negative_sequence, negative_label,percent=0.2)  # Independent test set positive sample extraction ratio is 0.2
        independent_test_sequence = pd.concat([p_sequence_test, n_sequence_test]) #Combine sequence data from positive and negative samples with 'sequence' header
        independent_test_label = pd.concat([p_label_test, n_label_test]) #Combine sequence data from positive and negative samples with 'label' header

        test_merge = pd.concat([independent_test_sequence, independent_test_label], axis=1,names=['sequence','label']) #Combine sequence data and label data
        test_merge.to_csv('test_+10.csv', encoding='gbk',sep='\t') #Save test set data to folder
        independent_test_sequence, independent_test_linenumber, independent_test_label = process('test_+10.csv')

        '''Use the remaining dataset after extracting test set to split into training set and validation set, do some preprocessing first'''
        sequence_train_vali = pd.concat([p_sequence_train_vali, n_sequence_train_vali])  # Combine sequence data from positive and negative samples with 'sequence' header
        label_train_vali = pd.concat([p_label_test_train_vali, n_label_test_train_vali])  # Combine sequence data from positive and negative samples with 'label' header
        train_vali_merge = pd.concat([sequence_train_vali, label_train_vali], axis=1,names=['sequence', 'label'])  # Combine sequence data and label data
        train_vali_merge.to_csv('train+vali_+10.csv', encoding='gbk', sep='\t')  # Save test set data to folder
        train_vali_sequence, train_vali_linenumber, train_vali_label = process('train+vali_+10.csv')
        print('##################################Test set division successful, training set and test set file names are: train+vali_+10.csv','\t','test_+10######################################','\n')
    '''3、Next, split the training set and validation set'''
    from sklearn.model_selection import train_test_split
    os.chdir(result_path)
    '''4、Get the independent test set that can be input into the model'''
    X_TEST=independent_test_sequence
    Y_TEST=independent_test_label
    x_vali, y_vali=X_TEST,Y_TEST
    ind_res, history, y_test = evaluate(train_vali_sequence, train_vali_label,x_vali, y_vali, X_TEST,Y_TEST,epochs=epoch, batch_size=64,line_number_train=train_vali_linenumber,line_number_vali=independent_test_linenumber,line_number_test=independent_test_linenumber)
    save_predict_result(ind_res, 'dp_sh_221005_20%_test.txt')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('dp_sh_221005_20%_test.png')

#Custom function to save results
def save_predict_result(data, output):
    with open(output, 'w') as f:
        f.write('# result for true and predict \n')
        for i in data:
            f.write('%d\t%f\n' % (i[0], float(i[1])))
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

    import pandas as pd
    '''Save list'''
    name1 = ['sequence']
    sequence = pd.DataFrame(columns=name1, data=sample_data)  # Define the table header and data, this line saves the data column
    name2 = ['label']
    label = pd.DataFrame(columns=name2, data=sample_label)  # Define the table header and data, this line saves the label column
    return sample_data,sample_label,sequence,label
def random_dataset_percent(data,label,percent):
    '''Randomly generate a set of numbers, with random.seed(1) to keep the numbers used in each loop the same. Here we choose the percentage of the independent test set: 2、Custom function to extract independent test set'''
    import random
    sample_num = int(percent * len(data))  # Assume 20% of the data is used as the independent test set
    random.seed(2)  # This line of code: if the number in parentheses is 0, the array generated each time is different; if it is non-0, the array generated each time is the same, ensuring the independent test set extracted each time is the same
    sample_list_all = [i for i in range(len(data))]  # [0, 1, 2, 3, 4, 5, 6, 7]
    sample_list = random.sample(sample_list_all, sample_num)  # Randomly selected [3, 4, 2, 0]
    sample_difference = list(set(sample_list_all).difference(set(sample_list)))
    sample_test_data = [data[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
    sample_test_label = [label[i] for i in sample_list]  # [3, 4, 2, 0]
    sample_train_data = [data[i] for i in sample_difference]  # ['d', 'e', 'c', 'a']
    sample_train_label = [label[i] for i in sample_difference]  # [3, 4, 2, 0]
    import pandas as pd
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

if __name__ == '__main__':
    main()

