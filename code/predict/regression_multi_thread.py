import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse

# Configure GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPU devices
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load path configuration (modify according to actual path)
model_save_path = "model/CNN_regression_model/"  # Replace with actual model path
standard_file_path = "standard_regression_file/SH2_R24_SUM_fre_regression.csv"  # Replace with standard data file path

# Initialize scalers (consistent with training)
data = pd.read_csv(standard_file_path, sep=',')
standard_values = data['value'].values.reshape(-1, 1)

# Create and fit scalers
scaler = StandardScaler().fit(standard_values)
minmax_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(scaler.transform(standard_values))

def one_hot_encode(sequence):
    """Encode 8 amino acid sequence into one-hot format"""
    AA = ['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 
          'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 
          'E', 'D', 'K', 'R']
    
    # Check sequence length
    if len(sequence) != 8:
        raise ValueError("Input sequence must be 8 amino acids long")
    
    encoding = []
    for aa in sequence:
        if aa == 'X':  # Handle unknown amino acid
            encoding += [0.05]*20
        else:
            encoding += [1 if aa == aa_class else 0 for aa_class in AA]
    return np.array(encoding).reshape(8, 20)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict amino acid sequence activity score')
    parser.add_argument('input_file', help='Input sequence file path')
    parser.add_argument('output_file', help='Output result file path')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    return parser.parse_args()

def predict_in_batches(model, input_file, output_file, batch_size):
    """Batch prediction and save results"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        batch = []
        for line in infile:
            sequence = line.strip()
            if sequence:
                batch.append(sequence)
                if len(batch) >= batch_size:
                    process_batch(model, batch, outfile)
                    batch = []
        
        if batch:  # Process remaining data less than one batch
            process_batch(model, batch, outfile)

def process_batch(model, batch, outfile):
    """Process a single batch"""
    try:
        # Batch encoding
        encoded_seqs = [one_hot_encode(seq) for seq in batch]
        
        # Convert to tensor and add batch dimension
        input_tensors = [tf.expand_dims(tf.constant(seq, dtype=tf.float32), axis=0) for seq in encoded_seqs]
        input_data = tf.concat(input_tensors, axis=0)
        
        # Batch prediction
        normalized_scores = model.predict(input_data, batch_size=len(batch))
        
        # Inverse normalization
        for seq, norm_score in zip(batch, normalized_scores[:,0]):
            score = minmax_scaler.inverse_transform([[norm_score]])
            original_score = scaler.inverse_transform(score)
            outfile.write(f"{seq}\t{round(original_score[0][0], 4)}\n")
    except Exception as e:
        print(f"Error processing batch: {str(e)}")

# 使用示例
if __name__ == "__main__":
    args = parse_args()
    model = tf.keras.models.load_model(model_save_path)
    predict_in_batches(model, args.input_file, args.output_file, args.batch_size)
