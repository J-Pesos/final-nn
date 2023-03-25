from nn import nn, preprocess, io
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility.
np.random.seed(15)

# Create simple neural networks for testing.
nn_test = nn.NeuralNetwork([{'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'},
                            {'input_dim': 1, 'output_dim': 3, 'activation': 'sigmoid'}
                            ],
                            lr = 0.1,
                            seed = 15,
                            batch_size = 3,
                            epochs = 5,
                            loss_function = 'bce'
                            )

def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    '''
    Unit test to ensure implementation calculates binary cross entropy correctly.
    '''
    # Generate y and y_hat. Manual calculation of bce is 7.37.
    y = np.array( [0., 1., 1., 1., 0.] )
    y_hat = np.array( [0., 1., 1., 0., 1.] )

    # Instantiate mse calculated by implementation.
    bce_method = nn_test._binary_cross_entropy(y, y_hat)

    # Assert that method bce matches manual calculation.
    assert round(bce_method, 2) == 7.37

def test_binary_cross_entropy_backprop():
    '''
    Unit test to ensure implementation calculates binary cross entropy backprop correctly.
    '''
    # Generate y and y_hat.
    y = np.array( [0., 1., 1., 1., 0.] )
    y_hat = np.array( [0., 1., 1., 0., 1.] )

    # Manual calculation of backprop.
    bce_bprop = np.array( [0.2, -0.2, -0.2, -20000000.0, 19999999.9] )

    # Instantiate bce backprop calculated by implementation.
    bce_bprop_method = nn_test._binary_cross_entropy_backprop(y, y_hat)

    # Round bce_prop_method values.
    round_bce_bprop_method = [round(i, 2) for i in bce_bprop_method]

    # Assert that method mse backprop matches manual calculation.
    assert np.all(bce_bprop == round_bce_bprop_method)

def test_mean_squared_error():
    '''
    Unit test to ensure implementation calculates mean squared error correctly.
    '''
    # Generate y and y_hat. Manual calculation of mse is 0.4.
    y = np.array( [0, 1, 1, 1, 0] )
    y_hat = np.array( [0, 1, 1, 0, 1] )

    # Instantiate mse calculated by implementation.
    mse_method = nn_test._mean_squared_error(y, y_hat)

    # Assert that method mse matches manual calculation.
    assert mse_method == 0.4

def test_mean_squared_error_backprop():
    '''
    Unit test to ensure implementation calculates mean squared error backprop correctly.
    '''
    # Generate y and y_hat.
    y = np.array( [0, 1, 1, 1, 0] )
    y_hat = np.array( [0, 1, 1, 0, 1] )

    # Manual calculation of backprop is an array of errors depending on differences in y and y_hat.
    mse_bprop = np.array( [ 0.,  0.,  0., -0.4,  0.4] )

    # Instantiate mse backprop calculated by implementation.
    mse_bprop_method = nn_test._mean_squared_error_backprop(y, y_hat)

    # Assert that method mse backprop matches manual calculation.
    assert np.all(mse_bprop == mse_bprop_method)

def test_sample_seqs():
    '''
    Unit test to ensure sampling of sequences have relatively balanced classes.
    '''
    alphabet = ['A', 'T', 'C', 'G']
    seqs = []

    # Create list of unbalanced sequences and corresponding labels.
    for seq in range(1000):
        seq = []
        for char in range(17):
            seq += np.random.choice(alphabet)
        seqs += [seq]

    labels = [True for lab in range(800)] + [False for x in range(200)]

    # Perform balanced sampling.
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)

    # Create separate lists for sampled labels.
    pos_labs = []
    neg_labs = []
    for lab in sampled_labels:
        if lab == True:
            pos_labs += [lab]
        else:
            neg_labs += [lab]

    # Assert that sampled sequences + labels are same size as original lists.
    assert len(seqs) == len(sampled_seqs), 'Sampled sequences do not match original list length.'
    assert len(labels) == len(sampled_labels), 'Sampled labels do not match original list length.'

    # Assert that positive and sequences are relatively balanced (0.05 error) based on length of their sampled lists.
    assert abs( len(pos_labs) - len(neg_labs) ) < 50, 'Classes are not balanced after sampling.' 

def test_one_hot_encode_seqs():
    '''
    Unit test to ensure one_hot_encoding is encodes nucleotide sequences correctly.
    '''
    # Initialize sequences to encode.
    seqs = ['ATCG',
            'GCTA']

    # Store what the actual encodings should be.
    actual_encodings = np.array( [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]]
                                  )
    
    encoded_seqs = preprocess.one_hot_encode_seqs(seqs)

    # Assert that the actual encodings matches one-hot encoding.
    assert np.all(actual_encodings == encoded_seqs), 'One-hot encoding is not outputting the expected sequence encoding.'