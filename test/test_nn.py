from nn import nn, preprocess, io
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility.
np.random.seed(15)

# Create neural networks for testing.
nn_test_class = nn.NeuralNetwork([{'input_dim': 68, 'output_dim': 34, 'activation': 'sigmoid'},
                            {'input_dim': 34, 'output_dim': 17, 'activation': 'sigmoid'},
                            {'input_dim': 17, 'output_dim': 1, 'activation': 'sigmoid'}
                            ],
                            lr = 0.1,
                            seed = 15,
                            batch_size = 10,
                            epochs = 10,
                            loss_function = 'bce'
                            )

nn_test_autoenc = nn.NeuralNetwork([{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                    {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'},
                                    ],
                                    lr = 0.1,
                                    seed = 15,
                                    batch_size = 10,
                                    epochs = 10,
                                    loss_function = 'mse'
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
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    '''
    Unit test to ensure sampling of sequences have relatively equal balancing.
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
    pass