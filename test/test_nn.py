from nn import nn, preprocess
import numpy as np

# Set random seed for reproducibility.
np.random.seed(15)

# Create simple neural networks for testing.
nn_test = nn.NeuralNetwork([{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'},
                            ],
                            lr = 0.1,
                            seed = 15,
                            batch_size = 1,
                            epochs = 1,
                            loss_function = 'mse'
                            )

def test_single_forward():
    '''
    Unit test to ensure implementation performs a correct single forward step.
    '''
    # Use own inputs and perform single forward step.
    A_curr, Z_curr = nn_test._single_forward(np.array( [[30, 40, 30, 20]] ),
                                             np.array( [[1]] ),
                                             np.array( [1, 2, 3, 4] ),
                                             'relu')

    # Assert single forward equals manually calculated values.
    assert A_curr == np.array( [[281]] )
    assert Z_curr == np.array( [281] )

def test_forward():
    '''
    Unit test to ensure implementation performs successful full forward pass.
    '''
    # Create own inputs and assess that forward pass is calculating as expected.
    nn_test._param_dict = {"b1": np.array([[1], [2]]),
                           "W1": np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
                           "b2": np.array([[1]]),
                           "W2": np.array([[1, 4]]),
                           }
    
    # Perform a forward pass.
    y_hat, cache = nn_test.forward(np.array([1, 2, 3, 4]))

    # Assert y_hat equals manually calculated value.
    assert y_hat == np.array( [[120]] )

def test_single_backprop():
    '''
    Unit test to ensure implementation performs a correct backprop step.
    '''
    # Use own inputs and perform single backprop step.
    dA_prev, dW_curr, db_curr = nn_test._single_backprop(np.array([[20, 30, 40, 20]]),
                                                        np.array([[1]]),
                                                        np.array([[1]]),
                                                        np.array([[1, 2, 3, 4]]),
                                                        np.array([[5]]),
                                                        "relu"
                                                        )

    # Assert single backprop step equals mannually calculated values.
    assert np.array_equal(dA_prev, np.array([[100, 150, 200, 100]]))
    assert np.array_equal(dW_curr, np.array([[5, 10, 15, 20]]))
    assert np.array_equal(db_curr, np.array([[5]]))

def test_predict():
    '''
    Unit test to ensure predictions have the same dimensions as the provided labels.
    '''
    X_train = np.random.rand(75, 3)
    y_train = np.random.rand(75, 1) 
    X_test = np.random.rand(25, 3)
    y_test = np.random.rand(25, 1)

    fit = nn_test.fit(X_train, y_train, X_test, y_test)

    pred = nn_test.predict(X_test)

    assert y_test.shape == pred.shape

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
    y = np.array( [0.5, 0.5, 0.4, 0.5, 0.4] )
    y_hat = np.array( [0.4, 0.5, 0.4, 0.4, 0.4] )

    # Manual calculation of backprop is an array of errors depending on differences in y and y_hat.
    bce_bprop = np.array( [-0.08, 0.0, 0.0, -0.08, 0.0] )

    # Instantiate mse backprop calculated by implementation.
    bce_bprop_method = nn_test._binary_cross_entropy_backprop(y, y_hat)

    # Round bce_prop_method values.
    round_bce_bprop_method = [round(i, 2) for i in bce_bprop_method]

    # Assert that method mse backprop matches manual calculation.
    assert np.all(bce_bprop == np.array(round_bce_bprop_method) )

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