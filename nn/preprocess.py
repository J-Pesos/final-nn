# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Create array for labels.
    labels_array = np.array(labels)

    # Get positive and negative sequence indices based on True and False labels.
    pos_seqs = np.where(labels_array == True)[0]
    neg_seqs = np.where(labels_array == False)[0]

    # Initialize sequences and sample lists.
    sampled_seqs = []
    sampled_labels = []

    for seq_idx in range( len(pos_seqs) + len(neg_seqs) ):
        # Set float that represents a value above or below 0.5 probability.
        p = np.random.uniform()
        # Sample from positive sequences.
        if p < 0.5:
            sample = int( np.random.uniform(0, len(pos_seqs)) ) # Select random index of a positive samples.
            sampled_labels += [ labels[ pos_seqs[sample] ] ]
            sampled_seqs += [ seqs[ pos_seqs[sample] ] ]
        # Sample from negative sequences.
        else:
            sample = int( np.random.uniform(0, len(neg_seqs)) ) # Select random index of a negative samples.
            sampled_labels += [ labels[ neg_seqs[sample] ] ]
            sampled_seqs += [ seqs[ neg_seqs[sample] ] ]

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Define nucleotide alphabet.
    alphabet = set( ('A', 'T', 'C', 'G') )

    # Define encoding dictionary.
    encoding_dict = {'A': [1, 0, 0, 0],
                 'T': [0, 1, 0, 0],
                 'C': [0, 0, 1, 0],
                 'G': [0, 0, 0, 1]
                 }
    
    # Initialize encoded sequence.
    encodings = []

    for seq in seq_arr:
        # Assert that there is no character in the sequence 
        assert len( set(seq) ) == len( set(alphabet) ), 'There is charater in the sequence that is not in the nucleotide alphabet.'
        enc_seq = []

        for nuc in seq:
            enc_seq += encoding_dict[nuc]
        
        encodings += [enc_seq]

    return encodings