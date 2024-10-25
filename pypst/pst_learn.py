import numpy as np

def pst_learn(
    f_mat,
    alphabet,
    N,
    L=7,
    p_min=0.0073,
    g_min=0.185,
    r=1.6,
    alpha=17.5,
    p_smoothing=0
):
    """
    PST Learn function based on Ron, Singer, and Tishby's 1996 algorithm "The Power of Amnesia".

    Args:
        f_mat (list): List of frequency tables.
        alphabet (str): String of symbols.
        N (list): Total entries per order.

        L (int): Maximum order (default: 7).
        p_min (float): Minimum occurrence probability (default: 0.0073).
        g_min (float): Minimum transition probability (default: 0.185).
        r (float): Minimum divergence (default: 1.8).
        alpha (float): Smoothing parameter (default: 0).
        p_smoothing (float): Smoothing for probability (default: 0).

    Returns:
        list: A tree array representing the probabilistic suffix tree.
    """

    # Initialize sbar: symbols whose probability >= p_min
    sequence_queue_sbar = [
        [value] for alphabet_index, value in enumerate(alphabet)
        if np.single(f_mat[0][alphabet_index] / N[0]) >= p_min
    ]

    # Initialize tree with empty node
    tbar = [{} for _ in range(L+1)]
    tbar[0] = {
        'string': [[]],
        'parent': [(0, 0)],
        'label': ['epsilon'],
        'internal': [0]
    }

    # Learning process
    while sequence_queue_sbar:
        # this is referred to as S_CHAR in the original code
        cur_sequence = sequence_queue_sbar.pop(0)

        # Convert the sequence to a list of alphabet indexes
        # this is referred to as S_INDEX in the original code
        cur_sequence_indexes = [alphabet.index(item) for item in cur_sequence]

        if len(cur_sequence_indexes) == 0:
            continue

        # Set the current depth in the tree
        cur_depth = len(cur_sequence_indexes)

        # Retrieve a row from f_mat
        #f_vec = f_mat[cur_depth][*cur_sequence_indexes]
        f_vec = get_next_symbol_freqs_for_sequence(f_mat, cur_sequence_indexes)

        # Calculate p(sigma|s) and other probabilities
        # Retrieve a row from f_mat starting from the second element
        if len(cur_sequence_indexes) > 1:
            f_suf = get_next_symbol_freqs_for_sequence(f_mat, cur_sequence_indexes[1:])
            #f_suf = f_mat[cur_depth][tuple(cur_sequence_indexes[1:])]
        else:
            f_suf = get_next_symbol_freqs_for_sequence(f_mat, [])
            #f_suf = f_mat[0] << should be equivalent

        p_sigma_s = f_vec / (np.sum(f_vec) + np.finfo(float).eps)
        p_sigma_suf = f_suf / (np.sum(f_suf) + np.finfo(float).eps)
        ratio = (p_sigma_s + np.finfo(float).eps) / (p_sigma_suf + np.finfo(float).eps)
        psize = p_sigma_s >= (1 + alpha) * g_min

        ratio_test = (ratio >= r) | (ratio <= 1 / r)
        total = np.sum(ratio_test & psize)

        if total > 0:
            tbar[cur_depth].setdefault('string', []).append(cur_sequence_indexes)
            node, depth = find_parent(cur_sequence_indexes, tbar)
            tbar[cur_depth].setdefault('parent', []).append((node, depth))
            tbar[cur_depth].setdefault('label', []).append(cur_sequence)
            tbar[cur_depth].setdefault('internal', []).append(0)



        if len(cur_sequence_indexes) < L:
            # this was already set
            #f_vec = f_mat[len(cur_sequence_indexes)][tuple(cur_sequence_indexes)]
            f_vec = retrieve_f_prime(f_mat, cur_sequence_indexes)
            p_sigmaprime_s = f_vec / (N[cur_depth] + np.finfo(float).eps)
            add_nodes = np.where(p_sigmaprime_s >= p_min)[0]

            for j in add_nodes:
                new_seq = [alphabet[j]] + cur_sequence
                sequence_queue_sbar.append(new_seq)




    # Post-processing for the tree
    tbar = fix_path(tbar)
    tbar = find_gsigma(tbar, f_mat, g_min, N, p_smoothing)

    return tbar



def find_parent(sequence, tbar):
    """
    Find the parent node of the given sequence in the tree.

    Args:
        sequence (list): The current sequence.
        tbar (list): Tree structure.

    Returns:
        tuple: Node and depth of the parent.
    """
    node, depth = 0, 0
    seq_length = len(sequence)
    if seq_length > 1:
        for i in range(2, seq_length+1):
            hits = [
                tbar[i]['string'][j] == sequence[-i:]
                for j in range(len(tbar[i].get('string', [])))
            ]

            if sum(hits) == 1:
                node = np.argmax(hits)
                depth = i

    return node, depth


def fix_path(tbar):
    """
    Fix the paths in the tree by ensuring every node has a clear parent path.
    """
    changes = 1
    while changes:
        changes = 0
        for i in range(2, len(tbar)):
            for j, curr_string in enumerate(tbar[i].get('string', [])):
                node, depth = find_parent(curr_string, tbar)
                parent_depth = tbar[i]['parent'][j][1]
                if depth > parent_depth:
                    tbar[i]['parent'][j] = (node, depth)
                    parent_depth = depth
                if parent_depth < i - 1:
                    tbar[i-1].setdefault('string', []).append(curr_string[1:])
                    node, depth = find_parent(curr_string[1:], tbar)
                    tbar[i-1].setdefault('parent', []).append((node, depth))
                    tbar[i-1].setdefault('label', []).append(tbar[i]['label'][j][1:])
                    tbar[i-1].setdefault('internal', []).append(1)
                    changes += 1
                    tbar[i]['parent'][j] = (len(tbar[i-1]['string']) - 1, i-1)
    return tbar


def get_next_symbol_freqs_for_sequence(f_mat, s):
    """Retrieve the frequency vector for given sequence s.

    Given a sequence, s, this returns a vector of length alphabet size
    that indicates the frequency of next symbol occurrences after the
    sequence s.

    if s is empty, return the ZERO ORDER frequency vector.

    aka retrieve_f_sigma

    """
    idx = tuple(s)
    if len(s) == 0:
        return f_mat[0]

    return f_mat[len(s)][idx]

def retrieve_f_prime(f_mat, s):
    """

    Given a sequence, s, this returns a vector of length alphabet size
    that indicates the frequency of next symbol occurrences after the
    sequence s.

    if s is empty, return the FIRST ORDER frequency vector.

    aka retrieve_f_prime

    """
    idx = tuple(s)
    if len(s) == 0:
        return f_mat[1]

    return f_mat[len(s)][idx]



def find_gsigma(tbar, f_mat, g_min, N, p_smoothing):
    """
    Compute the smoothed transition probabilities for each node in the tree.
    """
    for i in range(len(tbar)):
        for j in range(len(tbar[i].get('string', []))):
            f_vec = get_next_symbol_freqs_for_sequence(f_mat, tbar[i]['string'][j])
            p_sigma_s = f_vec / (np.sum(f_vec) + np.finfo(float).eps)
            if tbar[i]['string'][j]:
                f = retrieve_f(f_mat, tbar[i]['string'][j])
                p_s = f / N[len(tbar[i]['string'][j])]
            else:
                f, p_s = 0, 1
            sigma_norm = len(p_sigma_s)
            g_sigma_s = p_sigma_s * (1 - sigma_norm * g_min) + g_min
            tbar[i].setdefault('g_sigma_s', []).append(g_sigma_s if p_smoothing else p_sigma_s)
            tbar[i].setdefault('p', []).append(p_s)
            tbar[i].setdefault('f', []).append(f)
    return tbar

def retrieve_f(f_mat, s):
    """
    Retrieve the frequency for a given sequence s from the frequency matrix f_mat.

    Parameters:
    - f_mat: A list of frequency matrices for different sequence lengths
    - s: A list of indices representing the sequence (symbols) for which we are retrieving the frequency

    Returns:
    - f: The frequency count for the sequence s
    """
    # Convert sequence s to a tuple for indexing
    idx = tuple(s)

    # Use the length of the sequence to determine which matrix to index into
    if len(s) == 0:
        return np.sum(f_mat[0])  # Return the total sum of 0th-order frequencies if s is empty

    # For non-empty sequences, use the length of s to retrieve the appropriate matrix and index it
    return np.squeeze(f_mat[len(s)][idx])  # Access the correct frequency matrix and index by s
