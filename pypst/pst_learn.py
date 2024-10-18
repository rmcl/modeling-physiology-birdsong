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
    Build a probabilistic suffix tree from frequency matrices.

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
    sbar = [
        a for a in alphabet
        if np.single(f_mat[0][alphabet[a]] / N[0]) >= p_min
    ]

    # Initialize tree with empty node
    tbar = [{
        'string': [[]],
        'parent': [(0, 0)],
        'label': ['epsilon'],
        'internal': [0]
    }] + [
        {
            'string': [],
            'parent': [],
            'internal': [],
            'label': []
        }
        for _ in range(L)
    ]

    # Learning process
    while sbar:
        s_char = sbar.pop(0)
        s = [alphabet[c] for c in s_char]

        curr_depth = len(s) + 1

        # Calculate p(sigma|s) and other probabilities
        f_vec = retrieve_f_sigma(f_mat, s)
        f_suf = retrieve_f_sigma(f_mat, s[1:]) if len(s) > 1 else retrieve_f_sigma(f_mat, [])

        p_sigma_s = f_vec / (np.sum(f_vec) + np.finfo(float).eps)
        p_sigma_suf = f_suf / (np.sum(f_suf) + np.finfo(float).eps)

        ratio = (p_sigma_s + np.finfo(float).eps) / (p_sigma_suf + np.finfo(float).eps)
        psize = (p_sigma_s >= (1 + alpha) * g_min)
        ratio_test = (ratio >= r) | (ratio <= 1 / r)
        total = np.sum(ratio_test & psize)

        if total > 0:
            tbar[curr_depth]['string'].append(s)
            node, depth = find_parent(s, tbar)
            tbar[curr_depth]['parent'].append((node, depth))
            tbar[curr_depth]['label'].append(s_char)
            tbar[curr_depth]['internal'].append(0)

        if len(s) < L:
            f_vec = retrieve_f_prime(f_mat, s)
            p_sigmaprime_s = f_vec / N[curr_depth]
            add_nodes = [i for i, p in enumerate(p_sigmaprime_s) if p >= p_min]

            for j in add_nodes:
                print(j)
                sbar.append(alphabet[j] + s_char)

    # Post-processing for the tree
    tbar = fix_path(tbar)
    tbar = find_gsig(tbar, f_mat, g_min, N, p_smoothing)

    return tbar

def retrieve_f_sigma(f_mat, s):
    """
    Retrieve frequency vector for a given sequence.

    Args:
        f_mat (list): Frequency matrices.
        s (list): Sequence of indices.

    Returns:
        np.ndarray: Frequency vector.
    """
    length = len(s) + 1
    if length == 1:
        return f_mat[0]
    elif length == 2:
        return f_mat[1][s[0], :]
    elif length == 3:
        return np.squeeze(f_mat[2][s[0], s[1], :])
    # Add more cases if needed
    else:
        return np.zeros_like(f_mat[0])

def retrieve_f_prime(f_mat, s):
    """
    Retrieve the f' for a given sequence.

    Args:
        f_mat (list): Frequency matrices.
        s (list): Sequence of indices.

    Returns:
        np.ndarray: Frequency vector.
    """
    length = len(s) + 1
    if length == 2:
        return np.squeeze(f_mat[1][:, s[0]])
    elif length == 3:
        return np.squeeze(f_mat[2][:, s[0], s[1]])
    # Add more cases if needed
    else:
        return np.zeros_like(f_mat[0])

def find_parent(sequence, tbar):
    """
    Find the parent node for a given sequence.

    Args:
        sequence (list): The current sequence.
        tbar (list): Tree structure.

    Returns:
        tuple: Node and depth of the parent.
    """
    node, depth = 1, 1
    seq_length = len(sequence)

    if seq_length > 1:
        for i in range(2, seq_length + 1):
            hits = [len([s for s in tbar[i]['string'] if sequence[-j:] in s]) for j in range(len(sequence))]
            if sum(h > 0 for h in hits) == 1:
                node = hits.index(1) + 1
                depth = max(depth, i)

    return node, depth

def fix_path(tbar):
    """
    Ensure paths are well-formed within the tree.

    Args:
        tbar (list): Tree structure.

    Returns:
        list: Updated tree structure.
    """
    for i in range(2, len(tbar)):
        for j, string in enumerate(tbar[i]['string']):
            node, depth = find_parent(string, tbar)
            parent_depth = tbar[i]['parent'][j][1]
            if depth > parent_depth:
                tbar[i]['parent'][j] = (node, depth)
    return tbar

def find_gsig(tbar, f_mat, g_min, N, p_smoothing):
    """
    Compute smoothed transition probabilities.

    Args:
        tbar (list): Tree structure.
        f_mat (list): Frequency matrices.
        g_min (float): Minimum transition probability.
        N (list): Total occurrences per depth.
        p_smoothing (float): Smoothing parameter.

    Returns:
        list: Updated tree structure.
    """
    for i, level in enumerate(tbar):
        for j, s in enumerate(level['string']):
            f_vec = retrieve_f_sigma(f_mat, s)
            p_sigma_s = f_vec / (np.sum(f_vec) + np.finfo(float).eps)
            p_s = f_vec / N[i] if s else 1
            sigma_norm = len(p_sigma_s)
            g_sigma_s = p_sigma_s * (1 - sigma_norm * g_min) + g_min
            level['g_sigma_s'] = g_sigma_s if p_smoothing else p_sigma_s
            level['p'] = p_s
            level['f'] = f_vec
    return tbar
