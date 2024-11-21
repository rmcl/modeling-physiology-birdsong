from pypst import PST
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import math
import pandas as pd


def train_pst(sequence_dataset, L, alphabet=None):
    """Train a PST of order L on a dataset of sequences."""
    pst = PST(
        L = L,
        p_min = .00073, #0.0073,
        g_min = .01,
        r = 1.6,
        alpha = 17.5,
        alphabet = alphabet
    )
    pst.fit(sequence_dataset)

    return pst


def calculate_metrics(order, syllable_idx, pre_dist, post_dist):
    # Calculate KL Divergence
    kld = entropy(pre_dist, post_dist)

    # Calculate Earth Mover's Distance (EMD) using cumulative difference
    emd = np.sum(np.abs(np.cumsum(pre_dist) - np.cumsum(post_dist)))

    # Calculate Information Gain (IG)
    ig_pre = entropy(pre_dist)
    ig_post = entropy(post_dist)
    ig = ig_pre - ig_post

    return {
        'order': order,
        'syllable_idx': syllable_idx,
        'Kullback-Leibler Divergence': kld,
        'Earth Mover\'s Distance': emd,
        'Information Gain': ig,
        'pre_entropy': ig_pre,
        'post_entropy': ig_post
    }

def plot_before_and_after_distribution(syllables, pre_dist, post_dist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].stem(syllables, pre_dist, basefmt=" ")
    axes[0].set_title("Pre Lesion Distribution")
    axes[0].set_xlabel("Syllables")
    axes[0].set_ylabel("Probability")
    axes[1].stem(syllables, post_dist, basefmt=" ")
    axes[1].set_title("Post Lesion Distribution")
    axes[1].set_xlabel("Syllables")
    plt.tight_layout()
    plt.show()



def build_song_sequences_simple(dataset):
    """Builds song sequences from a dataset of syllables.

    This function ignores syllable length and only considers the order of syllables in a song.
    """
    song_sequences = []
    for result in dataset:
        if len(result['ordered_and_timed_syllables']) == 0:
            continue

        song_sequences.append([
            str(s[0]) for s in result['ordered_and_timed_syllables']
        ])

    return song_sequences



def build_song_sequences_with_timing(dataset):
    """Builds song sequences from a dataset of syllables.

    This function considers the timing of syllables in a song. It works
    by calculating the average length of each syllable and then using that
    to determine the number of times a syllable should be repeated in a song.
    """
    syllables_with_len = []
    for result in dataset:
        for song_syllable in result['ordered_and_timed_syllables']:
            s, start, end = song_syllable

            assert type(s) == str, f"Expected string, got {type(s)}"
            syllables_with_len.append({
                'syllable': s,
                'length': end - start
            })

    df = pd.DataFrame(syllables_with_len)
    syllable_stats = df.groupby("syllable")["length"].agg(["mean", "std"])

    songs = []
    for result in dataset:
        song = []

        if len(result['ordered_and_timed_syllables']) == 0:
            continue

        for song_syllable in result['ordered_and_timed_syllables']:
            s, start, end = song_syllable

            l = end - start

            syllable_mean = syllable_stats['mean'][s]

            num_syllables = math.ceil(l / syllable_mean)

            song.extend([s] * num_syllables)

        songs.append(song)
    return songs
