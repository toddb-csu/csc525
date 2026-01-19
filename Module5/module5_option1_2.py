# Todd Bartoszkiewicz
# CSC525: Introduction to Machine Learning
# Module 5: Critical Thinking Option #1
#
# Option #1: Text Dataset Augmentation
# State-of-the-art machine learning models use millions, sometimes billions, of parameters. When training our models,
# the number of distinct examples we need is proportional to the number of parameters our model has.
#
# Invariance is a trait of a neural network model (for instance, an image classification model) that can robustly
# classify objects even when the objects are placed in different orientations, have different sizes, and have
# illumination differences. Many of these potential differences in our dataset can be created artificially, instead of
# collecting more images. Methods could be to alter the brightness or contrast of the image, stretch or skew operations,
# or a variety of translation methods.
#
# But what about textual data for NLP projects? Possibilities include synonym replacement, which replaces words with
# words that mean the same thing, random inserts, swaps, and deletions, among others.
#
# For your assignment, submit a Python script that will take any text dataset and augment it in some way to expand the
# dataset. Submission must include a script that will augment any text dataset within its folder. Please include the
# un-augmented dataset with the augmented dataset and a short description of what was augmented.
#
import os
import sys
import random
import pandas as pd
import numpy as np
from typing import List
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# Auto-download NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)


class NLPAugmenter:
    def __init__(self, num_augs=5, sr_prob=0.2, ri_prob=0.1, rs_prob=0.15, rd_prob=0.1):
        self.num_augs = num_augs
        self.sr_prob, self.ri_prob, self.rs_prob, self.rd_prob = sr_prob, ri_prob, rs_prob, rd_prob

    def get_synonyms(self, word: str) -> List[str]:
        """Extract synonyms via WordNet."""
        synonyms = []
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                syn = lemma.name().replace('_', ' ').lower()
                if syn != word and syn not in synonyms and len(syn.split()) == 1:
                    synonyms.append(syn)
        return synonyms[:2]  # Top 2 only

    def synonym_replacement(self, words: List[str]) -> List[str]:
        new_words = words.copy()
        random_indices = [idx for idx, _ in enumerate(words) if random.random() < self.sr_prob]
        for idx in random_indices:
            syns = self.get_synonyms(words[idx])
            if syns: new_words[idx] = random.choice(syns)
        return new_words

    def random_insertion(self, words: List[str]) -> List[str]:
        new_words = words.copy()
        i = 0
        while i < len(new_words) and random.random() < self.ri_prob:
            syns = self.get_synonyms(new_words[i])
            if syns:
                new_words.insert(i, random.choice(syns))
            i += 1
        return new_words

    def random_swap(self, words: List[str]) -> List[str]:
        new_words = words.copy()
        num_swaps = max(1, int(len(words) * self.rs_prob))
        for _ in range(num_swaps):
            if len(new_words) > 1:
                i, j = random.sample(range(len(new_words)), 2)
                new_words[i], new_words[j] = new_words[j], new_words[i]
        return new_words

    def random_deletion(self, words: List[str]) -> List[str]:
        if len(words) < 2: return words
        new_words = words.copy()
        for i in range(len(new_words)):
            if random.random() < self.rd_prob:
                del new_words[i]
                break  # One deletion per augmentation
        return new_words

    def augment_single(self, text: str) -> str:
        """Full augmentation pipeline."""
        if len(text.split()) < 3: return text
        words = word_tokenize(text.lower())

        # Sequential augmentations
        words = self.synonym_replacement(words)
        words = self.random_insertion(words)
        words = self.random_swap(words)
        words = self.random_deletion(words)

        return ' '.join(words)


def auto_find_dataset():
    """Find first non-augmented CSV/TSV in folder."""
    for f in os.listdir('datasets/.'):
        if f.lower().endswith(('.csv', '.tsv')) and 'augmented' not in f.lower():
            return f
    raise FileNotFoundError("No suitable CSV/TSV found!")


if __name__ == "__main__":
    input_file = auto_find_dataset()
    print(f"🔍 Auto-selected: {input_file}")

    # Load
    sep = '\t' if input_file.endswith('.tsv') else ','
    df = pd.read_csv('datasets/' + input_file, sep=sep)

    # Auto-detect text column
    text_col = next((col for col in df.columns if col.lower() in ['text', 'sentence', 'content', 'review']),
                    df.columns[0])
    print(f"📝 Using text column: '{text_col}' | Original rows: {len(df)}")

    # Augment
    augmenter = NLPAugmenter()
    augmented = []

    for _, row in df.iterrows():
        orig_text = row[text_col]
        augmented.append(row.to_dict())  # Original

        for i in range(augmenter.num_augs):
            aug_text = augmenter.augment_single(orig_text)
            aug_row = row.copy()
            aug_row[text_col] = aug_text
            aug_row['is_augmented'] = True
            aug_row['aug_id'] = i
            augmented.append(aug_row)

    # Save
    aug_df = pd.DataFrame(augmented)
    output = input_file.replace('.', '_augmented.')
    aug_df.to_csv(output, index=False)

    print(f"✅ Saved: {output} ({len(aug_df)} rows, {augmenter.num_augs + 1}x expansion!)")

    # Preview
    print("\n📊 Samples:")
    print(aug_df[[text_col, 'is_augmented']].head().to_string(index=False))
