# src/utils/extract_features_from_email.py
# import re
# import numpy as np
# import pandas as pd

# # 48 most common words used in Spambase
# word_features = [
#     'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive',
#     'will', 'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your',
#     'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data',
#     '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project',
#     're', 'edu', 'table', 'conference'
# ]

# # 6 special characters used in Spambase
# char_features = [';', '(', '[', '!', '$', '#']

# # Function to count word frequency

# def compute_word_freq(text, word):
#     return (len(re.findall(rf'\\b{re.escape(word)}\\b', text)) / len(text.split())) * 100

# # Function to count char frequency

# def compute_char_freq(text, char):
#     return (text.count(char) / len(text)) * 100 if len(text) > 0 else 0

# # Function to compute capital run lengths

# def compute_capital_features(text):
#     capital_runs = re.findall(r'[A-Z]{2,}', text)
#     lengths = [len(run) for run in capital_runs]
#     if not lengths:
#         return 0, 0, 0
#     return np.mean(lengths), np.max(lengths), sum(lengths)


# def extract_features_from_email(email_text):
#     features = []

#     # Word frequencies (48 features)
#     for word in word_features:
#         features.append(compute_word_freq(email_text, word))

#     # Char frequencies (6 features)
#     for char in char_features:
#         features.append(compute_char_freq(email_text, char))

#     # Capital run lengths (3 features)
#     mean_cap, max_cap, total_cap = compute_capital_features(email_text)
#     features.extend([mean_cap, max_cap, total_cap])

#     # Return as DataFrame
#     columns = [f"word_freq_{w}" for w in word_features] + \
#               [f"char_freq_{c}" for c in char_features] + \
#               ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]

#     return pd.DataFrame([features], columns=columns)

# src/utils/extract_features_from_email.py
import numpy as np
import pandas as pd
import re
import string

# List of 48 most common words in Spambase
SPAMBASE_WORDS = [
    'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail',
    'receive', 'will', 'people', 'report', 'addresses', 'free', 'business', 'email', 'you',
    'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs',
    'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct',
    'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference'
]

# Characters that are used as special symbols in Spambase
SPECIAL_CHARS = [';', '(', '[', '!', '$', '#']


def compute_word_freq(text, word):
    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0
    return (tokens.count(word) / len(tokens)) * 100


def compute_char_freq(text, char):
    if len(text) == 0:
        return 0.0
    return (text.count(char) / len(text)) * 100


def compute_capital_run_lengths(text):
    capital_runs = [len(run) for run in re.findall(r'[A-Z]{2,}', text)]
    if len(capital_runs) == 0:
        return 0.0, 0.0, 0.0
    return (
        sum(capital_runs),  # average capital run length
        float(np.mean(capital_runs)),
        float(np.max(capital_runs))
    )


def extract_features_from_email(text):
    features = []

    # 1–48: Word frequencies
    for word in SPAMBASE_WORDS:
        features.append(compute_word_freq(text, word))

    # 49–54: Special character frequencies
    for char in SPECIAL_CHARS:
        features.append(compute_char_freq(text, char))

    # 55–57: Capital run statistics
    cap_total, cap_avg, cap_max = compute_capital_run_lengths(text)
    features.append(cap_total)
    features.append(cap_avg)
    features.append(cap_max)

    # Convert to DataFrame with correct shape
    return pd.DataFrame([features])
