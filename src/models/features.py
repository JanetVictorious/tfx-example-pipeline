"""TFX Klarna credit model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""

from typing import List

# At least one feature is needed.

# Name of features which have continuous float values. These features will be
# used as their own values.
DENSE_FLOAT_FEATURE_KEYS = ['avg_payment_span_0_12m',
                            'max_paid_inv_0_12m',
                            'max_paid_inv_0_24m',
                            'num_active_div_by_paid_inv_0_12m']

# These features will be bucketized using `tft.bucketize`, and will be used
# as categorical features. Number of buckets used by tf.transform for encoding
# each feature
BUCKET_FEATURE_DICT = {'account_days_in_rem_12_24m': 12,
                       'account_days_in_term_12_24m': 10,
                       'age': 5,
                       'num_arch_ok_0_12m': 4,
                       'num_arch_ok_12_24m': 3,
                       'status_2nd_last_archived_0_24m': 4,
                       'status_3rd_last_archived_0_24m': 3,
                       'status_last_archived_0_24m': 5,
                       'status_max_archived_0_12_months': 5,
                       'status_max_archived_0_24_months': 5,
                       'status_max_archived_0_6_months': 3,
                       'sum_paid_inv_0_12m': 10}

# These features will be labeled 0 or 1 depending on a threshold value
BOOL_FEATURE_KEYS = ['account_amount_added_12_24m',
                     'account_days_in_dc_12_24m',
                     'num_active_inv',
                     'num_arch_dc_0_12m',
                     'num_arch_dc_12_24m',
                     'num_arch_rem_0_12m',
                     'num_unpaid_bills',
                     'recovery_debt',
                     'sum_capital_paid_account_0_12m',
                     'sum_capital_paid_account_12_24m']

# This feature just feels important, let's keep it even though it has
# a high rate of missing values
WORST_STATUS = 'worst_status_active_inv'

# Name of features which have string values and are mapped to integers
# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_FEATURE_DICT = {'merchant_group': 12,
                      'has_paid': 2,
                      'merchant_category': 57,
                      'name_in_email': 8}

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES
# are hashed.
OOV_SIZE = 10

# Label key
LABEL_KEY = 'default'


def transformed_name(key: str) -> str:
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'


def vocabulary_name(key: str) -> str:
    """Generate the name of the vocabulary feature from original name."""
    return key + '_vocab'


def transformed_names(keys: List[str]) -> List[str]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]
