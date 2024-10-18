# %%
import pandas as pd
import numpy as np

# %%
"""
Setting N (Number of validators in era) fixed for now
"""

# %%
initial_N = 297

# %%
"""
Loading of data from era number 165 and session 1031
"""

# %%
nominators = pd.read_csv("data/polkadot_nominators_session_1031.csv")
validators = pd.read_csv("data/polkadot_validators_era_165.csv")

# %%
nominators

# %%
validators

# %%
target = nominators["targets"][0]
validators[validators["stash_address"] == target]

# %%
"""
Creation of binary table similar to one in presentation, so showing which validators nominator selected and his total money
"""

# %%
binary_matrix = nominators["targets"].str.get_dummies(sep=',')
binary_matrix.index = nominators["stash_address"]
binary_matrix["amount"] = nominators.set_index("stash_address")["bonded_amount"]
binary_matrix

# %%
"""
Creation of table as above, but with number of how many specific validators nominator selected instead of information if validator is present in targets of nominator.
"""

# %%
unique_validators = pd.Series(nominators["targets"].str.split(",").sum()).unique()
count_matrix_np = np.zeros((len(nominators), len(unique_validators)), dtype=int)
validator_to_idx = {validator: i for i, validator in enumerate(unique_validators)}

for i, targets in enumerate(nominators['targets']):
    for validator in targets.split(','):
        count_matrix_np[i, validator_to_idx[validator]] += 1

count_matrix = pd.DataFrame(count_matrix_np, index=nominators['stash_address'], columns=unique_validators)
count_matrix['amount'] = nominators.set_index('stash_address')['bonded_amount']
count_matrix