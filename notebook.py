# %%
import pandas as pd
import numpy as np
import requests
import os
import shutil
import string

# %%
google_cloud_url = "https://storage.googleapis.com/watcher-csv-exporter/"
session_filename_template = string.Template("polkadot_nominators_session_$id.csv")
era_filename_template = string.Template("polkadot_validators_era_$id.csv")

# %%
"""
Loading of data starting from era number 165 and session 1031
"""

# %%
def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

def download_batch(starting_era=165, starting_session=1031, destinationFolder="data/", number_of_eras_to_download=100):
    if os.path.exists(destinationFolder):
         shutil.rmtree(destinationFolder)
         
    os.makedirs(destinationFolder, exist_ok=True)
    era_id = starting_era
    session_id = starting_session
    total_eras_downloaded = 0

    while total_eras_downloaded < number_of_eras_to_download:
        era_filename = era_filename_template.substitute({'id': era_id})

        try:
            download_file(
                google_cloud_url + era_filename,
                destinationFolder + era_filename
            )
        except Exception as e:
            era_id += 1
            session_id += 6
            continue
        
        session_filename = session_filename_template.substitute({'id': session_id})
        download_file(
            google_cloud_url + session_filename,
            destinationFolder + session_filename
        )

        total_eras_downloaded += 1
        era_id += 1
        session_id += 6

download_batch()

# %%
nominators = pd.read_csv("data/polkadot_nominators_session_1031.csv")
validators = pd.read_csv("data/polkadot_validators_era_165.csv")

# %%
nominators

# %%
number_of_validators = len(validators)
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