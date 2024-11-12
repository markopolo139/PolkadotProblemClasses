# %%
import pandas as pd
import numpy as np
import requests
import os
import glob
import shutil
import string

# %%
def calc_target(kpis):
    return sum(5*(x_ref - x_ours) if x_ref > x_ours else x_ref - x_ours for x_ref, x_ours in kpis)

# %%
# Helper functions

def getAmount(solution):
    solution2 = solution.loc[~(solution.drop(columns="amount") == 0).all(axis=1)]
    return solution2["amount"].mean() * 1


def getVariance(solution):
    solution_amountless = solution.drop(columns="amount")
    temp = {col: 0 for col in solution_amountless.columns}
    for index, nominator in solution.iterrows():
        nominator_amountless = nominator.drop("amount")
        selected = nominator_amountless[nominator_amountless==1].index.tolist()

        if len(selected) == 0: continue

        amount_to_distribute = nominator["amount"] / len(selected)

        for validator in selected:
            temp[validator] += amount_to_distribute
            
    return np.array(list(temp.values())).std() * -1


def getAssignment(solution):
    nominatorsAssignments = (solution.iloc[:, :-1] == 1).sum(axis=1)
    return ((nominatorsAssignments - 1) ** 2).sum() * -1


def concatenateSolutions(sol1, sol2):
    result = []
    for s1, s2 in zip(sol1, sol2):
        normalizer = max(s1, s2)
        result.append((s1 / normalizer, s2 / normalizer))

    return result


def calculateKpis(sol):
    solution_amount = getAmount(sol)
    solution_variance = getVariance(sol)
    solution_assignment = getAssignment(sol)

    solution_kpis = [solution_amount, solution_variance, solution_assignment]

    return solution_kpis


def compareSolutions(sol1, sol2):
    return calc_target(concatenateSolutions(calculateKpis(sol1), calculateKpis(sol2)))


def getDataBatches(nominators_filepath_pattern='data/polkadot_nominators_session_*.csv', batch_size=1, default_min=0):
    file_paths = glob.glob(nominators_filepath_pattern)
    num_batches = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size != 0 else 0)
    data_batches = []
    for i in range(num_batches):
        nominators_batches = [pd.read_csv(file) for file in file_paths[i * batch_size: (i + 1) * batch_size]]
        all_data = pd.concat(nominators_batches, ignore_index=True)
        min_batch_amount = default_min if default_min == 0 else all_data["bonded_amount"].min()
        max_batch_amount = all_data["bonded_amount"].max()

        nominators_batches = list(map(lambda x: removeEmptyTargetsRow(x, min_batch_amount, max_batch_amount), nominators_batches))
        data_batches.append(nominators_batches)
    return data_batches

def removeEmptyTargetsRow(nominators_df, min_amount, max_amount):
    nominators_no_na = nominators_df[nominators_df["targets"].notna()]
    return normalizeAmountColumn(nominators_no_na, min_amount, max_amount)

def normalizeAmountColumn(nominators_df, min_amount, max_amount):
    nominators_df['bonded_amount'] = (nominators_df['bonded_amount'] - min_amount) / (max_amount - min_amount)
    return nominators_df


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

def download_batch(starting_era=1000, starting_session=6041, destinationFolder="data/", number_of_eras_to_download=1):
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
nominators = getDataBatches()[0][0]
validators = pd.read_csv("data/polkadot_validators_era_1000.csv")

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
binary_matrix_dropped = binary_matrix.drop(columns="amount")
random_solution = binary_matrix_dropped.sample(n=number_of_validators, axis=1)
random_solution

# %%
random_solution['amount'] = nominators.set_index('stash_address')['bonded_amount']
random_solution

# %%
random_solution_amount = getAmount(random_solution)
random_solution_variance = getVariance(random_solution)
random_solution_assignment = getAssignment(random_solution)

random_solution_kpis = [random_solution_amount, random_solution_variance, random_solution_assignment]
random_solution_kpis

# %%
random_solution2 = binary_matrix_dropped.sample(n=number_of_validators, axis=1)
random_solution2['amount'] = nominators.set_index('stash_address')['bonded_amount']
random_solution2_amount = getAmount(random_solution2)
random_solution2_variance = getVariance(random_solution2)
random_solution2_assignment = getAssignment(random_solution2)

random_solution_kpis2 = [random_solution2_amount, random_solution2_variance, random_solution2_assignment]
random_solution_kpis2

# %%
score = calc_target(concatenateSolutions(random_solution_kpis, random_solution_kpis2))
score

# %%
score = calc_target(concatenateSolutions(random_solution_kpis2, random_solution_kpis))
score

# %%
validators["stash_address"]
ref_sol = binary_matrix[validators["stash_address"]]

ref_sol['amount'] = nominators.set_index('stash_address')['bonded_amount']
ref_sol_amount = getAmount(ref_sol)
ref_sol_variance = getVariance(ref_sol)
ref_sol_assignment = getAssignment(ref_sol)


# %%
ref_sol_kpis = [ref_sol_amount, ref_sol_variance, ref_sol_assignment]
score = calc_target(concatenateSolutions(ref_sol_kpis, random_solution_kpis))
score

# %%
ref_sol_kpis = [ref_sol_amount, ref_sol_variance, ref_sol_assignment]
score = calc_target(concatenateSolutions(ref_sol_kpis, random_solution_kpis2))
score

# %%
"""
## Gready Solution
"""

# %%
## Focus On total amount
def solve_total_amount(nominators, num_of_vals):
    binary_matrix = nominators["targets"].str.get_dummies(sep=',')
    binary_matrix.index = nominators["stash_address"]
    binary_matrix["amount"] = nominators.set_index("stash_address")["bonded_amount"]
    selected_validators = set()

    binary_matrix.sort_values(by='amount', inplace=True, ascending=False)
    i = 0

    while len(selected_validators) < num_of_vals:
        cols = binary_matrix.columns[binary_matrix.iloc[i] == 1].tolist()
        to_add = num_of_vals - len(selected_validators)
        if len(cols) <= to_add:
            selected_validators.update(cols)
        else:
            selected_validators.update(cols[:to_add])
        i += 1

    result = binary_matrix[list(selected_validators)]
    result['amount'] = nominators.set_index('stash_address')['bonded_amount']

    return result

# %%
## Focus on amount variance
def solve_variance_only(nominators, num_of_vals):
    nominators_copy = nominators.copy(deep=True)
    nominators_copy['targets'] = nominators_copy['targets'].apply(lambda x: x.split(','))
    expanded_nominators = nominators_copy.explode('targets')

    validator_stakes = expanded_nominators.groupby('targets')['bonded_amount'].sum().reset_index()
    validator_stakes = validator_stakes.rename(columns={'targets': 'validator_id', 'bonded_amount': 'total_stake'})
    validator_stakes_sorted = validator_stakes.sort_values(by='total_stake', ascending=False)

    selected_validators = []

    for _, validator in validator_stakes_sorted.iterrows():
        stakes = [v['total_stake'] for v in selected_validators] + [validator['total_stake']]
        variance = np.var(stakes)

        if len(selected_validators) < num_of_vals or variance < np.var(stakes[:-1]):
            selected_validators.append(validator)

        if len(selected_validators) >= num_of_vals:
            break

    selected_validators_df = pd.DataFrame(selected_validators)

    binary_matrix = nominators["targets"].str.get_dummies(sep=',')
    binary_matrix.index = nominators["stash_address"]
    binary_matrix["amount"] = nominators.set_index("stash_address")["bonded_amount"]

    return binary_matrix[list(selected_validators_df['validator_id']) + ['amount']]

# %%
## Focus on assignments
def solve_assignments_only(nominators, num_of_vals):
    nominators_copy = nominators.copy(deep=True)
    nominators_copy['targets'] = nominators_copy['targets'].apply(lambda x: x.split(','))
    expanded_nominators = nominators_copy.explode('targets')

    validator_assignments = expanded_nominators.groupby('targets')['stash_address'].nunique().reset_index()
    validator_assignments = validator_assignments.rename(columns={'targets': 'validator_id', 'stash_address': 'num_assignments'})
    validator_assignments_sorted = validator_assignments.sort_values(by='num_assignments', ascending=False)

    selected_validators = []

    for _, validator in validator_assignments_sorted.iterrows():
        if len(selected_validators) < num_of_vals:
            selected_validators.append(validator)

        if len(selected_validators) >= num_of_vals:
            break

    selected_validators_df = pd.DataFrame(selected_validators)

    binary_matrix = nominators["targets"].str.get_dummies(sep=',')
    binary_matrix.index = nominators["stash_address"]
    binary_matrix["amount"] = nominators.set_index("stash_address")["bonded_amount"]

    return binary_matrix[list(selected_validators_df['validator_id']) + ['amount']]

# %%
## Focus on our kpis
def solve(nominators, num_of_vals):
    nominators_copy = nominators.copy(deep=True)
    print(nominators_copy[nominators_copy['targets'].apply(lambda x: isinstance(x, float))])

    nominators_copy['targets'] = nominators_copy['targets'].apply(lambda x: x.split(','))
    
    expanded_nominators = nominators_copy.explode('targets')
    expanded_nominators['bonded_amount'] = expanded_nominators['bonded_amount'].astype(float)
    
    validator_stakes = expanded_nominators.groupby('targets')['bonded_amount'].sum().reset_index()
    validator_stakes = validator_stakes.rename(columns={'targets': 'validator_id', 'bonded_amount': 'total_stake'})
    validator_stakes_sorted = validator_stakes.sort_values(by='total_stake', ascending=False)

    selected_validators = []
    selected_nominators = set()

    for _, validator in validator_stakes_sorted.iterrows():
        validator_id = validator['validator_id']
        validator_nominators = set(expanded_nominators[expanded_nominators['targets'] == validator_id]['stash_address'])
        overlap = len(validator_nominators & selected_nominators)
        
        stakes = [v['total_stake'] for v in selected_validators] + [validator['total_stake']]
        variance = np.var(stakes)
        variance_tolerance = 0.009

        if len(selected_validators) < num_of_vals or (overlap < 2 and variance - variance_tolerance < np.var(stakes[:-1])):
            selected_validators.append(validator)
            selected_nominators.update(validator_nominators)
        
        if len(selected_validators) >= num_of_vals:
            break

    selected_validators_df = pd.DataFrame(selected_validators)
    
    binary_matrix = nominators["targets"].str.get_dummies(sep=',')
    binary_matrix.index = nominators["stash_address"]
    binary_matrix["amount"] = nominators.set_index("stash_address")["bonded_amount"]

    return binary_matrix[list(selected_validators_df['validator_id']) + ['amount']]

# %%
gready_amount_solution = solve_total_amount(nominators, number_of_validators)
gready_amount_solution

# %%
calculateKpis(gready_amount_solution)

# %%
compareSolutions(ref_sol, gready_amount_solution)

# %%
gready_variance_solution = solve_variance_only(nominators, number_of_validators)
gready_variance_solution

# %%
calculateKpis(gready_variance_solution)

# %%
compareSolutions(ref_sol, gready_variance_solution)

# %%
gready_assignemts_solution = solve_assignments_only(nominators, number_of_validators)
gready_assignemts_solution

# %%
calculateKpis(gready_assignemts_solution)

# %%
compareSolutions(ref_sol, gready_assignemts_solution)

# %%
gready_solution = solve(nominators, number_of_validators)
gready_solution_kpis = calculateKpis(gready_solution)
print(gready_solution_kpis)
compareSolutions(ref_sol, gready_solution)