#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# Target function to be minimized:

# In[ ]:


def target(kpis):
    return sum(5*(x_ref - x_ours) if x_ref > x_ours else x_ref - x_ours for x_ref, x_ours in kpis)


# ### kips - criterias for our target function
# 
# format: [(amount_ref, amount_ours), (variance_ref, variance_ours), (assignment_ref, assignment_ours)]
# 
# - amount - total amount at stake (Summing up all the amounts from individual nominators who supported the selected validators) / **higher is better**
# - variance - variance in the stakes of selected validators / **lower is better**
# - assignment - score for assigning nominators to validators (Proposed approach: count the number of validators assigned to each nominator and square it. Then sum all the scores of individual nominators.) / **lower is better**
# 
# 
# #### Solution
# 
# format: 
# 
# <table>
#     <tr>
#         <td></td>
#         <td>validator_1</td>
#         <td>validator_2</td>
#         <td>validator_3</td>
#         <td>amount</td>
#     </tr>
#     <tr>
#         <td>nominator_1</td>
#         <td>0</td>
#         <td>1</td>
#         <td>1</td>
#         <td>2137</td>
#     </tr>
#     <tr>
#         <td>nominator_2</td>
#         <td>1</td>
#         <td>0</td>
#         <td>0</td>
#         <td>420</td>
#     </tr>
#     <tr>
#         <td>nominator_3</td>
#         <td>0</td>
#         <td>0</td>
#         <td>1</td>
#         <td>666</td>
#     </tr>
# </table>
# 
# - 1 if validator is assigned to nominator, 0 otherwise

# In[3]:


# Helper functions

def getAmount(solution):
    return solution["amount"].sum()


def getVariance(solution):
    return solution["amount"].var()

def getAssignment(solution):
    nominatorsAssignments = (solution.iloc[:, :-1] == 1).sum(axis=1)
    return (nominatorsAssignments ** 2).sum()

