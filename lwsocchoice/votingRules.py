
import random
import numpy as np

from lwsocchoice.funcs import *
### exact rules 

#borda winner
def policyBordaWinner(ord):
    '''
    ord = ordering matrix
    '''
    max_index = len(ord[0]) - 1 #max index of states
    results = [bordaScore(ord = ord,policy_state = i) for i in range(0,max_index+1)]
    winning_policy = results.index(max(results))
    return(winning_policy)
    

### exact rules - constrained choice

#B
def runoffBordaWinner(ORD,POLICIES):
    '''
    identifies the Borda winner out of the policies in the runoff 
    '''
    results = [bordaScore(ord = ORD,policy_state = i) for i in POLICIES]
    w = POLICIES[results.index(max(results))]
    return(w)

#identical structure to runoffBordaWinner
def runoffPlurality(ORD,POLICIES):
    '''
    identifies the Plurality winner out of the policies in the runoff 
    '''
    results = [pluralityScore(ord = ORD,policy_state = i) for i in POLICIES]
    w = POLICIES[results.index(max(results))]
    return(w)

### approximation rules

# random dictator
def policyRandomDict(ord):
    '''
    ord = ordering matrix
    '''
    N = len(ord) - 1
    winner_id = random.randint(0, N)
    return(ord[winner_id][0]) #return top policy for winning candidate

# random runoff

def policyRandomRunoff(ord,n_candidates,runoff_rule):
    '''
    Args:
        ord: ordering matrix
        n_candidates: number of candidates
        runoff_rule = rule used to determine the winner

    Picks two policies out of the set of possible policy states
    Everyone votes, majority winner wins
    '''
    p = len(ord[0]) #identify length of orderings i.e. total number of votes
    if n_candidates > p :
        raise("Error: more runoff candidates than number of policy positions") #error message
    runoff_policies = list(np.random.choice(p, n_candidates,replace=False)) #generate two random candidates (from the population)
    #pick winner
    winning_policy = runoff_rule(ORD = ord,POLICIES = runoff_policies) #inputs ordering and runoff policies, and returns the winning policy
    return(winning_policy)

def policyCitizenRunoff(ord,n_candidates,runoff_rule):
    '''
    ord = ordering matrix
    n_candidates = number of candidates

    Picks two random candidates from the population
    differs from policyRandomRunoff as pick two candidates from the population
    Each candidate proposes their preferred state of the world
    Everyone votes, majority winner wins
    '''
    N = len(ord) - 1 
    if n_candidates > N + 1:
        raise("Error: more runoff candidates than people in the population") #error message

    first_choices = [i[0] for i in ord]
    candidate_policy_pool = list(set(first_choices))
    N_CANDIDATES = min(n_candidates,len(candidate_policy_pool))
    runoff_policies = list(np.random.choice(candidate_policy_pool, N_CANDIDATES,replace=False)) #generate two random candidates (from the population)
    #pick winner
    winning_policy = runoff_rule(ORD = ord,POLICIES = runoff_policies) #inputs ordering and runoff policies, and returns the winning policy
    return(winning_policy)

def policyUSRunoff(ord,C):
    '''
    left vs right runoff - pick first axis
    - split candidates to those on 'left' vs 'right'.
    STV with C random candidates on each side
    '''
    pass




### find bilateraly polling method

### constrained method

### welfare loss under different welfare function
