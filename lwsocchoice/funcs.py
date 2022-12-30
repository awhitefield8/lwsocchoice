import random
import numpy as np
import copy
import nums_from_string
import itertools
 

### general helper functions



def borda(positions,ranks,policy):
    '''
    positions = number of options
    ranking = an individuals ranking
    policy = policy chosen, this is a number
    Returns the borda score of a policy for an individual given thier preference orderings
    '''
    id = ranks.index(policy)
    borda_score = positions - id - 1
    return(borda_score)

def plurality(ranks,policy):
    '''
    ranking = an individuals ranking
    policy = policy chosen, this is a number
    Returns one if an individuals top voted state is the same as the policy
    '''
    if ranks[0] == policy:
        return(1)
    else:
        return(0)

### formatting

def printR(x):
    print(round(x,3))

### functions for evaluating decisions

def bordaScore(ord,policy_state):
    '''
    ord = ordering matrix
    state = state determined by winner
    returns the total borda score for a given policy state
    '''    
    p = len(ord[0]) #identify length of orderings
    scores = [borda(positions = p,ranks = i,policy=policy_state) for i in ord] #use borda helper functions
    return(sum(scores))


def pluralityScore(ord,policy_state):
    '''
    ord = ordering matrix
    state = state determined by winner
    returns the total borda score for a given policy state
    '''    
    p = len(ord[0]) #identify length of orderings
    scores = [plurality(ranks = i,policy=policy_state) for i in ord] #use borda helper functions
    return(sum(scores))





### generators for orderings

def zeroCorrelOrdering(N,
                       S=None,
                       M=None,
                       K=None,
                       ALPHA_VAL=1,
                       asym_graph=False):
    """ Each state is independent, there are R = K^M states

    Args:
        N: number of voters
        M: number of attributes 
        K: number of categories for each choice
        S: number of state
    Returns:
        dictionary with following elements:
            perm_freq: frequency of each preference ordering (i.e permuation)
            edge_weights: weights on pairwise election graph
            prob_values: prob_values
            orderings: (if asym_graph=True)
    """
    #generate R: the number of states
    if S is None:
        R = K**M
    else:
        R = S
    
    #generate states
    STATES = [i for i in range(0,R)] 
    
    # generate possible orderings
    perms = list(itertools.permutations(STATES))
    n_perms = len(perms)
        
    # generate dist for each ordering
    prior = tuple([ALPHA_VAL for _ in range(n_perms)])
    prob_values = np.random.dirichlet(alpha=prior, size=1)[0] # probability of each profile
    perm_freq = dict(zip(perms,prob_values)) # frequency of each permutation
    # orderings

    if asym_graph == True:
        # return asy graph - note that this is independent of N
        
        edge_weights = {}
        for i in range(R):
            for j in range(i+1,R):
                edge_id = nameEdgeAtoB(i,j) #edge from i to j
                temp_sum_list = [ prob_values[perm_id] if perms[perm_id].index(i) < perms[perm_id].index(j) else 0 for perm_id in range(n_perms)]
                edge_weights[edge_id] = sum(temp_sum_list) #i) identify IDS for where i > j, sum weights
        return({"perm_freq":perm_freq,"edge_weights": edge_weights, "prob_values":prob_values,"orderings":None })    
    else:
        # return set of orderings
        def draw():
            res = np.random.multinomial(1, prob_values, size=None)
            order_id = np.where(res == 1)[0][0]
            return(perms[order_id])
        return({"perm_freq":perm_freq,"edge_weights": None, "prob_values":prob_values,"orderings":[draw() for i in range(0,N)] })



def partialCorrelOrdering(N,M,K,rho,weights=None):
    ''' correlation across states. Partial correlation means that given on choice for
    the first attribute (i.e. left  right),
    the other choices will be correlated with this first choice, but not perfectly
    
    Args
        N: number of voters
        M: number of attributes
        K: number of categories for each choice
        Rho: correlation across axis: 1 = perfectly correlated, 0 = not correlated at all
        weights: weight array for each matrix

    Returns:
        list of lists, each containing the preferences of a voter
    '''
    #calcs 
    if weights == None:
        weights = [1]*M
    R = K**M
    RANKINGS = [i for i in range(0,R)]
    LEANINGS = [i for i in range(0,K)]
    #generate a mapping between the R states/policies with which axis and category they represent
    #we can do this with a tensor
    t = np.arange(R).reshape([K]*M) #we can represent preferences in a tensor. Each element is as policy position
    temp_ordering = list() #initialise list where we save down preference of each pereson
    #next loop through each person and generate orderings
    for i in range(0,N):
        LEANING = random.sample(LEANINGS,K)
        result = dict(zip(RANKINGS, [0]*R))
        for k in result.keys():
            loc = np.where(t==k) #obtains indices of ts location
            temp_sum = np.random.uniform(0,0.01) #to randomly break ties in the sort
            for j in loc: #note that each j corresponds to an attribute (the cateogory in each attribute determines the location)
                s = weights[int(j)]*borda(positions=K,ranks=LEANING,policy = int(j)) #borda score
                rand = np.random.uniform(0,1)
                if rand < rho: #rho % of the time, include the score
                    temp_sum = temp_sum + s
            result[k] = temp_sum
        res_ranking = sorted(result, key=result.get, reverse=True)
        res_ranking = [int(i) for i in res_ranking]
        temp_ordering.append(res_ranking)
    return(temp_ordering)

        

### functions for graph based methods


def nameEdgeAtoB(A,B):
    return( "_" + str(A) + "_"+ "_" + str(B) + "_")

def genGraphfromOrd(orderings):
    """
    generate graph from orderings
    """
    S_number = len(orderings[0])
    # generate graph weights from ordering
    edges_weights = {}
    for i in range(S_number):
        for j in range(i+1,S_number):
            edge_id = nameEdgeAtoB(i,j) #edge from i to j
            edges_weights[edge_id] = prop_pref_AtoB(i,j,orderings)
    return(edges_weights)
    

def prop_pref_AtoB(A,B,ORDERINGS):
    """
    given orderings, calculate the proportion that prefer a to b
    """
    sum_pref_A = 0
    for ORDERING in ORDERINGS:
        A_rank = ORDERING.index(A)
        B_rank = ORDERING.index(B)
        if A_rank < B_rank:
            sum_pref_A = sum_pref_A + 1
    prop_pref_A = sum_pref_A/len(ORDERINGS)
    return(prop_pref_A)




def symBordaScoreFn(state,edge_weights):
    """ calculate symeetric borda scores for state, given edge weights
    Args:
        state: a state (int in range 0 to |S|)
        edge_weights: a dictionary of edge weights
    """
    edge_weights_complete = copy.deepcopy(edge_weights)
    for k,v in edge_weights.items():
        k_numbers = nums_from_string.get_nums(k)
        new_edge_name = nameEdgeAtoB(k_numbers[1],k_numbers[0])
        edge_weights_complete[new_edge_name] = 1 - v
    
    # back out list of states
    not_flat_list = [nums_from_string.get_nums(k) for k in edge_weights.keys()]
    flat_list = [x for xs in not_flat_list for x in xs]
    S = list(set(flat_list))
    
    symScore = 0
    for other_state in S:
        if int(state) == int(other_state):
            pass #no need to compare to itself
        else: 
            #count prop of people that prefer s to state
            edge_id = nameEdgeAtoB(state,other_state) 
            symScore = symScore + edge_weights_complete[edge_id] - 0.5 
    return(symScore)



### math hrelper funcs

def base_convert(base_10_number,base_new):
    """
    converts base 10 number to number is desired base  (only works up to base 10)
    Input:
        base_10_number = Number in base 10
    Output:
        number in base_new
    """

    if base_new > 10:
        return("error, not yet compatible with base > 10")
    n = base_10_number
    s = ''
    if n < base_new:
        return(str(n))
    while n != 0:
        s = str(n%(base_new)) + s
        n = n//(base_new)
    return(s)


def add_zeros(number,desired_length):
    diff = desired_length - len(str(number)) 
    if diff < 0:
        return("error")
    elif diff == 0:
        return(str(number))
    else:
        output_number= str(number)
        for i in range(diff):
            output_number = '0' + output_number
        return(output_number)

def genStatesDesc(S_number,c,a):
    states = ['']*S_number #allocate memory
    for i in range(S_number):
        state_desc = base_convert(i,c)
        states[i] = add_zeros(state_desc,a)
    return(states)
    