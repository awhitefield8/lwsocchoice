### class representing a set of voters - specifically for graph based voting rules
import math
import numpy as np
from gensocchoice.funcs import *



# V object
class V(object):
    """ A class of the stationary distribution of preferences
    """
    def __init__(self,n_states,edge_weights,prob_values):
        """ Default constructor with dummy initialization values.
        Args,
            n_states: number of alturnatives
            edge_weights: 
            prob_values: 
        
         """
        
        self.n_states = n_states
        n_edges = math.comb(n_states,2)
        self.n_edges = n_edges
        self.edge_weights = edge_weights 
        self.prob_values = prob_values #parameter realisation for each possible ordering



class M(object):
    """ A class representing pairwise election graph approximation
    
    Attributes:
        n_states (int): number of states
        n_voters (int): number of voters (who we can poll - in the model there are infinite voters)

    """
    
    def __init__(self,n_states,n_voters,V_INST):
        """ Default constructor with dummy initialization values.
        Args,
            n_states: 
            n_voters:
            V_inst: 
        
         """
        self.n_states = n_states
        n_edges = math.comb(n_states,2)
        self.n_edges = n_edges
        self.n_voters = n_voters
        self.V_inst = V_INST

        self.edge_weights_est = None    
        self.edge_strat = None
        self.MVdist = None

        self.votingRule = None
        self.est_winner = None
        self.true_winner = None
        self.exPostLoss = None

        self.exAnteSocialLoss = None
        self.pollingScheme = None
    
    def __str__(self):
        """ Return string """
        #for k, v in self.edge_weights_est.items():
        #    print("edge: " + str(k) + 
        #    "; true weight: " +  str(round(self.V_inst.edge_weights[k],3)) +
        #    "; est weight: " + str(round(v,3)))
        if self.MVdist is None:
            return "states %d ; voter count %d"  % (self.n_states, self.n_voters)
        else:
            return "states %d ; voter count %d ; MV distance %f" % (self.n_states, self.n_voters,round(self.MVdist,3))

    
    def calcExAnteSocialLoss(self,votingRule,sampling_scheme,iters=100,repBureau_size=10):
        loss_list = [-1]*iters
        for i in range(iters):
            if sampling_scheme == "updateGraphRS":
                self.updateGraphRS() #approximate graph
            elif sampling_scheme == "updateGraphFixed":
                self.updateGraphFixed()
            elif sampling_scheme == "updateGraphRepBureau":
                self.updateGraphRepBureau(repBureau_size)
            else:
                print("error - invalid graph updating scheme")
            self.applyVotingRule(votingRule) #apply rule
            loss_list[i] =  copy.deepcopy(self.exPostLoss)
        self.exAnteSocialLoss = np.mean(loss_list)

    def socialLossStats(self):
        loss_dict = {"voting_rule": self.votingRule.name,
                     "ex ante social loss":self.exAnteSocialLoss,
                     "polling scheme":self.pollingScheme,
                     "true_winner":self.true_winner,
                     "previous est_winner":self.est_winner,
                     "previous exPostLoss":self.exPostLoss}
        return(loss_dict)
    
    def updateGraphRS(self):
        """ Update graph using uniform sampling
        """
        ### update edge given voter preference
        self.edge_strat = [self.n_voters // self.n_edges + (1 if x < self.n_voters % self.n_edges else 0)  for x in range (self.n_edges)]
        
        # calculate edge weights
        counter = 0
        edge_weights_est = {}
        for i in range(self.n_states):
            for j in range(i+1,self.n_states):
                edge_id = nameEdgeAtoB(i,j)  #edge from i to j
                strata_size = self.edge_strat[counter]
                if strata_size > 0:
                    pairwise_result = np.random.binomial(strata_size,self.V_inst.edge_weights[edge_id]) #equivalent to iterating through each voter, simulating preferences, and seeing who wins
                    edge_weights_est[edge_id] = pairwise_result/strata_size # this is unbiased, but some variation  
                else:
                    print("Warning - strata size is 0, setting edge weight to prior point estimate (0.5)")
                    edge_weights_est[edge_id] = 0.5
                counter = counter + 1
        self.edge_weights_est = edge_weights_est #update edge weights
        
        # then update distance
        m_array = np.array([v for v in self.edge_weights_est.values() ] ) 
        v_array = np.array([v for v in self.V_inst.edge_weights.values() ] ) 
        self.MVdist = np.linalg.norm(m_array- v_array,ord=1) / self.n_edges #average distance
        self.pollingScheme = "updateGraphRS"

    
    def updateGraphFixed(self,states_polled=2):
        """ Update graph where all voters are shown the same choice
        for now, only allow 1 decision to be made by voters
        """
        max_states_ = states_polled #later endogenise this to function of effort (e)
        S = [i for i in range(self.n_states)]
        sampled_states = random.sample(S,k=max_states_)
        sampled_states.sort()
        
        small_id_state = str(sampled_states[0])
        big_id_state = str(sampled_states[1])
        key_edge_id = nameEdgeAtoB(small_id_state,big_id_state)
        
        counter = 0
        edge_weights_est = {}
        for i in range(self.n_states):
            for j in range(i+1,self.n_states):
                edge_id = nameEdgeAtoB(i,j)  #edge from i to j
                strata_size = self.n_voters
                if edge_id == key_edge_id:
                    pairwise_result = np.random.binomial(strata_size,self.V_inst.edge_weights[edge_id]) #equivalent to iterating through each voter, simulating preferences, and seeing who wins
                    edge_weights_est[edge_id] = pairwise_result/strata_size # this is unbiased, but some variation  
                else:
                    edge_weights_est[edge_id] = 0.5
                counter = counter + 1
        
        self.edge_weights_est = edge_weights_est #update edge weights
        
        m_array = np.array([v for v in self.edge_weights_est.values() ] ) 
        v_array = np.array([v for v in self.V_inst.edge_weights.values() ] ) 
        self.MVdist = np.linalg.norm(m_array- v_array,ord=1) / self.n_edges #average distance
        self.pollingScheme = "updateGraphFixed"
        
    def updateGraphRepBureau(self,n_bureau_members):


        #repeat calcs in zeroCorrelOrdering
        STATES = [i for i in range(0,self.n_states)] 
        perms = list(itertools.permutations(STATES)) #will always be the same  
        prob_values = self.V_inst.prob_values
        
        def draw():
            res = np.random.multinomial(1, prob_values, size=None)
            order_id = np.where(res == 1)[0][0]
            return(perms[order_id])
        
        draws = [draw() for i in range(0,n_bureau_members)] #draw n_bureau_members from the population

        ### update edge weights as below

        # calculate edge weights
        counter = 0
        edge_weights_est = {}
        for i in range(self.n_states):
            for j in range(i+1,self.n_states):
                edge_id = nameEdgeAtoB(i,j)  #edge from i to j
                temp_sum_list = [ 1 if draw.index(i) < draw.index(j) else 0 for draw in draws]
                edge_weights_est[edge_id] = sum(temp_sum_list)/n_bureau_members #i) identify IDS for where i > j, sum weights
                counter = counter + 1
        self.edge_weights_est = edge_weights_est #update edge weights
        
        # then update distance
        m_array = np.array([v for v in self.edge_weights_est.values() ] ) 
        v_array = np.array([v for v in self.V_inst.edge_weights.values() ] ) 
        self.MVdist = np.linalg.norm(m_array- v_array,ord=1) / self.n_edges #average distance
        self.pollingScheme = "updateGraphRepBureau"

        
    def applyVotingRule(self,votingRule_inst):
        """ apply voting rule
        Args:
            votingRule: instance of voting rule object
        Updates:
            self.votingRule
            self.winner
            self.exPostLoss
        """

        self.votingRule = votingRule_inst
        self.est_winner = votingRule_inst.getWinner(edge_weights=self.edge_weights_est) #call in voting object, pick winner
        self.true_winner = votingRule_inst.getWinner(edge_weights=self.V_inst.edge_weights) #call in voting object, pick winner
        self.exPostLoss = abs(
            votingRule_inst.scoreState(state=self.est_winner,edge_weights=self.V_inst.edge_weights) -
            votingRule_inst.scoreState(state=self.true_winner,edge_weights=self.V_inst.edge_weights)
            )



# Voting rule object
class VotingRule(object):
    """ A class of the stationary distribution of preferences
    """
    def __init__(self,name,scoreFn):
        """ Default constructor with dummy initialization values. """
        self.name = name
        self.scoreFn = scoreFn

    def __str__(self):
        """ Return string """
        return "voting rule: " + self.name

    def getWinner(self,edge_weights):
        """
        takes in edge_weights and, returns a winner
        """
        # extract all states - find the state with the highest score
        not_flat_list = [nums_from_string.get_nums(k) for k in edge_weights.keys()]
        flat_list = [x for xs in not_flat_list for x in xs]
        S = list(set(flat_list))

        S_scores = {}
        for s in S:
            S_scores[s] = self.scoreState(s,edge_weights)
        winner = max(S_scores, key=S_scores.get)
        return(winner)
    
    def scoreState(self,state,edge_weights):
        """
        takes in edge_weights and a state, returns the score of the state
        """
        score = self.scoreFn(state,edge_weights) 
        return(score)