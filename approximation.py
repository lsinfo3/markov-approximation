import numpy as np
from scipy import linalg
import os
import time
import gc

os.chdir(os.path.dirname(__file__) or '.')

#   Input param: Rate matrix Q
#   Fills out diagonal of Q and solves system
#   Returns: State probability vector X
def solveStateEquation (Q):
    np.fill_diagonal(Q, -Q.sum(axis=1))
    b = np.zeros(len(Q))

    Q[:, -1] = 1
    b = np.zeros(len(Q))
    b[-1] = 1

    QT = np.transpose(Q)
    bt = np.transpose(b)

    X = linalg.solve(QT, bt)
    X = np.transpose(X)

    threshold = -np.exp(-15)
    if np.all(X[X < 0] > threshold):
        X = np.abs(X) #for small numerical inaccuracies, negative values can occur here, assert positive state probabilities without too much accuracy lost
    else:
        print("Accuracy lost! Negative values smaller than the threshold occured!")
    
    return X

#   Input params: Rate matrix Q, number of layers, number of states_per_layer, start_layer, goal_layer, approximation method (standard value: sum)
#   Returns: The transition rate from start_layer to goal_layer in the layer system
def findApproxOfLayerTransition (Q, layer, states_per_layer, start_layer, goal_layer, method = sum):
    transition_rates = []

    assert(start_layer < layer and goal_layer < layer)

    #iterate over all state indices in start_layer
    for i in range(start_layer*states_per_layer, start_layer*states_per_layer + states_per_layer):
        #iterate over all state indices in goal_layer
        for j in range(goal_layer*states_per_layer, goal_layer*states_per_layer + states_per_layer):
            transition_rates.append(Q[i,j])

    return method(transition_rates)


# Input params: Number of layers, number of states per layer, functions for transition that map state index to transition rate
# Returns: Rate matrix Q
def createMatrix(layer, states_per_layer, func_lam, func_mu, func_alpha, func_beta):
    n = layer*states_per_layer
    
    Q = np.zeros((n, n))

    for l in range(0, layer):
        for s in range(0, states_per_layer):
            state_index = (l*states_per_layer) + s
            next_layer_neighbor = ((l+1)*states_per_layer) + s
           

            if (l == layer -1):
                # Last layer
                if(s == states_per_layer-1):
                    # Last state
                    continue
                Q[state_index, state_index+1] = func_lam(state_index)
                Q[state_index+1, state_index] = func_mu(state_index+1)
                continue

            if (s == states_per_layer-1):
                # Last state
                Q[state_index, next_layer_neighbor] = func_alpha(state_index)
                Q[next_layer_neighbor, state_index] = func_beta(next_layer_neighbor)
                continue

            # States in the middle with all four transitions
            Q[state_index, state_index+1] = func_lam(state_index)
            Q[state_index+1, state_index] = func_mu(state_index+1)
            Q[state_index, next_layer_neighbor] = func_alpha(state_index)
            Q[next_layer_neighbor, state_index] = func_beta(next_layer_neighbor)

    return Q

#   Input Params: Rate matrix Q, number of layers, number of states per layer
#   Calculates everything and returns results
def approximation (Q, layer, states_per_layer):
    Q1 = Q.copy()
    n = layer * states_per_layer

    gc.collect()

    # Real Solution
    st_original = time.process_time()
    X = solveStateEquation(Q1)
    et_original = time.process_time()

    gc.collect()

    # Approximation

    # Layer system
    st_approx = time.process_time()

    R = np.zeros((layer, layer))

    for i in range(0, layer-1):
        R[i, i+1] = findApproxOfLayerTransition(Q, layer, states_per_layer, i, i+1)
        R[i+1, i] = findApproxOfLayerTransition(Q, layer, states_per_layer, i+1, i)
    
    st1 = time.process_time()
    L = solveStateEquation(R)
    et1 = time.process_time()

    # Models of each layer
    all_layer_systems = [0] * layer
    st2 = [0] * layer
    et2 = [0] * layer
    for l in range(0, layer):
        Q_layer = np.zeros((states_per_layer, states_per_layer))
        seen_states = l*states_per_layer
        for i in range(0, states_per_layer-1):
            Q_layer[i,i+1] = Q[seen_states + i, seen_states + i+1]
            Q_layer[i+1,i] = Q[seen_states + i+1, seen_states + i]
    
        st2[l] = time.process_time()
        X_layer = solveStateEquation(Q_layer)
        et2[l] = time.process_time()

        all_layer_systems[l] = X_layer


    # Combine for approximated state probabilities
    X_approx = [0] * layer
    
    for i in range (0, layer):
        X_approx[i] = all_layer_systems[i] * L[i]
    
    X_approx = [x for xs in X_approx for x in xs]
    
    et_approx = time.process_time()

    gc.collect()


    # Error metrics
    errors_abs = []

    blocking_probability_real = 0
    blocking_probability_approx = 0

    unavailability_prob_real = 0
    unavailability_prob_approx = 0

    # Add up probabilities of all states of layer zero
    for i in range(0, states_per_layer):
        unavailability_prob_real += X[i]
        unavailability_prob_approx += X_approx[i]

    # Calculate MAE and blocking probabilities
    for i in range (0, n):
        error_abs = abs(X_approx[i] - X[i])
        errors_abs.append(error_abs)

        if (i+1)%states_per_layer == 0:
            blocking_probability_real += X[i]
            blocking_probability_approx += X_approx[i]

    MAE = max(errors_abs)

    # Calculate time measurements
    time_original = et_original - st_original
    time_approx = et_approx - st_approx
    time_approx_only_calculating = et1 - st1 
    for i in range(0, len(et2)):
        time_approx_only_calculating += (et2[i] - st2[i])

    # Print results
    print(f'MAE: {MAE}')
    print(f'Unavail. Prob. Real: {unavailability_prob_real}')
    print(f'Unavail. Prob. Approx: {unavailability_prob_approx}')
    print(f'Blocking Prob. Real: {blocking_probability_real}')
    print(f'Blocking Prob. Approx: {blocking_probability_approx}')

    print(f'Time Original: {time_original}')
    print(f'Time Approx.: {time_approx}')
    print(f'Time Approx. Only Calculations: {time_approx_only_calculating}')

    
    # Return results
    # return (X, L, all_layer_systems, X_approx, MAE, unavailability_prob_real, unavailability_prob_approx, blocking_probability_real, blocking_probability_approx, time_original, time_approx, time_approx_only_calculating)


# Scenario implementations
def runScenarioRL1():    
    print('Running Scenario RL1')

    states_per_layer = 500
    layer = 5

    lam = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    m = 100000
    a = 100
    bet = np.logspace(start=0, stop=5, num=20)
    
    for l in lam:
        print(f'lambda: {l}')
        for b in bet:
            print(f'beta: {b}')

            def func_lam(i): 
                return l

            def func_mu(i):
                current_X = i % states_per_layer                # from 0...states_per_layer-1 in each layer
                current_Y = (i - current_X)/states_per_layer    # from 0 ... layer-1
                return min(current_X, current_Y) * m
            
            def func_alpha(i): 
                current_X = i % states_per_layer
                current_Y = (i - current_X)/states_per_layer
                return (layer - current_Y - 1) * a
            
            def func_beta(i):
                current_X = i % states_per_layer
                current_Y = (i - current_X)/states_per_layer
                return current_Y * b

            Q = createMatrix(layer, states_per_layer,func_lam, func_mu, func_alpha, func_beta)
            approximation(Q, layer, states_per_layer)


def runScenarioRL2():    
    print('Running Scenario RL2')

    layer = 5 
    states_per_layer = 500

    lam = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    m = 100000
    a = 100
    bet = np.logspace(start=0, stop=5, num=20)

    for l in lam:
        print(f'lambda: {l}')
        for b in bet:
            print(f'beta: {b}')

            def func_lam(i): 
                return l

            def func_mu(i):
                current_X = i % states_per_layer                # from 0...states_per_layer-1 in each layer
                current_Y = (i - current_X)/states_per_layer    # from 0 ... layer-1
                return min(current_X, current_Y) * m

            def func_alpha(i): 
                current_X = i % states_per_layer
                current_Y = (i - current_X)/states_per_layer
                return (layer - current_Y - 1) * a
            
            def func_beta(i):
                current_X = i % states_per_layer
                return (current_X) * b


            Q = createMatrix(layer, states_per_layer, func_lam, func_mu, func_alpha, func_beta)
            approximation(Q, layer, states_per_layer)
            

runScenarioRL1()
runScenarioRL2()