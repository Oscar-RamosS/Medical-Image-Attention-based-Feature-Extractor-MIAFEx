import random
import copy
import math
import numpy as np
import matplotlib.pyplot as plt


def obj(x):
    """
    The objective function of pressure vessel design
    :param x: position vector
    :return: objective value
    """
    x1, x2, x3, x4 = x
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -math.pi * x3 ** 2 - 4 * math.pi * x3 ** 3 / 3 + 1296000
    g4 = x4 - 240
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3
    else:
        return 1e10


def boundary_check(value, lb, ub):
    """
    Ensure particle's position remains within the bounds
    :param value: particle position
    :param lb: lower bound
    :param ub: upper bound
    :return: bounded position
    """
    for i in range(len(value)):
        value[i] = max(value[i], lb[i])
        value[i] = min(value[i], ub[i])
    return value


def init_position(lb, ub, pop, dim):
    """
    Initialize particle positions randomly within bounds
    :param lb: lower bounds
    :param ub: upper bounds
    :param pop: population size
    :param dim: dimensionality
    :return: positions matrix
    """
    position = []
    for _ in range(pop):
        temp_position = [random.uniform(lb[j], ub[j]) for j in range(dim)]
        position.append(temp_position)
    return position


def init_velocity(vmin, vmax, pop, dim):
    """
    Initialize particle velocities randomly within velocity limits
    :param vmin: minimum velocity
    :param vmax: maximum velocity
    :param pop: population size
    :param dim: dimensionality
    :return: velocity matrix
    """
    velocity = []
    for _ in range(pop):
        temp_velocity = [random.uniform(vmin[j], vmax[j]) for j in range(dim)]
        velocity.append(temp_velocity)
    return velocity


def update_velocity(velocity, position, p_best_position, g_best_position, omega, c1, c2, pop, dim):
    """
    Update the velocity of particles based on personal best and global best
    :param velocity: current velocity
    :param position: current positions
    :param p_best_position: personal best positions
    :param g_best_position: global best position
    :param omega: inertia weight
    :param c1: cognitive coefficient
    :param c2: social coefficient
    :param pop: population size
    :param dim: dimensionality
    :return: updated velocity
    """
    for i in range(pop):
        for j in range(dim):
            r1 = random.random()
            r2 = random.random()
            velocity[i][j] = omega * velocity[i][j] + c1 * r1 * (p_best_position[i][j] - position[i][j]) + c2 * r2 * (
                        g_best_position[j] - position[i][j])
    return velocity


import numpy as np


def jfs(xtrain, ytrain, opts):
    """
    Main CPSO function for feature selection using binary PSO
    :param xtrain: training data (features)
    :param ytrain: training labels
    :param opts: options dictionary containing PSO parameters
    :return: dictionary with the best feature subset and other relevant information
    """
    # Parameters (defaults)
    ub = 1
    lb = 0
    thres = 0.5  # threshold for binary conversion
    w = 0.9  # inertia weight
    c1 = 2  # acceleration factor
    c2 = 2  # acceleration factor

    # Retrieve options from opts dictionary
    N = opts['N']  # Population size
    max_iter = opts['T']  # Number of iterations
    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']

    # Dimension of the feature space
    dim = np.size(xtrain, 1)

    # If lb and ub are scalar, convert to array of size dim
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position (X) and velocity (V)
    X = init_position(lb, ub, N, dim)
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim)

    # Preallocate arrays
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    # PSO iterations
    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness evaluation
        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)

            # Update personal best
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :] = X[i, :]
                fitP[i, 0] = fit[i, 0]

            # Update global best
            if fitP[i, 0] < fitG:
                Xgb[0, :] = Xpb[i, :]
                fitG = fitP[i, 0]

        # Store the best result for this iteration
        curve[0, t] = fitG.copy()
        print(f"Iteration: {t + 1}")
        print(f"Best (PSO): {curve[0, t]}")
        t += 1

        # Update velocity and position
        for i in range(N):
            for d in range(dim):
                # Update velocity
                r1 = rand()
                r2 = rand()
                V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + c2 * r2 * (Xgb[0, d] - X[i, d])

                # Bound velocity
                V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])

                # Update position
                X[i, d] = X[i, d] + V[i, d]

                # Bound position
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

    # Best feature subset selection
    Gbin = binary_conversion(Xgb, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]  # Features selected
    num_feat = len(sel_index)  # Number of selected features

    # Create a dictionary to store results
    pso_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return pso_data


if __name__ == '__main__':
    # Parameter settings
    params = {
        'pop': 50,
        'iter': 100,
        'iter_chaos': 300,
        'c1': 2,
        'c2': 2,
        'omega_min': 0.2,
        'omega_max': 1.2,
        'lbound': [0, 0, 10, 10],
        'ubound': [100, 100, 100, 100],
        'vmin': [-2, -2, -2, -2],
        'vmax': [2, 2, 2, 2]
    }

    print(main(params))
