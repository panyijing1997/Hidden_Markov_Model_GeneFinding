from utils import *


def compute_w_log(model, x):
    '''

    :param model: trained hmm model
    :param x: observations (String)
    :return: table w (K*N)
    '''
    K = len(model.init_probs)
    xt = translate_observations_to_indices(x)
    N = len(xt)
    w = make_table(K, N)

    # Base case: fill out w[i][0] for i = 0..k-1
    for i in range(K):
        w[i][0] = log(model.init_probs[i]) + log(model.emission_probs[0][xt[i]])


    for n in range(1, N):

        for k in range(K):
            compute_k_n = []
            for j in range(K):
                compute_k_n.append(w[j][n - 1] + log(model.trans_probs[j][k]))
            w[k][n] = max(compute_k_n) + log(model.emission_probs[k][xt[n]])
        if n % 10000 == 0:
            print(str(n)+"processed")
    return w


def opt_path_prob_log(w):
    w = np.array(w)
    return max(w.T[len(w.T) - 1])
    # max value of the last column of w


def backtrack_log(model, x, w):
    '''

    :param model:
    :param x: observations (String)
    :param w: computed table w (a list at size K*N)
    :return: the list of indices(1~K) of all hidden states
    '''
    xt = translate_observations_to_indices(x)
    w = np.array(w)
    N = w.shape[1]
    path_indices = []
    z_N = np.argmax(w[:, N - 1])  # z_N*
    path_indices.append(z_N)
    for n in range(N - 2, -1, -1):  # compute z_N-1* to z_1*
        list_k = []
        for k in range(len(w)):
            list_k.append(log(model.emission_probs[path_indices[len(path_indices) - 1]][xt[n + 1]]) + log(
                model.trans_probs[k][path_indices[len(path_indices) - 1]]) + w[k][n])
        path_indices.append(np.argmax(np.array(list_k)))
    path_indices.reverse()
    # print("backtrack result length", len(path))
    return path_indices


def get_ann(model, x):
    '''

    :param model: trained hmm model
    :param x: observations
    :return: annotations(String) of input observations based on the input model
    '''
    w = compute_w_log(model, x)
    z_indices = backtrack_log(model, x, w)
    z = translate_indices_to_ann(z_indices)
    return z
