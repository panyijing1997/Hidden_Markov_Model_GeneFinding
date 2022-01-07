from viterbi import *


class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


def count_transitions_and_emissions(K, D, all_x, all_z):
    '''

    :param K: number of types of hidden states
    :param D: number of types of observations
    :param all_x: an int list containing all sequences of observations (indices)
    :param all_z: an int list containing all sequences of hidden states (indices)
    :return: KxK matrix(transition_probs) and a KxD matrix(emission_probs)

    note: Before using this function, sequences of observations should be already translated into indices and
    sequences of annotations should be translated into indices of hidden states
    (so all_x is a list of lists of indices, so is all_z)
    '''

    Phi = np.zeros(shape=(K, D))  # emission probs matrix
    A = np.zeros(shape=(K, K))  # transition probs matrix

    """
    if using pseudo counting, then use np.ones instead of zeros
    """
    counter = 1
    for z in all_z:  # counting transition
        print("counting trasition " + str(counter))
        for i in range(len(z) - 1):
            A[z[i]][z[i + 1]] += 1
        counter += 1

    for i in range(len(all_x)):  # counting emission
        print("counting emission " + str(i + 1))
        for j in range(len(all_z[i]) - 1):
            Phi[all_z[i][j]][all_x[i][j]] += 1

    A /= A.sum(axis=1).reshape((-1, 1))
    Phi /= Phi.sum(axis=1).reshape((-1, 1))
    return A, Phi


def training_by_counting(K, D, all_x, all_z):
    '''

    :param K: number of types of hidden states
    :param D: number of types of observations
    :param all_x: an int list containing all sequences of observations (indices)
    :param all_z: an int list containing all sequences of hidden states (indices)
    :return: Trained K-state-hmm model

    Before using this function, sequences of observations should be already translated into indices and
    sequences of annotations should be translated into indices of hidden states
    (so all_x is a list of lists of indices, so is all_z)
    '''
    # first compute the initial transition probs (pi)
    Pi = np.zeros(shape=(K, 1))
    for z in all_z:
        Pi[z[0]] += 1
    Pi /= np.sum(Pi).reshape((-1, 1))
    A, Phi = count_transitions_and_emissions(K, D, all_x, all_z)
    hmm_trained = hmm(Pi, A, Phi)
    return hmm_trained



def training_for_gene_finding(va):
    '''

    :param va: we dont use genomeva.fa for training (when we use it for validation)
    :return: a trained model (using 4 genome and corresponded 4 annotations)
    '''
    genome_all = []
    ann_all = []
    for i in range(1, 6):
        if i == va:  # dont use genomeva.fa for training when we want to use it for validation
            continue
        genome_file_name = 'genome' + str(i)
        ann_file_name = 'true-ann' + str(i)
        genome = read_fasta_file('data/' + genome_file_name + '.fa')[genome_file_name]
        true_ann = read_fasta_file('data/' + ann_file_name + '.fa')[ann_file_name]
        genome_indices = translate_observations_to_indices(genome)
        hidden_states_indices = translate_ann_into_indices(true_ann, genome)
        genome_all.append(genome_indices)
        ann_all.append(hidden_states_indices)
    model = training_by_counting(43, 4, genome_all, ann_all)
    print("done with reading and translating all files")

    return model


def save_model(model, folder_name):
    np.save(folder_name + '/init_probs.npy', model.init_probs)
    np.save(folder_name + '/trans_probs.npy', model.trans_probs)
    np.save(folder_name + '/emission_probs.npy', model.emission_probs)


def load_model(folder_name):
    init_probs = np.load(folder_name + '/init_probs.npy')
    trans_probs = np.load(folder_name + '/trans_probs.npy')
    emission_probs = np.load(folder_name + '/emission_probs.npy')
    model = hmm(init_probs, trans_probs, emission_probs)
    return model

