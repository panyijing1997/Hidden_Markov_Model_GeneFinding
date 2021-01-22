import math
import numpy as np


def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


def translate_indices_to_ann(indices):
    mapping = ['N',
               'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
               'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R']
    return ''.join([mapping[i] for i in indices])


def translate_ann_into_indices(ann, genome):

    '''
    :param ann: original sequence of annoations (String)
    :param genome: original sequence of genome (String)
    :return: indices of hidden states
    For translating ann sequence into 43 hidden states


    '''
    N = len(genome)
    hidden_states_indices = []
    i = 0
    while i < N:  #
        if ann[i] == 'N':
            hidden_states_indices.append(0)
            if i + 3 < N:
                if ann[i + 1:i + 4] == 'CCC':
                    # "NCCC" Start-codon in normal genes( mainly 3 types but still some rare start codons)
                    if genome[i + 1:i + 4] == 'ATG':
                        hidden_states_indices.extend([1, 2, 3])
                    elif genome[i + 1:i + 4] == 'GTG':
                        hidden_states_indices.extend([4, 5, 6])
                    elif genome[i + 1:i + 4] == 'TTG':  # What about cases other than TTG
                        hidden_states_indices.extend([7, 8, 9])
                    else:  # very very rare cases
                        hidden_states_indices.extend([10, 11, 12])
                    i += 4
                elif ann[i + 1:i + 4] == 'RRR':
                    # "NRRR" Reversed stop-codon(3) in reversed genes:
                    if genome[i + 1:i + 4] == 'TTA':
                        hidden_states_indices.extend([22, 23, 24])
                    elif genome[i + 1:i + 4] == 'CTA':
                        hidden_states_indices.extend([25, 26, 27])
                    else:  # "CTA"
                        hidden_states_indices.extend([28, 29, 30])
                    i += 4
                else:  # NXXX , just continue to the next
                    i += 1
            else:  # less than 3 annotations to the end
                i += 1
        elif ann[i] == 'C':
            if ann[i + 3] == 'N':
                # "C(i) CC N(i+3)" Stop-codon in normal genes
                if genome[i:i + 3] == 'TAG':
                    hidden_states_indices.extend([13, 14, 15])
                elif genome[i:i + 3] == 'TGA':
                    hidden_states_indices.extend([16, 17, 18])
                else:  # TAA
                    hidden_states_indices.extend([19, 20, 21])
            else:  # trivial case: CCCC
                hidden_states_indices.extend([10, 11, 12])
            i += 3
        else:  # ann[i]="R"
            if ann[i + 3] == 'N':  # RRRN
                # "RRRN" Reversed start-codon in reversed genes(mainly 3 but there are some rare cases):
                if genome[i:i + 3] == 'CAT':
                    hidden_states_indices.extend([34, 35, 36])
                elif genome[i:i + 3] == 'CAC':
                    hidden_states_indices.extend([37, 38, 39])
                elif genome[i:i + 3] == 'CAA':
                    hidden_states_indices.extend([40, 41, 42])
                else:  # (very rare cases
                    hidden_states_indices.extend([0, 0, 0])
            else:  # trivial case RRRR
                hidden_states_indices.extend([31, 32, 33])
            i += 3
    return hidden_states_indices


def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]


def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)


def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))


def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]


def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

