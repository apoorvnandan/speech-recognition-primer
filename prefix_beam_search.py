import re
import numpy as np


class Cache:
    '''
    class to cache the probability values while performing prefix beam search
    '''
    def __init__(self):
        self.data = {}
    
    def add(self, key1, key2, value):
        if key1 not in self.data:
            self.data[key1] = {}
        self.data[key1][key2] = value
        
    def get(self, key1, key2):
        if key1 not in self.data:
            return 0
        if key2 not in self.data[key1]:
            return 0
        return self.data[key1][key2]
    
    # def get_prefix_list(timestep):
    #     if timestep not in self.data:
    #         return []
    #     return self.data[timestep]
    
def get_num_words(txt):
    '''
    function to count words in a text. The seperator can be space or end_token
    :param txt: input string
    :return: (int) number of words
    '''
    words = re.findall(r'\w+[\s|>]', txt)
    return len(words) + 1

def get_k_most_probable_prefixes(Pb, Pnb, timestep, k, beta):
    '''
    function to get k most probable prefixes from the blank and non blank probabilties
    cached for a particular timestep
    :param Pb: Cache for blank probabilities at each timestep
    :param Pnb: Cache for non blank probabilities at each timestep
    :param timestep:
    :param k: threshold for returning the k most probable prefixes
    :param W: function to split string into list of word tokens
    :param beta: language model compensation (comes from prefix search params)
    :return: list of k most probable prefixes
    '''
    prefix_list = []
    prob_map = {}
    if timestep in Pb.data:    
        for prefix in list(Pb.data[timestep].keys()):
            prefix_list.append(prefix)
            prob = Pb.get(timestep, prefix) + Pnb.get(timestep, prefix)
            prob_map[prefix] = prob * get_num_words(prefix) ** beta
    if timestep in Pnb.data:
        for prefix in list(Pnb.data[timestep].keys()):
            if prefix in prefix_list:
                continue
            prefix_list.append(prefix)
            prob = Pb.get(timestep, prefix) + Pnb.get(timestep, prefix)
            prob_map[prefix] = prob * get_num_words(prefix) ** beta
    prefix_list = sorted(prefix_list, key=lambda l: prob_map[l], reverse=True)
    return prefix_list[:k]


def prefix_beam_search(ctc,
                       alphabet,
                       blank_token,
                       end_token,
                       space_token,
                       lm,
                       k=25,
                       alpha=0.30,
                       beta=5,
                       prune=0.001):
    '''
    function to perform prefix beam search on output ctc matrix and return the best string
    :param ctc: output matrix
    :param alphabet: list of strings in the order their probabilties are present in ctc output
    :param blank_token: string representing blank token
    :param end_token: string representing end token
    :param space_token: string representing space token
    :param lm: function to calculate language model probability of given string
    :param k: threshold for selecting the k best prefixes at each timestep
    :param alpha: language model weight (b/w 0 and 1)
    :param beta: language model compensation (should be proportional to alpha)
    :param pruning threshold: threshold on the output matrix probability of a character. 
        If the probability of a character is less than this threshold, we do not extend the prefix with it
    :return: best string
    '''
    zero_pad = np.zeros((ctc.shape[0] + 1, ctc.shape[1]))
    zero_pad[1:, :] = ctc
    ctc = zero_pad
    total_timesteps = ctc.shape[0]

    # #### Initialization ####
    null_token = ''
    Pb, Pnb = Cache(), Cache()
    Pb.add(0, null_token, 1)
    Pnb.add(0, null_token, 0)
    prefix_list = [null_token]

    # #### Iterations ####
    for timestep in range(1, total_timesteps):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[timestep] > prune)[0]]
        for prefix in prefix_list:
            if len(prefix) > 0 and prefix[-1] == end_token:
                Pb.add(timestep, prefix, Pb.get(timestep - 1, prefix))
                Pnb.add(timestep, prefix, Pnb.get(timestep - 1, prefix))
                continue

            for character in pruned_alphabet:
                character_index = alphabet.index(character)

                # #### Iterations : Case A ####
                if character == blank_token:
                    value = Pb.get(timestep, prefix) + ctc[timestep][character_index] * (
                                Pb.get(timestep - 1, prefix) + Pnb.get(timestep - 1, prefix))
                    Pb.add(timestep, prefix, value)
                else:
                    prefix_extended = prefix + character
                    # #### Iterations : Case C ####
                    if len(prefix) > 0 and character == prefix[-1]:
                        value = Pnb.get(timestep, prefix_extended) + ctc[timestep][character_index] * Pb.get(
                            timestep - 1, prefix)
                        Pnb.add(timestep, prefix_extended, value)
                        value = Pnb.get(timestep, prefix) + ctc[timestep][character_index] * Pnb.get(timestep - 1,
                                                                                                     prefix)
                        Pnb.add(timestep, prefix, value)

                    # #### Iterations : Case B ####
                    elif len(prefix.replace(space_token, '')) > 0 and character in (space_token, end_token):
                        lm_prob = lm(prefix_extended.strip(space_token + end_token)) ** alpha
                        value = Pnb.get(timestep, prefix_extended) + lm_prob * ctc[timestep][character_index] * (
                                    Pb.get(timestep - 1, prefix) + Pnb.get(timestep - 1, prefix))
                        Pnb.add(timestep, prefix_extended, value)
                    else:
                        value = Pnb.get(timestep, prefix_extended) + ctc[timestep][character_index] * (
                                    Pb.get(timestep - 1, prefix) + Pnb.get(timestep - 1, prefix))
                        Pnb.add(timestep, prefix_extended, value)

                    if prefix_extended not in prefix_list:
                        value = Pb.get(timestep, prefix_extended) + ctc[timestep][-1] * (
                                    Pb.get(timestep - 1, prefix_extended) + Pnb.get(timestep - 1, prefix_extended))
                        Pb.add(timestep, prefix_extended, value)
                        value = Pnb.get(timestep, prefix_extended) + ctc[timestep][character_index] * Pnb.get(
                            timestep - 1, prefix_extended)
                        Pnb.add(timestep, prefix_extended, value)

        prefix_list = get_k_most_probable_prefixes(Pb, Pnb, timestep, k, beta)

    # #### Output ####
    return prefix_list[0].strip(end_token)
