from collections import defaultdict, Counter, namedtuple
from math import log
import numpy as np
import math
import pprint 

EPSILON = 1e-5

def smoothed_prob(arr, alpha=1):
    '''
    list of probabilities smoothed by Laplace smoothing
    input: arr (list or numpy.ndarray of integers which are counts of any elements)
           alpha (Laplace smoothing parameter. No smoothing if zero)
    output: list of smoothed probabilities

    E.g., smoothed_prob( arr=[0, 1, 3, 1, 0], alpha=1 ) -> [0.1, 0.2, 0.4, 0.2, 0.1]
          smoothed_prob( arr=[1, 2, 3, 4],    alpha=0 ) -> [0.1, 0.2, 0.3, 0.4]
    '''
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    _sum = arr.sum()
    if _sum:
        return ((arr + alpha) / (_sum + arr.size * alpha)).tolist()
    else:
        return ((arr + 1) / arr.size).tolist()

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    # 각 word에 대해 tag들을 list로 갖는 dictionary
    baseline_train={}
    # 각 word에 대해 tag의 빈도수를 저장하는 dictionary
    baseline_cnt={}
    baseline_predict=[]
    total_cnt={}

    
    for line in train :
        for word, tag in line :
                if word not in baseline_train:
                        baseline_train[word]=[tag]
                else:
                        baseline_train[word].append(tag)
    
    for i in baseline_train :
        baseline_cnt[i]=Counter(baseline_train[i])
        total_cnt=Counter(baseline_train[i])

    for line in test :
        line_predict=[]
        for word in line :
                if word in baseline_cnt:
                        max_key=max(baseline_cnt[word], key=baseline_cnt[word].get)
                else:
                     max_key=max(total_cnt, key=total_cnt.get)
                line_predict.append((word,max_key))
        baseline_predict.append(line_predict)
        
    return baseline_predict

"""
dictionary를 만드는데 key는 word고 tag list가 value
word별 tag를 list로 저장
tag 빈도수를 counter로 저장

각 단어에 대해 tag의 빈도수를 저장한 객체
그 중 최대값을 갖는 tag를 반환

value가 max인 key값 반환
"""

Node = namedtuple("Node", ["tag", "prev_node", "log_prob"])
viterbi_train = {}


def viterbi(train, test):
    """
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """

    # 확률 저장 딕셔너리 - 단어 생성 확률, 전이 확률
    for line in train:
        for index, (word, tag) in enumerate(line):
                if tag in viterbi_train:
                     pass
                else:
                        viterbi_train[tag]={"emission": {}, "transition": {}}
                
                #word 없으면 초기화
                if word in viterbi_train[tag]["emission"]:
                     pass
                else:
                        viterbi_train[tag]["emission"][word]=0

                viterbi_train[tag]["emission"][word] +=1
    
                if tag == line[-1][1]:
                     pass
                else:
                     next = line[index + 1][1]
                     if next  in viterbi_train[tag]["transition"]:
                          pass
                     else:
                          viterbi_train[tag]["transition"][next] = 0
                     viterbi_train[tag]["transition"][next] += 1

    for tag in viterbi_train:
        emission_total = sum(viterbi_train[tag]["emission"].values())
        viterbi_train[tag]["emission"]["PHONY"] = EPSILON
        for word in viterbi_train[tag]["emission"]:
            dividend= viterbi_train[tag]["emission"][word] + EPSILON
            divisor=emission_total + EPSILON * len(viterbi_train[tag]["emission"])
            viterbi_train[tag]["emission"][word] = dividend / divisor

        viterbi_train[tag]["transition"]["PHONY"] = EPSILON
        next_total= sum(viterbi_train[tag]["transition"].values())

        for next in viterbi_train[tag]["transition"]:
            dividend=viterbi_train[tag]["transition"][next] + EPSILON
            divisor=next_total + EPSILON * len(viterbi_train[tag]["transition"])
            viterbi_train[tag]["transition"][next] = dividend / divisor

    for line in test:
        if len(line) == False:
            continue
        node = [[Node("START", "START", 1)]]
        for index, word in enumerate(line[1:], 1):
            node.append([])
            for current in viterbi_train:
                most_log_prob = -math.inf
                most_prev = None
                for previous_node in node[index - 1]:
                    transition_probability = viterbi_train[previous_node.tag]["transition"]
                    emission_probability = viterbi_train[current]["emission"]
        
                    current_log_prob = (
                        previous_node.log_prob \
                        + math.log(transition_probability[current if current in transition_probability else "PHONY"]) \
                        + math.log(emission_probability[word if word in emission_probability else "PHONY"])
                    )
                    
                    if current_log_prob > most_log_prob:
                        most_log_prob = current_log_prob
                        most_prev_node = previous_node
                node[index].append(Node(current, most_prev_node, most_log_prob))

        current_node = max(node[-1], key=lambda x: x.log_prob)
        line[-1] = (line[-1], current_node.tag)
        for index, word in reversed(list(enumerate(line[:-1]))):
            line[index] = (word, current_node.prev_node.tag)
            current_node = current_node.prev_node

    return test




