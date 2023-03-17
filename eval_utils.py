# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
from data_utils import sentword2opinion, sentiment_word_list, laptop_acos_aspect_cate_list, res_acos_aspect_cate_list

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def extract_spans_para(task, seq, seq_type, num_id):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
   
    for s in sents:
        # food quality is bad because pizza is over cooked.
        try:
            ac_sp, at_ot = s.split(' because ')
            ac, sp = ac_sp.split(' is ')
            at, ot = at_ot.split(' is ')

            # if the aspect term is implicit
            if at.lower() == 'it':
                at = 'NULL'
        except ValueError:
            try:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
            ac, at, sp, ot = '', '', '', ''

        quads.append((ac, at, sp, ot))
    
    return quads

def extract_spans_extraction(seq, y_type, num_id, use_sent_flag, use_prompt_flag):
    extractions = []
    global at, ac, sp, ot
    all_pt = seq.split(' [SSEP] ')
    # print("sent flag is ", use_sent_flag, "prompt flag is ", use_prompt_flag)
    for pt in all_pt:
        # print("pt is",pt)
        if use_sent_flag:
            # 去除SI的部分
            if use_prompt_flag:
                SI_tokens = "opinion, "
                pt = pt[pt.find(SI_tokens)+len(SI_tokens):]
            try:
                # EA EI的情况
                ac_sp, at_ot = pt.split(' because ')
                ac, sp = ac_sp.split(' is ')
                at, ot = at_ot.split(' is ')
                # OA EI的情况
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                try:
                    # EA IO的情况
                    # SERVICE#GENERAL is negative because of the staff
                    ac_sp, at_ot = pt.split(' because ')
                    ac, sp = ac_sp.split(' is ')
                    ot, at = at_ot.split(' the ')
                    if ot.lower() == 'of':
                        ot = 'NULL'
                except ValueError:
                    try:
                        # IA IO的情况
                        # SERVICE#GENERAL is negative taking everything into account
                        ac_sp, at_ot = pt.split(' taking ')
                        ac, sp = ac_sp.split(' is ')
                        if at_ot.lower() == 'everything into account':
                            ot = 'NULL'
                            at = 'NULL'
                    except ValueError:
                        ac, at, sp, ot = '', '', '', ''
                # A C S O
            extractions.append((at, ac, sp, ot))
            # print("y_type is【", y_type, "】  ", num_id, "sent is:", pt, "****", "acso:",at, ac, sp, ot)
            # print(extractions)
        else:
            # if y_type == "gold":
            #     print(pt)
            try:
                ac, at, sp, ot = pt.split(', ')
            except ValueError:
                ac, at, sp, ot = '', '', '', ''
            extractions.append((ac, at, sp, ot))
        return extractions

def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents, use_sent_flag, use_prompt_flag):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_extraction(gold_seqs[i], 'gold', i, use_sent_flag, use_prompt_flag)
        pred_list = extract_spans_extraction(pred_seqs[i], 'pred', i, use_sent_flag, use_prompt_flag)

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    print("*-"*40)
    scores = compute_f1_scores(all_preds, all_labels)
    for i in zip(all_preds, all_labels):
        print("labels is :", i[1], "predictions is :", i[0])
    print(scores)

    return scores, all_labels, all_preds
