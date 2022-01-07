import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import json
import pickle
import itertools

'''
Plots :
    scores in each partition : histograms, overlapping/transparent (/ algorithm*k)
    decisions in each partition : stacked bar plot ( / k)
    KS : p and effect size for each algo (/)
    similarity : similarity for each algo (/)
'''


def get_outcomes(final_json, paper_idx):
    outcome_map = {}
    score_map = {}
    for elt in final_json:
        key = elt['url'].split('=')[-1] + '.txt'
        if key in paper_idx:
            idx = paper_idx[key]
            outcome_map[idx] = elt['decision']
            score_map[idx] = float(elt['rating']) if elt['rating'] != ' N/A' else np.nan
    return outcome_map, score_map


def get_outcome_counts(partitions, outcome_map):
    oral_counts = [sum(['Accept (Oral)' == outcome_map[p] for (_, p) in partition]) for partition in partitions]
    poster_counts = [sum(['Accept (Poster)' == outcome_map[p] for (_, p) in partition]) for partition in partitions]
    workshop_counts = [sum(['Invite to Workshop Track' == outcome_map[p] for (_, p) in partition]) for partition in partitions]
    reject_counts = [sum(['Reject' == outcome_map[p] for (_, p) in partition]) for partition in partitions]
    counts = zip(oral_counts, poster_counts, workshop_counts, reject_counts)
    #counts = { 'Oral' : oral_counts, 'Poster' : poster_counts, 'Workshop' : workshop_counts, 'Reject' : reject_counts } 
    return counts


def get_score_split(partitions, score_map):
    return [[score_map[p] for (_, p) in part if not np.isnan(score_map[p])] for part in partitions]


def evaluate_split_fairness(partitions, outcome_map):
    return
    counts = get_outcome_counts(partitions, outcome_map)
    accept_counts = [o + p for (o, p) in zip(counts['Oral'], counts['Poster'])]

    # 333 accepts, 23 orals 
    total_accepts = sum(accept_counts)

    accepts_per_group = np.mean(accept_counts)
    sp_accepts = sum([min(accepts_per_group, accepts) for accepts in accept_counts])
    sp_accepts_missed_frac = 1 - (sp_accepts / total_accepts)

    #sp_accepts_missed_frac = abs(accepts_v1 - accepts_v2) / (2 * total_accepts) # % of true accepts missed by SP assignment
    #sp_oral = abs(oral_v1 - oral_v2) / (2 * total_oral)
    return sp_accepts_missed_frac


def ks_test(partitions, score_map):
    scores = [[score_map[p] for (_, p) in part] for part in partitions]
    ps = []
    Ds = []
    for score1, score2 in itertools.combinations(scores, 2):
        result = scipy.stats.ks_2samp(score1, score2, mode='exact')
        p = result.pvalue # % that these partitions were randomly split
        D = result.statistic # unnormalized cdf difference
        ps.append(p)
        Ds.append(D)
    return min(ps), max(Ds)

if __name__ == '__main__':
    scores = np.load("data/iclr2018.npz", allow_pickle = True)
    paper_idx = scores["paper_idx"][()]
    with open('data/iclr2018.json', 'r') as f:
        final_json = json.loads(f.read())
    outcome_map, score_map = get_outcomes(final_json, paper_idx)
    with open('saved/iclr2018_opt.pkl', 'rb') as f:
        opts = pickle.load(f)

    all_results = {}
    ks = [1, 2, 3]
    for k in ks:
        with open(f'saved/iclr2018_k{k}.pkl', 'rb') as f:
            all_results[k] = pickle.load(f)
    alg_labels = {'multi' : 'multi-partition', 'k1' : 'cycle splitting', 'color' : 'coloring', 'random' : 'random'}
    algo_order = ['random', 'k1', 'color', 'multi']

    # plot similarities
    sim_data = {}
    for k, k_results in all_results.items():
        for algo, algo_results in k_results.items():
            sims = [s / opts[k] for (_, s) in algo_results]
            if algo not in sim_data:
                sim_data[algo] = {'mean' : [None]*len(ks), 'err' : [None]*len(ks)}
            i = ks.index(k)
            sim_data[algo]['mean'][i] = np.mean(sims)
            if len(sims) > 1:
                sim_data[algo]['err'][i] = scipy.stats.sem(sims)

    x = ks
    args = dict(markersize=14, marker='_', linestyle='')
    for algo in algo_order:
        data = sim_data[algo]
        if all([e == None for e in data['err']]):
            data['err'] = None
        print(algo, data['mean'])
        plt.errorbar(x, data['mean'], yerr=data['err'], label=alg_labels[algo], **args)
    plt.legend()
    #plt.tight_layout()
    plt.ylim(bottom=-0.05, top=1.05)
    plt.xticks(ticks=x)
    plt.xlabel('k')
    plt.ylabel('Fraction of optimal non-SP similarity, ICLR')
    plt.savefig('similarity_iclr.png')
    #plt.show()
    plt.close()

    # KS table
    print('k\talgo\tp\tD')
    for k in ks:
        for algo in algo_order:
            ks_results = [ks_test(partition, score_map)  for (partition, _) in all_results[k][algo]]
            p, D = (np.mean(x) for x in zip(*ks_results))
            print(f'{k}\t{algo}\t{p:.4f}\t{D:.4f}')

    '''
    ks_data = {}
    for k, k_results in all_results.items():
        for algo, algo_results in k_results.items():
            ks_results = [ks_test(partition, score_map)  for (partition, _) in algo_results]
            ps, Ds = zip(*ks_results)

            if algo not in ks_data:
                ks_data[algo] = {'mean' : [None]*len(ks)*2, 'err' : [None]*len(ks)*2}
            i = ks.index(k)
            ks_data[algo]['mean'][2*i] = np.mean(ps)
            ks_data[algo]['mean'][2*i + 1] = np.mean(Ds)
            if len(ps) > 1:
                ks_data[algo]['err'][2*i] = scipy.stats.sem(ps)
                ks_data[algo]['err'][2*i + 1] = scipy.stats.sem(Ds)

    x = [k+delt for k in ks for delt in [-0.2, 0.2]]
    args = dict(markersize=14, marker='_', linestyle='')
    for algo, data in ks_data.items():
        if all([e == None for e in data['err']]):
            data['err'] = None
        plt.errorbar(x, data['mean'], yerr=data['err'], label=alg_labels[algo], **args)
    plt.legend()
    #plt.tight_layout()
    plt.ylim(bottom=-0.05, top=1.05)
    plt.xticks(ticks=x, labels=[s for k in ks for s in [f'k={k}, p', f'k={k}, D']])
    plt.ylabel('K-S test p and D')
    plt.savefig('ks.png')
    #plt.show()
    plt.close()
    '''

    # plot scores in each partition 
    for k, k_results in all_results.items():
        for algo, algo_results in k_results.items():
            if algo == 'random':
                continue
            partition = algo_results[0][0]
            score_partition = get_score_split(partition, score_map)
            for part in score_partition:
                plt.hist(part, bins=20, alpha=0.5)
            plt.xlabel('Score')
            plt.title(f'Scores by partition: {alg_labels[algo]}, k={k}')
            plt.savefig(f'scores_{algo}_k{k}.png')
            #plt.show()
            plt.close()

    # plot decisions in each partition 
    for k, k_results in all_results.items():
        base = np.arange(len(k_results))
        i = 0
        x = []
        vals = []
        widths = []
        labs = []
        for algo in algo_order:
            if algo == 'random':
                continue
            algo_results = k_results[algo]
            partition = algo_results[0][0]
            outcome_counts = get_outcome_counts(partition, outcome_map)
            width = 0.8 / len(partition)
            for j, v in enumerate(outcome_counts):
                pos = base[i] + (j * width) - ((width * (len(partition) - 1)) / 2)
                x.append(pos)
                widths.append(width)
                vals.append(v)
            labs.append(alg_labels[algo])
            i += 1
        o, p, w, r = (np.array(v) for v in zip(*vals))
        plt.bar(x, o, widths, label='Oral')
        plt.bar(x, p, widths, bottom=o, label='Poster')
        plt.bar(x, w, widths, bottom=p+o, label='Workshop')
        plt.bar(x, r, widths, bottom=w+p+o, label='Reject')

        plt.legend()
        plt.xticks(ticks=base, labels=labs)
        plt.title(f'Outcomes by partition: k={k}')
        plt.savefig(f'outcomes_k{k}.png')
        #plt.show()
        plt.close()

