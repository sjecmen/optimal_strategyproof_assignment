import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import json
import pickle
import itertools

'''
Output plots and data for saved results.
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
    counts = list(zip(oral_counts, poster_counts, workshop_counts, reject_counts))
    return counts


def get_score_split(partitions, score_map):
    return [[score_map[p] for (_, p) in part if not np.isnan(score_map[p])] for part in partitions]



def ks_test(partitions, score_map):
    scores = [[score_map[p] for (_, p) in part] for part in partitions]
    ps = []
    Ds = []
    for score1, score2 in itertools.combinations(scores, 2):
        result = scipy.stats.ks_2samp(score1, score2, mode='exact')
        p = result.pvalue
        D = result.statistic
        ps.append(p)
        Ds.append(D)
    return min(ps), max(Ds)



if __name__ == '__main__':
    scores = np.load("data/iclr2018.npz", allow_pickle = True)
    paper_idx = scores["paper_idx"][()]
    with open('data/iclr2018.json', 'r') as f:
        final_json = json.loads(f.read())
    outcome_map, score_map = get_outcomes(final_json, paper_idx)

    # load saved results
    all_results = {}
    ks = [1, 2, 3]
    for k in ks:
        with open(f'saved/iclr2018_k{k}.pkl', 'rb') as f:
            all_results[k] = pickle.load(f)
        with open(f'saved/iclr2018_k{k}_fix.pkl', 'rb') as f:
            all_results[k]['k1'] = pickle.load(f)['k1']

    opts = {}
    try: # old format
        with open('saved/iclr2018_opt.pkl', 'rb') as f:
            opts = pickle.load(f)
    except:
        for k in ks:
            with open(f'saved/iclr2018_opt_k{k}.pkl', 'rb') as f:
                opts[k] = pickle.load(f)

    alg_labels = {'multi' : 'multi-partition', 'k1' : 'cycle-breaking', 'color' : 'coloring', 'random' : 'random'}
    algo_order = ['random', 'k1', 'color', 'multi']
    markers = {'random': '_', 'k1' : 'x', 'color' : '2', 'multi' : '.'}
    colors = {'random': 'black', 'k1' : 'green', 'color' : 'blue', 'multi' : 'red'}

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
    args = dict(markersize=16, linestyle='')
    plt.rcParams.update({'font.size': 18})
    for algo in algo_order:
        data = sim_data[algo]
        if all([e == None for e in data['err']]):
            data['err'] = None
        print(algo, 1-np.array(data['mean']))
        plt.errorbar(x, 1-np.array(data['mean']), yerr=data['err'], label=alg_labels[algo], color=colors[algo], marker=markers[algo], **args)
    plt.legend()
    plt.tight_layout()
    plt.ylim(bottom=-0.01, top=0.4)
    plt.xticks(ticks=x)
    plt.xlabel('k')
    plt.ylabel('Fraction of optimal similarity lost')
    plt.savefig('plots/similarity_iclr.pdf', bbox_inches="tight")
    plt.show()
    plt.close()

    # KS table
    print('k\talgo\tp\tD')
    for k in ks:
        for algo in algo_order:
            ks_results = [ks_test(partition, score_map)  for (partition, _) in all_results[k][algo]]
            p, D = (np.mean(x) for x in zip(*ks_results))
            print(f'{k}\t{algo}\t{p:.4f}\t{D:.4f}')

    # plot scores in each partition 
    for k, k_results in all_results.items():
        for algo, algo_results in k_results.items():
            if algo == 'random':
                continue
            if algo == 'multi' and k > 1:
                continue
            partition = algo_results[0][0]
            score_partition = get_score_split(partition, score_map)
            colors = ['red', 'blue', 'yellow']
            for i, part in enumerate(score_partition):
                plt.hist(part, bins=20, range=(min(score_map.values()), max(score_map.values())), alpha=0.5, histtype='stepfilled', edgecolor='black', facecolor=colors[i])
            plt.tight_layout()
            plt.xlabel('Mean review score')
            plt.ylabel('Frequency in subset')
            plt.savefig(f'plots/scores_{algo}_k{k}.pdf', bbox_inches="tight")
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
            if algo == 'multi' and k == 1:
                plt.rcParams.update({'font.size': 15})
            else:
                plt.rcParams.update({'font.size': 18})
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
        widths = np.array(widths) - 0.02
        args = dict(edgecolor='black')
        plt.bar(x, o, widths, label='Oral', hatch='', **args)
        plt.bar(x, p, widths, bottom=o, label='Poster', hatch='/', **args)
        plt.bar(x, w, widths, bottom=p+o, label='Workshop', hatch='|', **args)
        plt.bar(x, r, widths, bottom=w+p+o, label='Reject', hatch='\\', **args)

        plt.tight_layout()
        plt.legend()
        plt.xticks(ticks=base, labels=labs)
        plt.ylabel('Frequency in subset')
        plt.savefig(f'plots/outcomes_k{k}.pdf', bbox_inches="tight")
        #plt.show()
        plt.close()


    ##########
    # General authorship
    ##########

    # load saved results
    with open(f'saved/iclr2018_gen_rl6pl3.pkl', 'rb') as f:
        results = pickle.load(f)
    with open(f'saved/iclr2018_opt_gen_rl6pl3.pkl', 'rb') as f:
        opt = pickle.load(f)

    
    alg_labels = {'heuristic':'heuristic', 'random' : 'random'}
    algo_order = ['random', 'heuristic']
    markers = {'random': '_', 'heuristic' : 'x'}
    colors = {'random': 'black', 'heuristic' : 'green'}


    # find imbalances
    def imbalance(Xs):
        return abs(len(Xs[0]) - len(Xs[1])) #/ (len(Xs[0]) + len(Xs[1]))
    Rs, Ps, _ = results['heuristic'][0]
    print('Heuristic imbalance:', imbalance(Rs), imbalance(Ps))
    print([len(R) for R in Rs], [len(P) for P in Ps])
    r_imbs = []
    p_imbs = []
    for (Rs, Ps, _) in results['random']:
        r_imbs.append(imbalance(Rs))
        p_imbs.append(imbalance(Ps))
    print('Random imbalance:', np.mean(r_imbs), np.mean(p_imbs))


    # plot similarities
    sim_data = {}
    for algo, algo_results in results.items():
        sims = [s / opt for (_, _, s) in algo_results]
        sim_data[algo] = {}
        sim_data[algo]['mean'] = np.mean(sims)
        if len(sims) > 1:
            sim_data[algo]['err'] = scipy.stats.sem(sims)
        else:
            sim_data[algo]['err'] = None
    print(sim_data)

    args = dict(markersize=16, linestyle='')
    size = plt.rcParamsDefault["figure.figsize"]
    size[0] = 3.5
    plt.rcParams.update({'font.size': 18, 'figure.figsize' : size})
    for algo in algo_order:
        data = sim_data[algo]
        print(algo, data['mean'])
        plt.errorbar(0, 1 - np.array(data['mean']), yerr=data['err'], label=alg_labels[algo], color=colors[algo], marker=markers[algo], **args)
    plt.legend()
    plt.tight_layout()
    plt.xticks(ticks=[0], labels=[])
    plt.ylim(bottom=0, top=0.4)
    plt.ylabel('Fraction of optimal similarity lost')
    plt.savefig('plots/similarity_iclr_gen.pdf',bbox_inches="tight" )
    #plt.show()
    plt.close()

    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    # plot scores in each partition 
    partition = [[(p, p) for p in P] for P in results['heuristic'][0][1]]
    score_partition = get_score_split(partition, score_map)
    colors = ['red', 'blue']
    for i, part in enumerate(score_partition):
        plt.hist(part, bins=20, range=(min(score_map.values()), max(score_map.values())), alpha=0.5, histtype='stepfilled', edgecolor='black', facecolor=colors[i])
    plt.tight_layout()
    plt.xlabel('Mean review score')
    plt.ylabel('Frequency in subset')
    plt.savefig(f'plots/scores_gen.pdf', bbox_inches="tight")
    #plt.show()
    plt.close()

    # plot decisions in each partition 
    x = []
    widths = []
    plt.rcParams.update({'font.size': 16})
    outcome_counts = get_outcome_counts(partition, outcome_map)
    width = 0.8 / len(partition)
    for j in range(len(outcome_counts)):
        pos = (j * width) - ((width * (len(partition) - 1)) / 2)
        x.append(pos)
        widths.append(width)
    sizes = [len(part) for part in partition]
    o, p, w, r = (np.array(v) / np.array(sizes) for v in zip(*outcome_counts))
    widths = np.array(widths) - 0.02
    args = dict(edgecolor='black')
    plt.bar(x, o, widths, label='Oral', hatch='', **args)
    plt.bar(x, p, widths, bottom=o, label='Poster', hatch='/', **args)
    plt.bar(x, w, widths, bottom=p+o, label='Workshop', hatch='|', **args)
    plt.bar(x, r, widths, bottom=w+p+o, label='Reject', hatch='\\', **args)
    plt.tight_layout()
    plt.legend()
    plt.xticks(ticks=[0], labels=[alg_labels['heuristic']])
    #plt.xticks(ticks=[], labels=[])
    plt.xlim(left=-.5, right=1.5)
    plt.ylabel('Relative frequency in subset')
    plt.savefig(f'plots/outcomes_gen.pdf', bbox_inches="tight")
    #plt.show()
    plt.close()

  
