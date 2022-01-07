import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import itertools
from collections import defaultdict

fnames = ['iclr2018', 'preflib3', 'DA1', 'query']
ks = [1, 2]

# TODO fix this to make it not shit
# for each type of result, want to concat the results for each 
# (sim, fair, p) * (multi, k1, rand) -> list of results
# main_results == defaultdict(list)
def collect_filename_results(fname_results, main_results):
    results = {}
    opt = fname_results['multi'][0][0]
    for alg in ['multi', 'k1', 'random']:
        results[(alg, 'sim')] = [s / opt for (s, f) in fname_results[alg]]
        results[(alg, 'fair')] = [f for (s, (f, p)) in fname_results[alg]]
        results[(alg, 'p')] = [p for (s, (f, p)) in fname_results[alg]]
        for res_type in ['sim', 'fair', 'p']:
            r = results[(alg, res_type)]
            main_results[(alg, res_type, 'mean')].append(np.mean(r))
            main_results[(alg, res_type, 'err')].append(sem(r) if len(r) > 1 else 0)



rand_means = []
rand_errs = []
k1_means = []
multi_means = []
rand_fair_means = []
rand_fair_errs = []
k1_fairs = []
multi_fairs = []
rand_p_means = []
rand_p_errs = []
k1_ps = []
multi_ps = []
# map type -> algo -> list of results

main_results = defaultdict(list)
for k, fname in itertools.product(ks, fnames):
    print(fname, k)
    infilename = f'{fname}_results_k{k}.npz'
    d = np.load(infilename, allow_pickle=True)
    results = d['results'][()] # method -> list of (score, fairness)

    collect_filename_results(results, main_results)

    #opt, (multi_fair, multi_p) = results['multi'][0]
    #k1_mean, (k1_fair, k1_p) = results['k1'][0]

    '''
    if fname == 'iclr k=2':
        infilename = 'iclr2018_results_k2.npz'
        d = np.load(infilename, allow_pickle=True)
        results = d['results'][()] # method -> list of (score, fairness)
        opt, multi_fair = results['multi'][0]
        k1_mean = -1 * opt
        k1_fair = -1
    else:
        infilename = fname + '_results.npz'
        d = np.load(infilename, allow_pickle=True)
        results = d['results'][()] # method -> list of (score, fairness)
        infilename = fname + '_results2.npz'
        d = np.load(infilename, allow_pickle=True)
        results2 = d['results'][()]

        opt, multi_fair = results2['multi'][0]
        k1_mean, k1_fair = results['k1'][0]
    '''

    '''
    rand_scores = [s for (s, f) in results['random']]
    rand_fairs = [f for (s, (f, p)) in results['random']]
    rand_ps = [p for (s, (f, p)) in results['random']]
    rand_mean = np.mean(rand_scores)
    rand_err = sem(rand_scores)
    rand_fair_mean = np.mean(rand_fairs)
    rand_fair_err = sem(rand_fairs)
    rand_p_mean = np.mean(rand_ps)
    rand_p_err = sem(rand_ps)

    rand_means.append(rand_mean / opt)
    rand_errs.append(rand_err / opt)
    k1_means.append(k1_mean / opt)
    multi_means.append(opt / opt)
    if 'iclr' in fname:
        rand_fair_means.append(rand_fair_mean)
        rand_fair_errs.append(rand_fair_err)
        k1_fairs.append(k1_fair)
        multi_fairs.append(multi_fair)
        rand_p_means.append(rand_p_mean)
        rand_p_errs.append(rand_p_err)
        k1_ps.append(k1_p)
        multi_ps.append(multi_p)
    '''



x = np.arange(4)
ms = 14
plt.errorbar(x, main_results[('multi', 'sim', 'mean')][:4], label='multi-partition', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, main_results[('k1', 'sim', 'mean')][:4], label='k=1 algorithm', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, main_results[('random', 'sim', 'mean')][:4], yerr=main_results[('random', 'sim', 'err')][:4], label='random partition', linestyle='', marker='_', markersize=ms)
plt.legend()

#plt.rcParams.update({'font.size': 16})
#plt.figure(figsize=figsize)
#plt.tight_layout()
plt.ylim(bottom=-0.05, top=1.05)
plt.xticks(ticks=x, labels=fnames)
plt.ylabel('Fraction of optimal non-SP similarity, k=1')
plt.savefig('similarity_k1.png')
plt.show()

x = np.arange(4)
plt.errorbar(x, main_results[('multi', 'sim', 'mean')][4:], label='multi-partition', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, main_results[('k1', 'sim', 'mean')][4:], label='k=1 algorithm', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, main_results[('random', 'sim', 'mean')][4:], yerr=main_results[('random', 'sim', 'err')][4:], label='random partition', linestyle='', marker='_', markersize=ms)
plt.legend()

plt.ylim(bottom=-0.05, top=1.05)
plt.xticks(ticks=x, labels=fnames)
plt.ylabel('Fraction of optimal non-SP similarity, k=2')
plt.savefig('similarity_k2.png')
plt.show()


def select_iclr(l):
    return [l[0], l[4]]


x = np.arange(2)
plt.errorbar(x, select_iclr(main_results[('multi', 'fair', 'mean')]), label='multi-partition', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, select_iclr(main_results[('k1', 'fair', 'mean')]), label='k=1 algorithm', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, select_iclr(main_results[('random', 'fair', 'mean')]), yerr=select_iclr(main_results[('random', 'fair', 'err')]), label='random partition', linestyle='', marker='_', markersize=ms)
plt.legend()
plt.ylim(bottom=0, top=0.05)
plt.xticks(ticks=x, labels=['iclr k=1', 'iclr k=2'])
plt.ylabel('Fraction of true accepts missed')
plt.savefig('fairness.png')
plt.show()

x = np.arange(2)
plt.errorbar(x, select_iclr(main_results[('multi', 'p', 'mean')]), label='multi-partition', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, select_iclr(main_results[('k1', 'p', 'mean')]), label='k=1 algorithm', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, select_iclr(main_results[('random', 'p', 'mean')]), yerr=select_iclr(main_results[('random', 'p', 'err')]), label='random partition', linestyle='', marker='_', markersize=ms)
plt.legend()
plt.ylim(bottom=0, top=1)
plt.xticks(ticks=x, labels=['iclr k=1', 'iclr k=2'])
plt.ylabel('KS test, p-value')
plt.savefig('p_values.png')
plt.show()




