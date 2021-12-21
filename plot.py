import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

fnames = ['iclr2018', 'preflib3', 'DA1', 'query', 'iclr k=2']
rand_means = []
rand_errs = []
k1_means = []
multi_means = []
rand_fair_means = []
rand_fair_errs = []
k1_fairs = []
multi_fairs = []

for fname in fnames:
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

    rand_scores = [s for (s, f) in results['random']]
    rand_fairs = [f for (s, f) in results['random']]
    rand_mean = np.mean(rand_scores)
    rand_err = sem(rand_scores)
    rand_fair_mean = np.mean(rand_fairs)
    rand_fair_err = sem(rand_fairs)

    rand_means.append(rand_mean / opt)
    rand_errs.append(rand_err / opt)
    k1_means.append(k1_mean / opt)
    multi_means.append(opt / opt)
    if 'iclr' in fname:
        rand_fair_means.append(rand_fair_mean)
        rand_fair_errs.append(rand_fair_err)
        k1_fairs.append(k1_fair)
        multi_fairs.append(multi_fair)


x = np.arange(5)
ms = 14
plt.errorbar(x, multi_means, label='multi-partition', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, k1_means, label='k=1 algorithm', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, rand_means, yerr=rand_errs, label='random partition', linestyle='', marker='_', markersize=ms)
plt.legend()

#plt.rcParams.update({'font.size': 16})
#plt.figure(figsize=figsize)
#plt.tight_layout()
plt.ylim(bottom=-0.05, top=1.05)
plt.xticks(ticks=x, labels=fnames)
plt.ylabel('Fraction of optimal non-SP similarity')
plt.savefig('similarity.png')
plt.show()


x = np.arange(2)
plt.errorbar(x, multi_fairs, label='multi-partition', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, k1_fairs, label='k=1 algorithm', linestyle='', marker='_', markersize=ms)
plt.errorbar(x, rand_fair_means, yerr=rand_fair_errs, label='random partition', linestyle='', marker='_', markersize=ms)
plt.legend()
plt.ylim(bottom=0, top=0.05)
plt.xticks(ticks=x, labels=[fnames[0], fnames[4]])
plt.ylabel('Fraction of true accepts missed')
plt.savefig('fairness.png')
plt.show()



