import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, required=True, help='net type')
parser.add_argument('--dataset', type=str, required=True, help='train dataset')
parser.add_argument('--ood', type=str, required=True, help='ood name')
parser.add_argument('--specific', type=str, required=True, help='specific information')
args = parser.parse_args()


score_array = []
with open('./result/'+args.net+'_'+args.dataset+'_'+args.ood+'_'+args.specific+'.csv') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        score_array.append( row[1:] )

np_score_array = np.asarray( score_array, dtype=float )
acc_ind = np_score_array[0,:]
acc_ood = np_score_array[1,:]
score = 0
for i in range(101):
    ood_dist = acc_ood[i+1] - acc_ood[i]
    score += (acc_ind[i] + acc_ind[i+1]) * ood_dist / 2

score_label = 'AUAAC: {:.3f}'.format(score)

plt.rc('font', size=14)
plt.ylim((0,1))
plt.xlim((0,1))
plt.plot(acc_ood, acc_ind, label=score_label)
plt.xlabel("ACC-OOD")
plt.ylabel("ACC-IND")
plt.legend()
plt.show()




