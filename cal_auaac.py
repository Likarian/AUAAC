import numpy as np
import csv
import glob
import os

result_csvs = sorted(glob.glob('./result/*.csv'))

if os.path.exists('AUAAC.csv') == False:
    with open('AUAAC.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name','AUAAC'])

for csvfile in result_csvs:
    current_file = []
    with open(csvfile) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            current_file.append( row[1:] )

    np_current_file = np.asarray( current_file, dtype=float )
    performance = np_current_file[0,:]
    ood = np_current_file[1,:]
    score = 0
    for i in range(101):
        ood_dist = ood[i+1] - ood[i]
        score += (performance[i] + performance[i+1]) * ood_dist / 2
    
    with open('./AUAAC.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([ csvfile, score ])
