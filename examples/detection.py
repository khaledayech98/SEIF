import csv


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings

from examples.thorax import f1, tri_perm, ex_mat, mesh_new, mesh_obj, anoms
from examples.thorax1 import voltage

warnings.filterwarnings('ignore')

# opening the csv file in 'w+' mode




with open('voltage.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(['voltage number','voltage amplitude','permittivity'])
    for i in range(len(f1)):
     writer.writerow([(i), (f1[i]), ((tri_perm[i]))])





#df = pd.DataFrame(columns=["Data"])
#df= df=pd.read_csv(r'D:/home/pyEIT-master/examples/data2.csv')
#print(df)
'''
for dataset in datasets:
    df = pd.DataFrame(dataset)
    df = df[columns]
    mode = 'a+'
    df.to_csv('./new.csv', encoding='utf-8', mode=mode, header=header, index=False)
    header = False
'''


