sizes = [64,128,256,512,1024,2048,4096]
kernel_names = ['Tiling', 'Loop unrolling','comp. optimization','prefetching','cublass']
K=1000

table = [
         [
          5.7, 10.53, 33.05, 207.93, 1.552*K, 12.174*K, 92.0312*K
         ],
         [
          11.03, 17.29, 31.85, 74.7, 521.64, 3.6185*K, 29.8318*K
         ],
         [
          9.96, 15.62, 28.42, 71.8, 495.22, 3.71534*K, 30.7892*K
         ],
         [
          9.22, 13.52,  23.86,  68.73, 453.1, 0, 30.7892*K
         ],
         [
          13.32, 14.91, 20.154, 58.03, 386.83, 2.5049*K, 18.2695*K
         ]
        ]

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {'kernel':[],'size':[],'GFLOPS':[]}

for i,kernel in enumerate(kernel_names):
  for j,size in enumerate(sizes):
    data['kernel'].append(kernel)
    data['size'].append(size)
    if  table[i][j]!=0:
      data['GFLOPS'].append(2*(size**3)/table[i][j]/K)
    else:
      data['GFLOPS'].append(None)
data
res = pd.DataFrame.from_dict(data)

fig,ax = plt.subplots(figsize=(15,10))
sns.pointplot(data=res[['kernel','size','GFLOPS']],
             y='GFLOPS',x='size',
             hue='kernel',ax=ax)
ax.set(title='size vs GFLOPS for each kernel')
plt.show()

