import matplotlib.pyplot as plt
import numpy as np

# Sample data
values = [43809
,31326
,20569
,32380
,41188
,42534
,50698
,43609
,36068
,34025
,41853
,29183
,34562
,22589
,38385
,60172
,46384
,44756
,38960
,40101
,40924
,37993
,50444
,51814
,39565
,36056
,34143
,35742
,7485
,48520
,42601
,40490
,26050
,33967
,0
,5978
,16629
,9529
,29413
,28053
,38060
,38329
,45545
,57031
,71563
,89936
,125240
,122058
,28593
,39503
,160925
,0
,60425
,42652
,28102
,40064
,63834
,38885
,14596
,46555
,55420
,41995
,34208
,34308
,38820
,50211
,40522
,52207
,45424
,23708
,38135
,35029
,40240
,42240
,31576
,26177
,31408
,66509
,140887
,67503
,48193
,22926
,37480
,28053
,48973
,35057
,21049
,38292
,41954
,60414
,43322
,42429
,47724
,68670
,40298
,62381
,0
,8732
,5073
,32583
,72978
,89665
,72784
,41387
,62777
,68571
,120533
,139900
,106186
,120288
,108068
,105137
,88731
,51020
,52470
,9686
,13033
,11683
,13208]

# Calculate percentiles
percentiles = np.percentile(values, np.arange(0, 101, 10))  # 0th to 100th percentile in steps of 10

# Plotting the percentile distribution
plt.figure(figsize=(8, 4))
plt.plot(np.arange(0, 101, 10), percentiles, marker='o')  # Connect percentile points with a line
plt.xticks(np.arange(0, 101, 10))
plt.xlabel('Percentile')
plt.ylabel('Score')
plt.title('Percentile Distribution of Scores')
plt.grid(True)
plt.show()
