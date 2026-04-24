import numpy as np

from itertools import combinations

numbers = range(9)  # 0到8的数字
combinations_list = combinations(numbers, 3)  # 生成所有可能的三元组






label=np.loadtxt('train_label_1+7mer_all.csv')

def three_l_p(g,k,h):
    # print([g,k,h])
    a,b,c=label[:,g],label[:,k],label[:,h]
    intersection_count = 0
    union_count = 0


    for i in range(len(a)):
        if a[i]==1 or b[i]==1 or c[i]==1 :
            union_count +=1

        if (a[i] == 1 and b[i] == 1) or (a[i] == 1 and c[i] == 1) or (b[i] == 1 and c[i] == 1) or(a[i] == 1 and b[i] == 1 and c[i] == 1):
            intersection_count +=1
    ratio=intersection_count/union_count
    return ratio


for combination in combinations_list:

    g, k, h=combination

    r=three_l_p(g, k, h)
    #print(g, k, h)
    #print(r)





