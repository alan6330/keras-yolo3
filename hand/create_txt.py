import pandas as pd
import os
csv=pd.read_csv('dev.csv')
del csv['p1']
del csv['p2']
csv['hand_0']=0
csv['hand_1']=1
csv['hand_2']=2
csv['hand_3']=3
csv['hand_4']=4
csv['hand_5']=5
out=csv[['image_path',
         'p4','p3','p6','p5','hand_0',
         'p8','p7','p10','p9','hand_1',
         'p12','p11','p14','p13','hand_2',
         'p16','p15','p18','p17','hand_3',
         'p20','p19','p22','p21','hand_4',
         'p24','p23','p26','p25','hand_5']]

out.to_csv('hand.txt',header=False,index=False,)
new_txt=[]
index_out=[1,6,11,16,21,26]
with open('hand.txt','r') as f:
    txt=f.readlines()
    for i in txt:
        dot=0
        ii=list(i)
        for j,jj in enumerate(i):
            if ii[j]==',':
                dot+=1
                if dot in index_out:
                    ii[j]=' '
        iii=''.join(ii)
        new_txt.append(iii)
fp = open("all_hand.txt",'w')
with open('all_hand.txt','a') as f:
    for i in new_txt:
        ii='hand/'+i
        f.write(ii)

my_file = 'hand.txt'
if os.path.exists(my_file):
    os.remove(my_file)