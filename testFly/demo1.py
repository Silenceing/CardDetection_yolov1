

classes=['I' ,'II', 'III', 'IV','V','VI',
         'VII','VIII', 'IX','X','XI','XII',
         'XIII','XIV','XV','XVI', 'XVII','XVIII',
         'XIX','XX','XXI','XXII','XXIII','XXIV']

cls=['VII','VIII', 'IX','X','XI','XII']

# for cl in cls:
#     for i, cls in enumerate(classes):
#         if cl==cls:
#             print(i)
import os
path='./data/input/CardDetection/images/0.jpg'
p=path.split('/',5)[5]
print(p.split('.')[0])

if os.path.isfile(path):
    print('-----------')
    file = path.split('/', 5)[5]
    print(file.split('.')[0])
image_name = os.path.basename(path)
print(image_name)

dict={}
list_=[]

for i in range(5):
    dict['a']=1
    dict['b'] = 2
    dict['c'] = 3
    dict['d'] = 4
    dict['e'] = 5
    list_.append(dict)
print(list_)

box=[1,2,3,4,5,6]
print(box[1:5]*2)


import torch

aa=torch.randn(2,7,7,34)
print(aa.size()[-2:])