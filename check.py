from metrics import calc_map
import json 
import numpy as np

with open('/datasets/pku-autonomous-driving/annotations/json/train_objects_bbox.json','r') as f:
    distros_dict = json.load(f)
new_list = []
for i in range(9000):
    if distros_dict['annotations'][i]['iscrowd'] == 0:
        new_list.append(distros_dict['annotations'][i])
        new_list[-1]['score'] = 1
        #print(np.array(new_list[-1]['position']))

a = calc_map(new_list, new_list)
print(a)