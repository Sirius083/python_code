# find argmax index of a 3d array

import numpy as np

result = []
tmp = np.array(range(24)).reshape(2,3,4)
print('tmp', tmp)
tmp_max = np.max(tmp, axis =(1,2))
tmp_ind = [np.argwhere(x == tmp) for x in tmp_max]
tmp_ind = np.squeeze(np.array(tmp_ind))[:,1:]
tmp_max = np.expand_dims(tmp_max, axis=1)
res = np.concatenate((tmp_max, tmp_ind), axis=1)
result = result + [res]


tmp = np.array(range(24,48)).reshape(2,3,4)
print('tmp', tmp)
tmp_max = np.max(tmp, axis =(1,2))
tmp_ind = [np.argwhere(x == tmp) for x in tmp_max]
tmp_ind = np.squeeze(np.array(tmp_ind))[:,1:]
tmp_max = np.expand_dims(tmp_max, axis=1)
res = np.concatenate((tmp_max, tmp_ind), axis=1)
result = result + [res]



res = np.array(result)
res = res.reshape(res.shape[0] * res.shape[1], res.shape[2])

# sort by second column
top9 = res[res[:,0].argsort()][-9:][::-1]

