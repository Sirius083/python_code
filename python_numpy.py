# calculate mean over all other axises
tmp = np.mean(batch,axis =(0,1,2))


# using pickle save numpy array to disk
import pickle
with open('tiny_test.pickle', 'wb') as handle:
    # pickle.dump(images_decode, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_decode, handle)

with open('tiny_train.pickle', 'rb') as handle:
    images_decode = pickle.load(handle)
   


# 改成one-hot编码
a = np.array([1, 0, 3])
b = np.zeros((3, 4))
b[np.arange(3), a] = 1
print(b)

# 检查一个object的类型是否为 np.ndarray
# version1: isinstance(obj, np.ndarray)
# version2: type(obj) is np.ndarray
