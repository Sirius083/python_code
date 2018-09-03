# calculate mean over all other axises
tmp = np.mean(batch,axis =(0,1,2))


# using pickle save numpy array to disk
import pickle
with open('tiny_test.pickle', 'wb') as handle:
    # pickle.dump(images_decode, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_decode, handle)

with open('tiny_train.pickle', 'rb') as handle:
    images_decode = pickle.load(handle)
