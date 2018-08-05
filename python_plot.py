# visualize jpeg file in python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread(filenames_labels[0][0])
imgplot = plt.imshow(img)
plt.show()
