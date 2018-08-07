# visualize jpeg file in python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread(filenames_labels[0][0])
imgplot = plt.imshow(img)
plt.show()


# Note: path format in windows, file_path = r''
import matplotlib.pyplot as plt
# image_path = 'E:/tiny_imagenet/tiny-imagenet-200/train\n01443537\images\n01443537_0.JPEG' # wrong
image_path = 'E:/tiny_imagenet/tiny-imagenet-200/train\n01443537\images\n01443537_0.JPEG'   # correct
img = plt.imread(image_path)
imgplot = plt.imshow(img)
plt.show()

