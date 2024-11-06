import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

img = cv2.imread('test_image.png')

#img_2d = img.reshape((-1, 3)).astype(np.float32) / 255.0

img_2d = img.reshape((-1, 3))
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(img_2d)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

compressed = kmeans.cluster_centers_[kmeans.labels_]
compressed = (compressed * 255).astype(np.uint8)

m = 0
height, width, channels = np.shape(img)
for i in range(width):
  for j in range(height):
    pixel = img[j][i]
    newValue = compressed[m]
    img[j][i] = newValue
    m += 1

cv2.imwrite('compressed_image.png', img)

score = silhouette_score(img_2d, kmeans.labels_)
print("Silhouette: ", score)

plt.scatter(img_2d[:, 0], img_2d[:, 1], c=kmeans.labels_, s=1)
plt.title("K-means Clustering of Image Pixels")
plt.xlabel("Red")
plt.ylabel("Green")
plt.show()
