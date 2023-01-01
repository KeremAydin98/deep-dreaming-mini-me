import cv2
import numpy as np
from generating_dream import DreamyImages
import time

start = time.time()

img = cv2.imread("forest.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img)


dreamer = DreamyImages(5)
new_img = dreamer.generate_dream(img, 100, 0.01)

end = time.time()

print("Elapsed Time: ",end-start)

# concatenate image Horizontally
Hori = np.concatenate((img, new_img), axis=1)

cv2.imshow('Default', Hori)
cv2.waitKey(0)
cv2.destroyAllWindows()