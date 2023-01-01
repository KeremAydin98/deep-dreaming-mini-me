import cv2
from generating_dream import DreamyImages
import time

start = time.time()

video_path = "./Data/giraffes.mp4"
dreamer = DreamyImages()

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('Result/output.mp4',fourcc, 20.0, (300,300))

print("Video started!!")
while(cap.isOpened()):

    ret, frame = cap.read()

    if ret:

        frame = cv2.resize(frame, (300, 300))
        frame = dreamer.generate_dream(frame, 100, 0.01)
        print("Frame ended!!")
        out.write(frame)

    else:

        break


cap.release()
out.release()
cv2.destroyAllWindows()

end = time.time()

print("Elapsed Time: ",end-start)



