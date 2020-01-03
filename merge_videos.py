import cv2

cap1 = cv2.VideoCapture('jiaxin_2.avi')
cap2 = cv2.VideoCapture('jiaxin_1.avi')

if (cap1.isOpened()== False):
  print("Error opening video stream or file")
if (cap2.isOpened()== False):
  print("Error opening video stream or file")

width = int(cap1.get(3))
height = int(cap1.get(4))
out = cv2.VideoWriter('jiaxin.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (width*2,height))
while (cap1.isOpened() and cap2.isOpened()):
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()

    if ret is True:
        merged = cv2.hconcat([frame2, frame1])
        out.write(merged)
        cv2.imshow("output", merged)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap1.release()
cap2.release()
out.release()

cv2.destroyAllWindows()
