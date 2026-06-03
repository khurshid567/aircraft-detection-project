import cv2

img = cv2.imread("/home/zestiot-lat/Downloads/Untitled.jpeg")
# resized=cv2.resize(img,(600,600))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur=cv2.GaussianBlur(gray,(15,15),0)
# # crop=blur[100:150,100:150]
# canny=cv2.Canny(gray,50,150)

_,thresh=cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
contours,hirerachy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img,contours,-1,(0,255,0),3)
for cnt in contours:

    x, y, w, h = cv2.boundingRect(cnt)

    cv2.rectangle(img,
                  (x,y),
                  (x+w, y+h),
                  (255,0,0),
                  2)
cv2.imshow("Gray Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()