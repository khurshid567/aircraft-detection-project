import cv2

img = cv2.imread("/home/zestiot-lat/Downloads/bottle.avif")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
for cnt in contours:

    area = cv2.contourArea(cnt)

    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(img,
                      (x,y),
                      (x+w, y+h),
                      (0,255,0),
                      2)
cv2.imshow("image",img)
cv2.waitKey(0)