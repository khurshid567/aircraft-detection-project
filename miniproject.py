import cv2
from ultralytics import YOLO

model = YOLO("/home/zestiot-lat/PyCharmMiscProject/runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("/home/zestiot-lat/Downloads/5865604-uhd_3840_2160_25fps.mp4")

while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame)
    count=0
    for r in results:
        names=r.names
        for box in r.boxes:
            conf=float(box.conf[0])
            if conf>0.5:
                count+=1
                x1,y1,x2,y2=map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cls=int(box.cls[0])
                label=names[cls]
                cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(frame,f"count{count}",(40,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
