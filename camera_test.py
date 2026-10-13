import cv2

cap = cv2.VideoCapture(1)   #0是电脑前置摄像头，1是9DTact摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("摄像头打开成功")

while True:
    ret, frame = cap.read()

    if not ret:
        print("无法读取图像")
        break

    cv2.imshow("9DTact Camera", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()