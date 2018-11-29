import cv2

face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
success,frame = cap.read()

yida = cv2.imread('./guo.jpg')


while (success):
    gray = cv2.cvtColor(frame, code= cv2.COLOR_BGR2GRAY)
    face_zones = face_detector.detectMultiScale(gray, 1.3, 10, minSize=(100, 100))

    for x,y,w,h in face_zones:
        img1 = frame[y:y+h, x:x+w]
        img2 = cv2.resize(yida,dsize=(w,h))
        # yida2 = cv2.resize(yida, dsize=(w,h))
        # frame[y:y+h,x:x+w ] = yida2

        rows, cols, channels = img2.shape
        roi = img1[0:rows, 0:cols]
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 二值化处理，第一个值原图，第二个进行分类的阈值，
        # 第三个高于/低于阈值时 赋予的新值，第四个方法选择参数，这里是黑白二值
        #ret, dst = cv2.threshold(src, thresh, maxval, type)
        #src： 输入图，只能输入单通道图像，通常来说为灰度图
        #dst： 输出图
        #thresh： 阈值
        #maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
        #type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV；
        # cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
        ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)

        # 取反 bitwise_not是对二进制数据进行“非”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“非”操作，~1=0，~0=1
        mask_inv = cv2.bitwise_not(mask)

        # 进行与运算 bitwise_and是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # 与运算
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

        # 加运算
        dst = cv2.add(img1_bg, img2_fg)
        img1[0:rows, 0:cols] = dst
        frame[y:y + h, x:x + w] = img1

    cv2.imshow('capture', frame)
    if cv2.waitKey(41) == ord('q'):
        break
    success, frame = cap.read()


cap.release()
cv2.destroyAllWindows()

