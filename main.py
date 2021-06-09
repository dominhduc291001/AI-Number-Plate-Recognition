import cv2
import imutils
import numpy as np
import pytesseract
import pandas as pd
import time
import json
import textToSpeech as tts

#Load data
with open('db.json', encoding='utf-8') as fh:
    data = json.load(fh)

# Tham số
max_size = 5000
min_size = 1500

# Thêm đường dẫn tesseract orc
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract\tesseract.exe'


# Nhập hình ảnh
myImg = cv2.imread('./image/3.jpg', cv2.IMREAD_COLOR)
# IMREAD_COLOR: đặt chuẩn màu RGB
# Đặt lại kích thước hình ảnh
myImg = cv2.resize(myImg, (620, 480))

# Phát hiện cạnh
gray = cv2.cvtColor(myImg, cv2.COLOR_BGR2GRAY)  # Chuyển đổi màu sắc sang thang xám
gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Loại bỏ nhiễu bằng phương pháp làm mịn
edged = cv2.Canny(gray, 30, 200)  # Tìm các cạnh trong hình đã được làm mịn ở bước trên (Canny edge detection)

# Tìm đường viền trong hình ảnh
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
# Lọc contour theo area chỉ lấy 10 contour có giá trị lớn nhất( không lấy nhiều vì sẽ có nhiễu)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# Lặp lại các đường viền
for c in contours:
    #  Tính chu vi của viền
    peri = cv2.arcLength(c, True)
    # Lấy xấp xỉ đa giác
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)
    # Biển số là hình chữ nhật nên ta giữ lại các đường viền có 4 cạnh
    if len(approx) == 4 and max_size > cv2.contourArea(c) > min_size:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("Không phát hiện biển số !!!")
    tts.speak("Không phát hiện biển số !!!")
else:
    detected = 1

if detected == 1:
    # vẽ đường viền
    cv2.drawContours(myImg, [screenCnt], -1, (0, 255, 0), 3)

    # Che các bộ phận không phải biển số
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(myImg, myImg, mask=mask)

    # Cắt ảnh
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 2, topy:bottomy + 2]

    # Đọc ký tự từ hình ảnh được cắt
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Biển số được phát hiện là: ",text)
    
    #Xuất file csv
    raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
        'v_number': [text]}
    df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
    df.to_csv('data.csv')

    #Đọc Biển số 
    bs = text[0] + text[1]
    for i in data['bienso']:
        if(bs == str(i['id'])):
            mytext = "Biển số của bạn được đăng ký ở " + i['title'] + " Mã biển số nhận dạng được là " + text
            break
        else:
            mytext = "Không đọc được biển số trong hình"
    tts.speak(mytext) 

    # Hiển thị hình ảnh
    cv2.imshow('Input image', myImg)
    cv2.imshow('Cropped image', Cropped)
   
    # Chờ phím để thoát
    cv2.waitKey(0)
    cv2.destroyAllWindows()