# https://stackoverflow.com/questions/58751101/count-number-of-cells-in-the-image
# https://blog.csdn.net/qq_38395705/article/details/106683455
# pip install opencv-python
# pip install win32gui
# pip install pandas


import cv2
import numpy as np
import win32gui
import win32con
import os
import pandas as pd

def countalgae (image):
    img = cv2.resize(image, (687,550))
    #return img
    original = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array([0,0,0])
    hsv_upper = np.array([180,60,255])
    mask1 = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = 255-mask1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cells = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h>60:
            cv2.rectangle(original, (x-3,y-3),(x+w+6,y+h+6), (0,0,255), 1)
            cells += 1
    windowname = 'Cells:{}'.format(cells)

    img2 = cv2.copyMakeBorder(img, 0, 0, 10, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    showpic = np.concatenate((original, img2), axis=1)
    #showpic = original

    cv2.imshow(windowname, showpic)
    hwnd = win32gui.FindWindow(None,windowname)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_SHOWWINDOW)
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
    win32gui.SetFocus(hwnd)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    try:
        err = int(input("Please enter a err:"))
    except:
        err = 0
    return [cells,err]


'''
imagepath =  '.\\20230312\\S0001.tif'
image = cv2.imread(imagepath)
a = countalgae (image)

'''
df_result = pd.DataFrame(columns=["imagepath", "date", "group", "groupnum","countnum","err","celldensity"])
datelist = [di for di in os.listdir('.\\') if '.' not in di]
for datestr in datelist:
    imagepaths =  ['.\\'+datestr+'\\'+i for i in os.listdir('.\\'+datestr) if i.endswith('.tif')]
    for imagepath in imagepaths:
        filename =  imagepath.split('\\')[-1].split('.')[-2]
        groupname = filename[:-4]
        groupnum = filename[-4:]
        image = cv2.imread(imagepath)
        countr = countalgae (image)
        countnum = countr[0]
        err = countr[1]
        df_result.loc[len(df_result.index)]=[imagepath,datestr,groupname,groupnum,countnum,err,(countnum+err)*129500]
        print(imagepath)
df_result.to_excel('count_result.xlsx',index= None)



