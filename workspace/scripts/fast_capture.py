print("[info] -> loading librarys...")
import os
print("       ---> os OK")
import cv2
print("       ---> cv2 OK")
import torch
print("       ---> torch OK")
from utils.torch_utils import select_device
print("       ---> select_device OK")
import time
print("       ---> time OK")
from models.yolo import Model
from yolov7.hubconf import custom
print("       ---> Model OK")
import mss
print("       ---> mss OK")
import numpy as np
print("       ---> numpy OK")
import win32con, win32api, win32gui, win32ui
print("       ---> win32 OK")
import argparse
print("       ---> argparse OK")
import ctypes
print("       ---> ctypes OK")
print("[info] -> librarys OK.")

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
STATE = True
dc = win32gui.GetDC(0)
dcObj = win32ui.CreateDCFromHandle(dc)
hwnd = win32gui.WindowFromPoint((0,0))
monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

def getMonitorDim():
    user32 = ctypes.windll.user32
    width, height = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    print("w", width, "h", height)
    return width, height
 
def moveMouse(x, y):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(x/SCREEN_WIDTH*65535.0), int(y/SCREEN_HEIGHT*65535.0))

def click(x, y):
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def getCursorPos():
    x, y = win32gui.GetCursorPos()
    return x, y

def checkCoord():
    while(True):
        getCursorPos()
        os.system('cls')

def load_model(path_or_model='path/to/model.pt', autoshape=True):
    model = torch.load(path_or_model, map_location=torch.device('cpu'))
    #model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    print("[info] -> selected device : ", device)
    return hub_model.to(device)

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(10, 10),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return img

def moveMouseToCenter(result):
    if(len(result.xyxy[0]) != 0):
        for i in range(0, len(result.xyxy[0])):
            tensor = result.xyxy[0][i] 
            name = int(tensor[5].item()) #name of object detected
            if name == 1:  
                x1 = int(tensor[0].item()) #top left x
                y1 = int(tensor[1].item()) #top left y
                x2 = int(tensor[2].item()) #bottom right x
                y2 = int(tensor[3].item()) #bottom right y
            
                xCenter = x1+(x2-x1)/2
                yHead = y1+(y2-y1)/3
                moveMouse(xCenter, yHead)

if __name__ == "__main__":
    #Getting args
    print("[info] -> getting args...")
    parser = argparse.ArgumentParser( prog = 'Fortnite AI Detection', description = 'Program for object detection in Fortnite')
    parser.add_argument('-m', '--model', help="Path to the model weights")
    args = parser.parse_args()
    modelPath = args.model
    if modelPath is None:
        modelPath = "PATH TO MODEL WEIGHTS"
    print("[info] -> args OK.")
    
    #Loading model
    print("[info] -> loading model...")
    #model = load_model(path_or_model = modelPath)
    model = custom(path_or_model="PATH TO MODEL WEIGHTS")
    print("[info] -> model OK.")
    
    #Screen recording
    print("[info] -> screen recording...")
    with mss.mss() as sct:
        while True:
            #close the program with ESC + F1
            if(win32api.GetAsyncKeyState(win32con.VK_ESCAPE) != 0 and win32api.GetAsyncKeyState(win32con.VK_F1) != 0):
                exit(0)
            
            #enable or disable cheat with F2         
            if(win32api.GetAsyncKeyState(win32con.VK_F2) != 0):
                if(STATE == True):
                    STATE = False
                    print("Status : Disable")
                    cv2.waitKey(100)
                else:
                    STATE = True
                    print("Status : Enable")
                    cv2.waitKey(100)
                                
            start = time.time()
            
            #getting img
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            
            #detection on the img
            if STATE:
                result = model(img)
                img = np.squeeze(result.render())
                moveMouseToCenter(result)             
            
            #FPS
            fps = 1/(time.time() - start)
            fps_string = "FPS: " + str(int(fps))
            print(fps_string)
            
            #image rendering
            img = draw_text(img, fps_string)
            cv2.imshow('Fortnite AI', img)
            cv2.waitKey(1)
