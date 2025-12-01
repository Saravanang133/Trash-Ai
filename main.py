# main.py
import os
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
import cv2
import shutil
import time
import PIL.Image
from PIL import Image, ImageChops
import numpy as np
import argparse
import imagehash
import mysql.connector
import urllib.request
import urllib.parse
from werkzeug.utils import secure_filename
from urllib.request import urlopen
import webbrowser

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="trash"

)

UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    '''cutoff=10
    hash0 = imagehash.average_hash(Image.open("static/upload/image2.jpg")) 
    hash1 = imagehash.average_hash(Image.open("upload/getimg3.jpg"))
    cc=hash0 - hash1
    if hash0 - hash1 <= cutoff:
        print("yes="+str(cc))
    else:
        print("no="+str(cc))'''
        
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            #session['loggedin'] = True
            #session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('admin'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/home', methods=['GET', 'POST'])
def home():
    msg=""
    if request.method=='GET':
        msg = request.args.get('msg')
    if request.method=='POST':
        otype = request.form['otype']
        name = request.form['name']

        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM store_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        imgname="image"+str(maxid)+".jpg"


        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(imgname)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            timg="train"+str(maxid)+".jpg"
            shutil.copy("static/upload/"+imgname, 'static/trained/'+timg)
            
            sql = "INSERT INTO store_data(id, otype, name, imgname) VALUES (%s, %s, %s, %s)"
            val = (maxid, otype, name, imgname)
            mycursor.execute(sql,val)
            mydb.commit()
            msg="Upload success"
            return redirect(url_for('home', msg=msg))
    
    return render_template('home.html', msg=msg)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM admin')
    data = cursor.fetchone()
    if request.method=='POST':
        mobile=request.form['mobile']
        email=request.form['email']
        cursor1 = mydb.cursor()
        sql = "update admin set mobile=%s, email=%s"
        val = (mobile, email)
        cursor1.execute(sql, val)
        mydb.commit()

        ff=open("mobile.txt","w")
        ff.write(mobile)
        ff.close()

        ff=open("email.txt","w")
        ff.write(email)
        ff.close()
        
    
    return render_template('admin.html',data=data)



@app.route('/viewdata', methods=['GET', 'POST'])
def viewdata():
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM store_data')
    data = cursor.fetchall()

    if request.method=='GET':
        act = request.args.get('act')
        did = request.args.get('did')
        if act=="del":
            cursor = mydb.cursor()
            cursor.execute('SELECT * FROM store_data WHERE id = %s', (did, ))
            drow = cursor.fetchone()
            fn=drow[3]
            #os.remove("static/upload/"+fn)
            cursor1 = mydb.cursor()
            cursor1.execute('delete FROM store_data WHERE id = %s', (did, ))
            mydb.commit()   
            return redirect(url_for('viewdata'))
    return render_template('viewdata.html',data=data)

def getbox(im, color):
    bg = Image.new(im.mode, im.size, color)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    return diff.getbbox()

def split(im):
    retur = []
    emptyColor = im.getpixel((0, 0))
    box = getbox(im, emptyColor)
    width, height = im.size
    pixels = im.getdata()
    sub_start = 0
    sub_width = 0
    offset = box[1] * width
    for x in range(width):
        if pixels[x + offset] == emptyColor:
            if sub_width > 0:
                retur.append((sub_start, box[1], sub_width, box[3]))
                sub_width = 0
            sub_start = x + 1
        else:
            sub_width = x + 1
    if sub_width > 0:
        retur.append((sub_start, box[1], sub_width, box[3]))
    return retur


@app.route('/selection', methods=['GET', 'POST'])
def selection():
    if request.method=='POST':
        return redirect(url_for('training', act="on", page='0', img='0'))
    return render_template('selection.html')


@app.route('/training', methods=['GET', 'POST'])
def training():
    act="on"
    page="0"
    pg=""
    fn=""
    fnn=""
    img='1'
    tit=""
    m=0
    n=0
    #tot=5
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM store_data order by id')
    drow = cursor.fetchall()
    cursor.close()
    tot=len(drow)
    
    
    #if request.method=='POST':
        
        #return redirect(url_for('training', act="on"))
    if request.method=='GET':
        act = request.args.get('act')
        page = request.args.get('page')
        img = request.args.get('img')
        n = int(page)
        if n==0:
            m = int(img)+1
        else:
            m = int(img)
            
        pg=str(n)
        page=pg
        img = str(m)
        
        mg = m-1
        cursor2 = mydb.cursor()
        cursor2.execute('SELECT * FROM store_data order by id limit %s, 1', (mg, ))
        drow2 = cursor2.fetchall()
        cursor2.close()
        fn=drow2[0][3]
        fid=drow2[0][0]
        df="t"+str(fid)+".jpg"
        path="static/upload/"+fn
        path2="static/trained/"+df
        timg="train"+str(fid)+".jpg"
        #shutil.copy(path, 'static/trained/'+timg)
        if m<=tot:
            act="on"
            
            if n<4:
                if n==0:
                    tit="Preprocessing"
                    image = Image.open("static/trained/"+timg)
                    new_image = image.resize((300, 300))
                    new_image.save(path2)
                    fnn=timg
                    '''im = Image.open(path)

                    for idx, box in enumerate(split(im)):
                        im.crop(box).save("static/trained/result.jpg".format(idx))
                    fn="result.jpg"'''

                    # construct the argument parse 
                    parser = argparse.ArgumentParser(
                        description='Script to run MobileNet-SSD object detection network ')
                    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
                    parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                                      help='Path to text network file: '
                                                           'MobileNetSSD_deploy.prototxt for Caffe model or '
                                                           )
                    parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                                     help='Path to weights: '
                                                          'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                                          )
                    parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
                    args = parser.parse_args()

                    # Labels of Network.
                    classNames = { 0: 'background',
                        1: 'mobile', 2: 'bicycle', 3: 'cup', 4: 'glass',
                        5: 'bottle', 6: 'paper', 7: 'car', 8: 'cat', 9: 'chair',
                        10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                        14: 'motorbike', 15: 'person', 16: 'pottedplant',
                        17: 'plastic', 18: 'sofa', 19: 'cellphone', 20: 'tvmonitor' }

                    # Open video file or capture device. 
                    '''if args.video:
                        cap = cv2.VideoCapture(args.video)
                    else:
                        cap = cv2.VideoCapture(0)'''

                    #Load the Caffe model 
                    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

                    #while True:
                    # Capture frame-by-frame
                    #ret, frame = cap.read()
                    frame = cv2.imread("static/trained/"+timg)
                    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

                    # MobileNet requires fixed dimensions for input image(s)
                    # so we have to ensure that it is resized to 300x300 pixels.
                    # set a scale factor to image because network the objects has differents size. 
                    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
                    # after executing this command our "blob" now has the shape:
                    # (1, 3, 300, 300)
                    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
                    #Set to network the input blob 
                    net.setInput(blob)
                    #Prediction of network
                    detections = net.forward()

                    #Size of frame resize (300x300)
                    cols = frame_resized.shape[1] 
                    rows = frame_resized.shape[0]

                    #For get the class and location of object detected, 
                    # There is a fix index for class, location and confidence
                    # value in @detections array .
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2] #Confidence of prediction 
                        if confidence > args.thr: # Filter prediction 
                            class_id = int(detections[0, 0, i, 1]) # Class label

                            # Object location 
                            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                            yLeftBottom = int(detections[0, 0, i, 4] * rows)
                            xRightTop   = int(detections[0, 0, i, 5] * cols)
                            yRightTop   = int(detections[0, 0, i, 6] * rows)
                            
                            # Factor for scale to original size of frame
                            heightFactor = frame.shape[0]/300.0  
                            widthFactor = frame.shape[1]/300.0 
                            # Scale object detection to frame
                            xLeftBottom = int(widthFactor * xLeftBottom) 
                            yLeftBottom = int(heightFactor * yLeftBottom)
                            xRightTop   = int(widthFactor * xRightTop)
                            yRightTop   = int(heightFactor * yRightTop)
                            # Draw location of object  
                            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                          (0, 255, 0))
                            try:
                                        
                                image = cv2.imread("static/trained/"+timg)
                                cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                                
                                cv2.imwrite("static/trained/"+timg, cropped)
                                mm2 = PIL.Image.open('static/trained/'+timg)
                                rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                                rz.save('static/trained/'+timg)
                            except:
                                print("none")
                                #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                            # Draw label and confidence of prediction in frame resized
                            if class_id in classNames:
                                label = classNames[class_id] + ": " + str(confidence)
                                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                                yLeftBottom = max(yLeftBottom, labelSize[1])
                                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                                     (255, 255, 255), cv2.FILLED)
                                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                                print(label) #print class and confidence


                    
                    
                elif n==1:
                    tit="Grayscale"
                    image = Image.open(path2)
                    new_image = image.resize((300, 300))
                    new_image.save(path2)
                    fnn=df
                    
                elif n==2:
                    image = Image.open(path2).convert('L')
                    image.save(path2)
                    tit="Resizing"
                    fnn=df
                elif n==3:
                    tit="Feature Selection"
                    frame = cv2.imread(path2)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
                      
                    # define range of red color in HSV 
                    lower_red = np.array([30,150,50]) 
                    upper_red = np.array([255,255,180]) 

                    # create a red HSV colour boundary and  
                    # threshold HSV image 
                    mask = cv2.inRange(hsv, lower_red, upper_red) 

                    # Bitwise-AND mask and original image 
                    res = cv2.bitwise_and(frame,frame, mask= mask)

                    edges = cv2.Canny(frame,100,200)
                    cv2.imwrite(path2, frame)
                    fnn=df
                    
                    
                    cursor3 = mydb.cursor()
                    cursor3.execute('update store_data set train_st=1 where id=%s', (fid, ))
                    mydb.commit()    
                else:
                    
                    tit="Classified"
                    fnn=timg
                n = int(page)+1
                pg=str(n)
                page=pg
            else:
                tit="Classified"
                fnn=timg
                page='0'
                if m==tot:
                    
                    act="ok"
               
        else:
            act="ok"
                
    
    return render_template('training.html',tit=tit, img=img, page=page, act=act, fn=fnn)

######New Training####
@app.route('/training2', methods=['GET', 'POST'])
def training2():
    try:
        
        act="on"
        page="0"
        pg=""
        fn=""
        fnn=""
        img='1'
        tit=""
        m=0
        n=0
        #tot=5
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM store_data where train_st=0 order by id')
        drow = cursor.fetchall()
        cursor.close()
        tot=len(drow)
        
        
        #if request.method=='POST': && train_st=0
            
            #return redirect(url_for('training', act="on"))
        if request.method=='GET':
            act = request.args.get('act')
            page = request.args.get('page')
            img = request.args.get('img')
            n = int(page)
            if n==0:
                m = int(img)+1
            else:
                m = int(img)
                
            pg=str(n)
            page=pg
            img = str(m)
            
            mg = m-1
            cursor2 = mydb.cursor()
            cursor2.execute('SELECT * FROM store_data where train_st=0 order by id limit %s, 1', (mg, ))
            drow2 = cursor2.fetchall()
            cursor2.close()
            print(drow2)
            fn=drow2[0][3]
            fid=drow2[0][0]
            df="t"+str(fid)+".jpg"
            path="static/upload/"+fn
            path2="static/trained/"+df
            timg="train"+str(fid)+".jpg"
            shutil.copy(path, 'static/trained/'+timg)
            
            if m<=tot:
                act="on"
                
                if n<4:
                    if n==0:
                        tit="Preprocessing"
                        image = Image.open("static/trained/"+timg)
                        new_image = image.resize((300, 300))
                        new_image.save(path2)
                        fnn=timg

                        # construct the argument parse 
                        parser = argparse.ArgumentParser(
                            description='Script to run MobileNet-SSD object detection network ')
                        parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
                        parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                                          help='Path to text network file: '
                                                               'MobileNetSSD_deploy.prototxt for Caffe model or '
                                                               )
                        parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                                         help='Path to weights: '
                                                              'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                                              )
                        parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
                        args = parser.parse_args()

                        # Labels of Network.
                        classNames = { 0: 'background',
                            1: 'mobile', 2: 'bicycle', 3: 'cup', 4: 'glass',
                            5: 'bottle', 6: 'paper', 7: 'car', 8: 'cat', 9: 'chair',
                            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                            14: 'motorbike', 15: 'person', 16: 'pottedplant',
                            17: 'plastic', 18: 'sofa', 19: 'cellphone', 20: 'tvmonitor' }

                        # Open video file or capture device. 
                        '''if args.video:
                            cap = cv2.VideoCapture(args.video)
                        else:
                            cap = cv2.VideoCapture(0)'''

                        #Load the Caffe model 
                        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

                        #while True:
                        # Capture frame-by-frame
                        #ret, frame = cap.read()
                        frame = cv2.imread("static/trained/"+timg)
                        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

                        # MobileNet requires fixed dimensions for input image(s)
                        # so we have to ensure that it is resized to 300x300 pixels.
                        # set a scale factor to image because network the objects has differents size. 
                        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
                        # after executing this command our "blob" now has the shape:
                        # (1, 3, 300, 300)
                        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
                        #Set to network the input blob 
                        net.setInput(blob)
                        #Prediction of network
                        detections = net.forward()

                        #Size of frame resize (300x300)
                        cols = frame_resized.shape[1] 
                        rows = frame_resized.shape[0]

                        #For get the class and location of object detected, 
                        # There is a fix index for class, location and confidence
                        # value in @detections array .
                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2] #Confidence of prediction 
                            if confidence > args.thr: # Filter prediction 
                                class_id = int(detections[0, 0, i, 1]) # Class label

                                # Object location 
                                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                                xRightTop   = int(detections[0, 0, i, 5] * cols)
                                yRightTop   = int(detections[0, 0, i, 6] * rows)
                                
                                # Factor for scale to original size of frame
                                heightFactor = frame.shape[0]/300.0  
                                widthFactor = frame.shape[1]/300.0 
                                # Scale object detection to frame
                                xLeftBottom = int(widthFactor * xLeftBottom) 
                                yLeftBottom = int(heightFactor * yLeftBottom)
                                xRightTop   = int(widthFactor * xRightTop)
                                yRightTop   = int(heightFactor * yRightTop)
                                # Draw location of object  
                                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                              (0, 255, 0))
                                try:
                                            
                                    image = cv2.imread("static/trained/"+timg)
                                    cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                                    
                                    cv2.imwrite("static/trained/"+timg, cropped)
                                    mm2 = PIL.Image.open('static/trained/'+timg)
                                    rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                                    rz.save('static/trained/'+timg)
                                except:
                                    print("none")
                                    #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                                # Draw label and confidence of prediction in frame resized
                                if class_id in classNames:
                                    label = classNames[class_id] + ": " + str(confidence)
                                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                                    yLeftBottom = max(yLeftBottom, labelSize[1])
                                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                                         (255, 255, 255), cv2.FILLED)
                                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                                    print(label) #print class and confidence




                        
                        '''im = Image.open(path)

                        for idx, box in enumerate(split(im)):
                            im.crop(box).save("static/trained/result.jpg".format(idx))
                        fn="result.jpg"'''
                        
                    elif n==1:
                        tit="Grayscale"
                        image = Image.open(path2)
                        new_image = image.resize((300, 300))
                        new_image.save(path2)
                        fnn=df
                        
                    elif n==2:
                        image = Image.open(path2).convert('L')
                        image.save(path2)
                        tit="Resizing"
                        fnn=df
                    elif n==3:
                        tit="Feature Selection"
                        frame = cv2.imread(path2)
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
                          
                        # define range of red color in HSV 
                        lower_red = np.array([30,150,50]) 
                        upper_red = np.array([255,255,180]) 

                        # create a red HSV colour boundary and  
                        # threshold HSV image 
                        mask = cv2.inRange(hsv, lower_red, upper_red) 

                        # Bitwise-AND mask and original image 
                        res = cv2.bitwise_and(frame,frame, mask= mask)

                        edges = cv2.Canny(frame,100,200)
                        cv2.imwrite(path2, frame)
                        fn="grayscale.jpg"
                        timg="train"+str(fid)+".jpg"
                        #shutil.copy('static/trained/grayscale.jpg', 'static/trained/'+timg)
                        cursor3 = mydb.cursor()
                        cursor3.execute('update store_data set train_st=1 where id=%s', (fid, ))
                        mydb.commit()
                        fnn=timg
                    else:
                        
                        tit="Classified"
                        fnn=timg
                    n = int(page)+1
                    pg=str(n)
                    page=pg
                else:
                    tit="Classified"
                    page='0'
                    fnn=timg
                    if m==tot:
                        act="ok"
                   
            else:
                act="ok"
    except:
        act="ok"
                
    
    return render_template('training2.html',tit=tit, img=img, page=page, act=act, fn=fnn)
######
@app.route('/testing', methods=['GET', 'POST'])
def testing():
    cnt=0
    act=""
    ff=open("mess.txt","w")
    ff.write("")
    ff.close()

    ff=open("sms.txt","w")
    ff.write("1")
    ff.close()
                    
    if request.method=='POST':
        cursor2 = mydb.cursor()
        cursor2.execute('SELECT * FROM store_data WHERE train_st=0 order by id')
        drow3 = cursor2.fetchall()
        cnt=len(drow3)
        print(cnt)
        if cnt==0:
            print("test")
            return redirect(url_for('monitor'))
        else:
            act="train"
    return render_template('testing.html',act=act)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    return render_template('login.html')



@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    msg=""
    f2=open("mess.txt","r")
    act=f2.read()
    #print("xx="+str(act))
    f2.close()

    
        
    '''if act=="1":
        msg="Object Detected.."
        ff=open("mess.txt","w")
        ff.write("2")
        ff.close()'''
    return render_template('monitor.html',msg=act)

@app.route('/get_trash', methods=['GET', 'POST'])
def get_trash():
    msg=""
    st=""
    mess=""
    mycursor = mydb.cursor()
    f2=open("mess.txt","r")
    act=f2.read()
    #print("xx="+str(act))
    f2.close()
    print(act)
    '''cursor4 = mydb.cursor()
    cursor4.execute('SELECT * FROM admin')
    aaa = cursor4.fetchall()
    mobile=str(aaa[0][2])
    email=aaa[0][3]'''
    mobile=""
    email=""

    ff=open("mobile.txt","r")
    mobile=ff.read()
    ff.close()

    ff=open("email.txt","r")
    email=ff.read()
    ff.close()

    if act=="":
        s=1
    else:
        mess="Trash Detected, "+act
        print(mess)

        mycursor.execute("SELECT max(id)+1 FROM trash_alert")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO trash_alert(id, otype, name) VALUES (%s, %s, %s)"
        val = (maxid, '', act)
        mycursor.execute(sql,val)
        mydb.commit()

        ff=open("sms.txt","r")
        n=ff.read()
        ff.close()

        n1=0
        
        n1=int(n)
        
        if n1>0 and n1<3:
            st="1"
        n2=n1+1
        n3=str(n2)
        ff=open("sms.txt","w")
        ff.write(n3)
        ff.close()

        
    return render_template('get_trash.html',msg=mess,st=st,mess=mess,mobile=mobile,email=email)

@app.route('/alert', methods=['GET', 'POST'])
def alert():
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM trash_alert order by id desc')
    data = cursor.fetchall()

    if request.method=='GET':
        act = request.args.get('act')
        did = request.args.get('did')
        if act=="del":
            
            cursor1 = mydb.cursor()
            cursor1.execute('delete FROM trash_alert WHERE id = %s', (did, ))
            mydb.commit()   
            
    
    return render_template('alert.html',data=data, act=act)

def gen(camera):
    message=""
    n=0
    cursor4 = mydb.cursor()
    cursor4.execute('SELECT * FROM admin')
    aaa = cursor4.fetchall()
    mobile=str(aaa[0][2])
    email=aaa[0][3]
    #print(mobile)

    ff=open("mess.txt","r")
    n=ff.read()
    ff.close()
    

    
    n1=0
    if n=="":
        s=1
        
    else:
        n1=int(n)+1
                
    while True:
        frame = camera.get_frame()
        f2=open("log.txt","r")
        act=f2.read()
        #print("xx="+str(act))
        f2.close()
        if act=="20":
            s=1
            #print("begin")
            try:
                shutil.copy('getimg.jpg', 'static/trained/test.jpg')
            except:
                shutil.copy('getimg.jpg', 'static/trained/test.jpg')
        '''im = Image.open("getimg.jpg")

        for idx, box in enumerate(split(im)):
            im.crop(box).save("static/trained/test.jpg".format(idx))
        image = Image.open("static/trained/test.jpg").convert('L')
        image.save("static/trained/test.jpg")'''

        '''cursor = mydb.cursor()
        cursor.execute('SELECT * FROM store_data order by id')
        drow = cursor.fetchall()
        cursor.close()
        cutoff=15
        for rr in drow:
            print(rr[3])
            ff=rr[0]
            otype=rr[1]
            oname=rr[2]
            fn="train"+str(ff)+".jpg"
            hash0 = imagehash.average_hash(Image.open("static/trained/test.jpg")) 
            hash1 = imagehash.average_hash(Image.open("static/trained/"+fn))
            cc=hash0 - hash1
            print("cc="+str(cc))
            if hash0 - hash1 <= cutoff:
                
                message="Trash Detected, Please remove this"
                

                ff=open("mess.txt","r")
                n=ff.read()
                ff.close()

                n1=int(n)
                n2=n1+1
    
                ff=open("mess.txt","w")
                ff.write(str(n2))
                ff.close()
        
                print(message)
                mycursor = mydb.cursor()
                mycursor.execute("SELECT max(id)+1 FROM trash_alert")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1
                sql = "INSERT INTO trash_alert(id, otype, name) VALUES (%s, %s, %s)"
                val = (maxid, otype, oname)
                mycursor.execute(sql,val)
                mydb.commit()
                ##SMS
                cursor = mydb.cursor()
                cursor.execute('SELECT count(*) FROM trash_alert where status=0')
                cnt = cursor.fetchone()[0]
                if cnt>=0:
                    if n1<2:
                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name=Admin&mess="+message+"&mobile="+str(mobile)
                        webbrowser.open_new(url)
                        url="http://iotcloud.co.in/testmail/testmail1.php?email="+email+"&message="+message+"&subject=Trash"
                        webbrowser.open_new(url)
                        cursor1 = mydb.cursor()
                        cursor1.execute('update trash_alert set status=1 WHERE status=0')
                        mydb.commit()
                        print("Mail Sent....")
                        #params = urllib.parse.urlencode({'token': 'b81edee36bcef4ddbaa6ef535f8db03e', 'credit': 2, 'sender': 'RandDC', 'message':message, 'number':mobile})
                        #url = "http://pay4sms.in/sendsms/?%s" % params
                        #with urllib.request.urlopen(url) as f:
                        #    print(f.read().decode('utf-8'))
                
        
            else:
                print("aa")
                #if n1>6:
                #    ff=open("mess.txt","w")
                #    ff.write("")
                #    ff.close()
                #print(fn+"no="+str(cc))'''

                
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
