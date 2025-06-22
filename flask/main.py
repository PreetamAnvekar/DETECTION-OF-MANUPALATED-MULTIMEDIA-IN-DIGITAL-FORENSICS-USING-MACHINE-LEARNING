#import all libraries
from flask import Flask,render_template,request
import numpy as np
import tensorflow as tf
import os,glob
from PIL import Image, ImageChops, ImageEnhance
import librosa
import librosa.display 
import soundfile as sf
import matplotlib.pyplot as plt
import cv2
import imageio

app=Flask("__name__")
# declaration 
name1=0
name2="progtrckr-done"
path="static/files/"
file="temp_file.jpg"
msg_file=open("image_msg.txt","r")
msg_file=msg_file.read().splitlines()
CLASS_NAMES = ["Fake","Real"]
html_page="image.html"

# declaration for image
data,f,file_name,image,predicted_class,confidence,MODEL,frame_number=0,0,0,0,0,0,0,0
image_ext=['png','jpg',"jpeg"] # valid extension for image

# declaration for audio
mfcc_scaled_features=0
audio_image_file="audio_spectrogram.jpg"
audio_ext=['wav','mp3'] # valid extension for audio

# declaration video
video_ext=['mp4',"mov"] # valid extension for video
fps=0
ext=0



def convert_to_ela_image(img, quality): #convert to image to ela image (return ela image)
    global file
    file="static/files/temp_file.jpg"
    temp_image = Image.open(img)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image.save(file,'JPEG', quality = quality)
    return ela_image

def convert_to_ela_frames(img, quality): #convert to video frames to ela image (return ela frames of video)
    path=img
    temp_image = Image.open(img)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image.save(path,'JPEG', quality = quality)
    return ela_image

def frames_to_vid(): #convert ela frame to video (return None)
    global file
    frames_dir = 'static/frames/'
    file = 'static/output_video.mp4'
    frame_files = sorted([f for f in os.listdir(frames_dir)])
    frames = [imageio.imread(os.path.join(frames_dir, f)) for f in frame_files]
    imageio.mimsave(file, frames, fps=fps)

def feature_extration(file): # convert audio to mfcc (return numpy array)
    audio,sr=librosa.load(file)
    mfccs=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=25)
    mfcc_scaled_features=np.mean(mfccs.T,axis=0)
    return mfcc_scaled_features

# Index page (common for all)   
@app.route("/")
def index(): 
    return render_template("index.html")

# Result Page (common for all)   
@app.route("/Result")           
def Result(): 
    name1[4]=name2
    return render_template(html_page,name=name1,result=predicted_class,confidence=confidence,filename=path+file_name,filename1=file,msg=msg_file[:4])

 # Prediction Page (common for all)          
@app.route("/Predicting_Results")           
def Predicting_Results():
    global predicted_class,confidence ,file 
    predictions = MODEL.predict(data)
    if ext in video_ext:
        sum=np.sum(predictions,axis=0)
        predictions=[np.divide(sum,frame_number)]
        dir="static/frames"
        file1=glob.glob(os.path.join(dir,"*"))
        for f in file1:
            os.remove(f)    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(np.max(predictions[0])*100,2)
    #confidence=52
    if confidence>45 and confidence<55:
        predicted_class="it_might_be_fake"
    name1[3]=name2
    return render_template(html_page,name=name1,redirect="/Result",result=predicted_class,confidence=confidence,filename=path+file_name,filename1=file,msg=msg_file[:4])

# Model loading (image and video)
@app.route("/Model_Loading")           
def image_Model_Loading():
    global MODEL
    MODEL = tf.keras.models.load_model("C:/Users/preet/Desktop/Final year Project/model/image_detection")
    name1[2]=name2
    return render_template(html_page,name=name1,redirect="/Predicting_Results",filename=path+file_name,filename1=file,msg=msg_file[:3])


# Image converting to ELA page
@app.route("/Image_ELA_Converting")
def Image_ELA_Converting():
    global data,image
    image=convert_to_ela_image(path+file_name,100)
    img_batch = np.expand_dims(image, 0)
    data=img_batch
    name1[1]=name2
    return render_template(html_page,name=name1,redirect="/Model_Loading",filename=path+file_name,filename1=file,msg=msg_file[:2])

# Image resizing uploaded image
@app.route("/Image")
def Image():
    import cv2
    import numpy as np

    # Load the image
    image_path = path+f.filename  # Replace with your image path
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve the result of Laplacian
    blurred = cv2.GaussianBlur(gray, (11,7),0)


    # Apply Laplacian filter to enhance edges
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Convert the Laplacian result back to uint8
    laplacian_abs = np.uint8(np.absolute(laplacian))

    # Combine the original image with the enhanced edges using addWeighted
    enhanced_image = cv2.addWeighted(image, 1, cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR), -1.5, 0)

    # Save the enhanced image
    output_path = path+"enhanced_image.jpg"  # Replace with desired output path
    cv2.imwrite(output_path, enhanced_image)
    import pytesseract

    # Path to the Tesseract executable (update this path according to your installation)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Load the image
      # Replace with your image path
    image = cv2.imread(path+"enhanced_image.jpg")


    # Convert the image to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to do OCR on the grayscale image
    custom_config =("-l eng --oem 3 --psm 11")  # Use OCR Engine mode 3 (both standard and LSTM OCR are used)
    #detected_text = pytesseract.image_to_string(gray, config=custom_config)
    detected_text = pytesseract.image_to_string(gray)
    # Print the detected text
    print("Detected text:")
    print(detected_text)
    image=Image.open(path+f.filename).resize((224,224))
    image.save(path+f.filename, quality = 100)

    return render_template(html_page,msg=detected_text,filename=path+f.filename)



# Model loading of audio
@app.route("/audio_Model_Loading")           
def audio_Model_Loading():
    global MODEL
    MODEL = tf.keras.models.load_model("C:/Users/preet/Desktop/Final year Project/model/audio5")
    name1[2]=name2
    return render_template(html_page,name=name1,redirect="/Predicting_Results",filename=path+file_name,filename1=path+file,msg=msg_file[:3])

# mfcc numpy array resizing page
@app.route("/resizing_mfcc")
def mfcc_converting_resizing():
    global data,mfcc_scaled_features
    reshaping_mfcc=mfcc_scaled_features.reshape(1,5,5,1)
    data=reshaping_mfcc
    name1[1]=name2
    return render_template(html_page,name=name1,redirect="/audio_Model_Loading",filename=path+file_name,filename1=path+file,msg=msg_file[:2])

# audio to mfcc converting page
@app.route("/Audio")
def mfcc_converting():
    import speech_recognition as sr
    r = sr.Recognizer()
    audio_file_path = path+f.filename
    print('start')
    with sr.AudioFile(audio_file_path) as source:
        audio = r.record(source)  # Record the audio from the file

        try:
            recognized_text = r.recognize_google(audio)
            print("Recognized text: " + recognized_text)
            print(type(recognized_text))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        print('end')        
    return render_template(html_page,filename=path+f.filename,msg=recognized_text)






#  Extract frames and resizing frames of video page
@app.route("/Frames_resizing")
def Frames_resizing():
    global frame_number,fps
    frame_number=0
    cap= cv2.VideoCapture(path+file_name)
    while True:
                    # Read the next frame
        ret, frame = cap.read()
        if not ret:
            fps = cap.get(cv2.CAP_PROP_FPS)
            break

                    # Save the frame as an image
        cv2.imwrite(f'static/frames/image{frame_number}.jpg', frame)
        frame_number += 1
                # Release the video file
    cap.release()
    name1[0]=name2
    return render_template(html_page,name=name1,redirect="/video_Ela_converting",filename=path+file_name,msg=msg_file[:1])

# upoaded file is analysis formate and redirect to specific page
@app.route('/upload', methods = ['POST'])  
def upload():  
    global f,file_name,name1,html_page,msg_file,ext
    file_name=None
    
    if request.method == 'POST':  
        name1=["progtrckr-todo","progtrckr-todo","progtrckr-todo","progtrckr-todo","progtrckr-todo"]
        f = request.files['file']
        f.save(path+f.filename) 
        file_name=f.filename 
        ext=f.filename.split(".")[-1]
        if ext.lower() in image_ext:
            html_page="image.html"
            msg_file=open("image_msg.txt","r")
            msg_file=msg_file.read().splitlines()
            name1[0]=name2
            return render_template(html_page,name=name1,redirect="/Image")
        
        if ext.lower() in audio_ext: 
            html_page="audio.html"
            msg_file=open("audio_msg.txt","r")
            msg_file=msg_file.read().splitlines()
            return render_template(html_page,name=name1,redirect="/Audio",filename=path+file_name)
        
        if ext.lower() in video_ext: 
            html_page="video.html"
            msg_file=open("video_msg.txt","r")
            msg_file=msg_file.read().splitlines()
            return render_template(html_page,name=name1,redirect="/Frames_resizing",filename=path+file_name)
        else:
            error_msg="Invalid File Extension"
            return render_template("index.html",error_msg=error_msg)
    else:
        error_msg="Please Upload file"
        return render_template("index.html",error_msg=error_msg)

        
        

if __name__=="__main__":
    app.run(debug=True)