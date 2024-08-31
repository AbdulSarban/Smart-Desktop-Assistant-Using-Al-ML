import cv2
import numpy as np
from tkinter import Tk, Label, Button
from PIL import Image as PILImage, ImageTk
import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import time
import random
import pyjokes
import PyPDF2
import requests
import subprocess
from bs4 import BeautifulSoup
import pywhatkit
import pyautogui
import re
import tempfile



from http import HTTPStatus
import time
import requests
import smtplib
import random
import pyjokes
# import pytorch
import PyPDF2
#from transformers import BertForQuestionAnswering, BertTokenizer
#import torch
#from gensim.summarization import summarize
import tempfile
from PIL import Image, ImageDraw, ImageFont
import re
import subprocess
from bs4 import BeautifulSoup
import pywhatkit
import pyautogui
from pywhatkit import sendwhatmsg
from requests import get
import pyttsx3  # pip install pyttsx3
import speech_recognition as sr  # pip install speechRecognition
import datetime
import wikipedia  # pip install wikipedia
import webbrowser
import os
import smtplib





engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am Jarvis Sir. Please tell me how may I help you")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
    except Exception as e:
        print("Say that again please...")
        return "None"
    return query

def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('youremail@gmail.com', 'your-password')
    server.sendmail('youremail@gmail.com', to, content)
    server.close()

def pdf_reader():
    book_path = r'C:\Users\Admin\Downloads\Web Dev Syllabus.pdf'
    try:
        book = open(book_path, 'rb')
        pdfReader = PyPDF2.PdfReader(book)
        pages = len(pdfReader.pages)
        speak(f"Total number of pages in this book: {pages}")
        speak("Sir, please enter the page number I have to read")
        pg = int(input("Please enter the page number:"))
        if 0 <= pg < pages:
            page = pdfReader.pages[pg]
            text = page.extract_text()
            speak(text)
        else:
            speak("Invalid page number. Please enter a valid page number.")
    except FileNotFoundError:
        speak(f"File not found at the specified path: {book_path}")
    except Exception as e:
        speak(f"An error occurred: {str(e)}")
    finally:
        try:
            book.close()
        except:
            pass

def read_text_from_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        page_text = '\n'.join([paragraph.get_text() for paragraph in paragraphs])
        return page_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from the web page: {e}")
        return None

def open_notepad():
    subprocess.Popen(['notepad.exe'])

def write_to_notepad(text):
    pyautogui.typewrite(text)

def main():
    speak("Hello! I will open Notepad and write down what you say.")
    open_notepad()
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            speak("Please start speaking. You can say 'stop' to finish.")
            recognizer.adjust_for_ambient_noise(source)
            start_time = time.time()
            timeout_seconds = 60
            while time.time() - start_time < timeout_seconds:
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio)
                    speak(f"You said: {text}")
                    if text.lower() == 'stop':
                        speak("Stopping.")
                        break
                    write_to_notepad(text)
                    speak("I've written it in Notepad.")
                except sr.UnknownValueError:
                    speak("Sorry, I couldn't understand what you said.")
                except sr.RequestError as e:
                    speak(f"Error connecting to Google Speech Recognition service: {e}")
            speak("Session ended.")
    except KeyboardInterrupt:
        speak("Manually interrupted.")
    except Exception as e:
        speak(f"An error occurred: {str(e)}")

def calculate(expression):
    cleaned_expression = re.sub(r'[^0-9+\-*/().]', '', expression.replace(" ", ''))
    try:
        if "multiply" in expression:
            cleaned_expression = cleaned_expression.replace("multiply", "*")
        elif "divide" in expression:
            cleaned_expression = cleaned_expression.replace("divide", "/")
        result = eval(cleaned_expression)
        return result
    except Exception as e:
        return str(e)

def take_screenshot():
    screenshot = pyautogui.screenshot()
    temp_dir = tempfile.gettempdir()
    screenshot_path = f"{temp_dir}/screenshot.png"
    screenshot.save(screenshot_path)
    speak("Screenshot taken and saved.")
    return screenshot_path


def owner_verification():
    known_image_path = 'abdul.jpg'
    known_image = cv2.imread(known_image_path)

    # Convert the known image to grayscale and detect faces
    known_image_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    known_faces = face_cascade.detectMultiScale(known_image_gray, scaleFactor=1.1, minNeighbors=5)

    if len(known_faces) == 0:
        print("No face found in the known image.")
        return

    # Use the first detected face in the known image
    (x, y, w, h) = known_faces[0]
    known_face = known_image_gray[y:y+h, x:x+w]

    cap = cv2.VideoCapture(0)
    verified = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]

            # Resize the detected face to match the known face size
            resized_face = cv2.resize(face, (known_face.shape[1], known_face.shape[0]))

            # Compare the detected face with the known face
            diff = cv2.absdiff(known_face, resized_face)
            if np.mean(diff) < 50:
                verified = True
                break

        if verified:
            speak("Owner Verified! Starting JARVIS...")
            break

        cv2.imshow('Face Verification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not verified:
        speak("Verification failed. Exiting.")
        exit()

def detect_objects():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    owner_verification()
    wishMe()
    while True:
        query = takeCommand().lower()

        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            speak(results)
        elif 'open youtube' in query:
            webbrowser.open("youtube.com")
        elif 'open google' in query:
            webbrowser.open("google.com")
        elif 'open stackoverflow' in query:
            webbrowser.open("stackoverflow.com")
        elif 'play music' in query:
            music_dir = 'D:\\Music'
            songs = os.listdir(music_dir)
            os.startfile(os.path.join(music_dir, songs[0]))
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"Sir, the time is {strTime}")
        elif 'open code' in query:
            codePath = "C:\\Users\\YourUserName\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
            os.startfile(codePath)
        elif 'send email' in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                to = "youremail@gmail.com"
                sendEmail(to, content)
                speak("Email has been sent!")
            except Exception as e:
                speak("Sorry Sir. I am not able to send this email")
        elif 'read pdf' in query:
            pdf_reader()
        elif 'search on google' in query:
            speak("What should I search?")
            search_query = takeCommand().lower()
            webbrowser.open(f"https://www.google.com/search?q={search_query}")
        elif 'calculate' in query:
            speak("Please tell me the expression to calculate.")
            expression = takeCommand().lower()
            result = calculate(expression)
            speak(f"The result is {result}")
        elif 'take screenshot' in query:
            screenshot_path = take_screenshot()
            speak(f"Screenshot saved at {screenshot_path}")
        elif 'detect objects' in query:
            speak("Starting object detection...")
            detect_objects()
