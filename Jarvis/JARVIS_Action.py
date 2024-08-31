import csv
import time
import requests
import smtplib
import random
import pyjokes
import PyPDF2
from PIL import Image, ImageDraw, ImageFont
import re
import subprocess
from bs4 import BeautifulSoup
import pywhatkit
import pyautogui
from requests import get
import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import tempfile

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
        elif "square" in expression:
            cleaned_expression = cleaned_expression.replace("square", "**2")
        elif "square root" in expression:
            cleaned_expression = cleaned_expression.replace("square root", "**0.5")
        result = eval(cleaned_expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def generate_image(text):
    output_directory = "D:\\output_image"
    os.makedirs(output_directory, exist_ok=True)
    image = Image.new("RGB", (300, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (image.width - text_width) // 2
    y = (image.height - text_height) // 2
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    image_path = os.path.join(output_directory, "output_image.png")
    image.save(image_path)
    speak(f"Image generated. Check the {image_path} file.")


def open_vscode():
    vscode_path = r'C:\Users\Admin\AppData\Local\Programs\Microsoft VS Code\Code.exe'
    subprocess.Popen([vscode_path])


def write_to_temp_file(text, file_type):
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False, mode="w", encoding="utf-8")
    temp_file.write(text)
    temp_file.close()
    return temp_file.name


def vscode():
    speak("Hello! I will open Visual Studio Code and write down what you say.")
    speak("What type of file would you like to create? For example, you can say 'text' or 'python'.")
    file_type = input("Enter the file type: ").lower()
    if file_type not in ['text', 'python']:
        speak(f"Unsupported file type: {file_type}. Defaulting to 'text' file.")
        file_type = 'text'
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
                    file_path = write_to_temp_file(text, file_type)
                    speak(f"I've written it in a {file_type} file in Visual Studio Code.")
                    open_vscode(file_path)
                except sr.UnknownValueError:
                    speak("Sorry, I couldn't understand what you said.")
                except sr.RequestError as e:
                    speak(f"Error connecting to Google Speech Recognition service: {e}")
            speak("Session ended.")
    except KeyboardInterrupt:
        speak("Manually interrupted.")
    except Exception as e:
        speak(f"An error occurred: {str(e)}")


def open_wordpad():
    wordpad_path = os.path.join(os.environ['SystemRoot'], 'System32', 'write.exe')
    subprocess.Popen([wordpad_path])


def normalize_word(word):
    return word.lower().rstrip(".,?!")


def write_to_wordpad(text):
    text = text.replace('comma', ',')
    text = text.replace('space', ' ')
    text = text.replace('\n', ' \n')
    pyautogui.typewrite(text)


def wordpad():
    speak("Hello! I will open WordPad and write down what you say.")
    open_wordpad()
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
                    write_to_wordpad(text)
                    speak("I've written it in WordPad.")
                except sr.UnknownValueError:
                    speak("Sorry, I couldn't understand what you said.")
                except sr.RequestError as e:
                    speak(f"Error connecting to Google Speech Recognition service: {e}")
            speak("Session ended.")
    except KeyboardInterrupt:
        speak("Manually interrupted.")
    except Exception as e:
        speak(f"An error occurred: {str(e)}")


def read_commands_from_csv(file_path):
    commands = []
    try:
        with open(file_path, mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                commands.append(row)
    except FileNotFoundError:
        speak(f"CSV file not found at the specified path: {file_path}")
    except Exception as e:
        speak(f"An error occurred while reading the CSV file: {str(e)}")
    return commands


def execute_commands(commands):
    for command in commands:
        if command[0] == 'iloveyou':
            open_notepad()
        elif command[0] == 'vscode':
            vscode()
        elif command[0] == 'word':
            wordpad()
        elif command[0] == 'cal':
            if len(command) > 1:
                result = calculate(command[1])
                speak(result)
            else:
                speak("No expression provided to calculate.")
        elif command[0] == 'generate_image':
            if len(command) > 1:
                generate_image(command[1])
            else:
                speak("No text provided to generate an image.")
        elif command[0] == 'thanks':
            pdf_reader()
        else:
            speak(f"Unknown command: {command[0]}")


if __name__ == "__main__":
    wishMe()
    while True:
        commands = read_commands_from_csv('D:\Jarvis_Project\predictions.csv')
        execute_commands(commands)
        time.sleep(10)  # Check for new commands every 10 seconds
