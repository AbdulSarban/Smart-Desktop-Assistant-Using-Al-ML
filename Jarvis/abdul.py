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
u=time.time()


# Things to Add
# Security take owners voice only

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
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
    # It takes microphone input from the user and returns string output

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
        # print(e)
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
    # Use a raw string literal (r"your_path_here") or double backslashes ("your_path_here")
    book_path = r'C:\Users\Admin\Downloads\Web Dev Syllabus.pdf'
    
    try:
        book = open(book_path, 'rb')
        
        # Use PdfReader instead of PdfFileReader
        pdfReader = PyPDF2.PdfReader(book)
        
        pages = len(pdfReader.pages)  # Use len(pdfReader.pages) to get the number of pages
        speak(f"Total number of pages in this book: {pages}")
        speak("Sir, please enter the page number I have to read")
        pg = int(input("Please enter the page number:"))
        
        # Check if the page number is valid
        if 0 <= pg < pages:
            page = pdfReader.pages[pg]  # Access the page using pdfReader.pages
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
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from all the paragraphs on the page
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
            
            # Continue listening until a stop word is spoken or a timeout occurs
            start_time = time.time()
            timeout_seconds = 60  # Set a timeout of 60 seconds

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
    # Specify the directory path
    output_directory = "D:\\output_image"
    
    # Ensure the directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create the image
    image = Image.new("RGB", (300, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Use a default font
    font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (image.width - text_width) // 2
    y = (image.height - text_height) // 2

    draw.text((x, y), text, font=font, fill=(0, 0, 0))

    # Save the image to the specified directory
    image_path = os.path.join(output_directory, "output_image.png")
    image.save(image_path)
    
    speak(f"Image generated. Check the {image_path} file.")

def open_vscode():
    vscode_path = r'C:\Users\Admin\AppData\Local\Programs\Microsoft VS Code\Code.exe'
    subprocess.Popen([vscode_path])

def write_to_temp_file(text, file_type):
    # Create a temporary file with the specified content and file extension
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False, mode="w", encoding="utf-8")
    temp_file.write(text)
    temp_file.close()
    return temp_file.name




def vscode():
    speak("Hello! I will open Visual Studio Code and write down what you say.")
    
    # Ask the user for the file type
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
            
            # Continue listening until a stop word is spoken or a timeout occurs
            start_time = time.time()
            timeout_seconds = 60  # Set a timeout of 60 seconds

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
    # Use os.path.join to build the full path to wordpad.exe
    wordpad_path = os.path.join(os.environ['SystemRoot'], 'System32', 'write.exe')
    subprocess.Popen([wordpad_path])
def normalize_word(word):
    # Normalize spoken words by converting to lowercase and removing trailing characters
    return word.lower().rstrip(".,?!")

def write_to_wordpad(text):
    # Explicitly handle the spoken word "comma" and "space"
    text = text.replace('comma', ',')
    text = text.replace('space', ' ')
    text = text.replace('\n', ' \n')  # Add a space before new lines
    pyautogui.typewrite(text)


def wordpad():
    speak("Hello! I will open WordPad and write down what you say.")
    open_wordpad()
    
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            speak("Please start speaking. You can say 'stop' to finish.")
            recognizer.adjust_for_ambient_noise(source)
            
            # Continue listening until a stop word is spoken or a timeout occurs
            start_time = time.time()
            timeout_seconds = 60  # Set a timeout of 60 seconds

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



def image():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        speak("Hello! I'm ready to generate an image. Please give a command.")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            speak(f"You said: {command}")

            if "generate image" in command:
                generate_image("Hello, Abdul!")

        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand your command.")
        except sr.RequestError as e:
            speak(f"Error connecting to Google Speech Recognition service: {e}")




def perform_math_operations():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        speak("Hello! I'm ready to perform mathematical calculations. Please ask a question.")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source)
            question = recognizer.recognize_google(audio).lower()
            speak(f"You asked: {question}")
            print(f"You asked: {question}")
            

            # Perform mathematical calculation
            answer = calculate(question)
            print(f"The answer is: {answer}")

        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand your question.")
        except sr.RequestError as e:
            speak(f"Error connecting to Google Speech Recognition service: {e}")
    
# def summarize_text(text):
#     # Use gensim's summarize function
#     summary = summarize(text)
#     return summary
def pdf_reader1():
    # Take PDF file path as input from the user
    book_path = input("Enter the path to the PDF file: ")

    try:
        book = open(book_path, 'rb')
        pdfReader = PyPDF2.PdfReader(book)

        pages = len(pdfReader.pages)
        speak(f"Total number of pages in this book: {pages}")
        speak("Sir, please enter the page number I have to read")
        pg = int(input("Please enter the page number:"))

        # Check if the page number is valid
        if 0 <= pg < pages:
            page = pdfReader.pages[pg]
            text = page.extract_text()
            speak(text)

            # # Summarize the text
            # summary = summarize_text(text)
            
            # # Speak the summary
            # speak("Here is a summary of the text:")
            # speak(summary)
        else:
            speak("Invalid page number. Please enter a valid page number.")
    except FileNotFoundError:
        speak("File not found. Please enter a valid file path.")
    except Exception as e:
        speak(f"An error occurred: {str(e)}")



def news():
    main_url = 'url + api'

    main_page = requests.get(main_url).json()

    articles = main_page["articles"]

    head = []
    day=["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"]
    for ar in articles:
        head.append(ar["title"])
    for i in range (len(day)):

        speak(f"today's{day[i]} news is: {head[i]}")


if __name__ == "__main__":
    wishMe()
    while True:
        # if 1:
        for i in range(10):
            query = takeCommand().lower()

            # Logic for executing tasks based on query
            if 'wikipedia' in query:
                speak('Searching Wikipedia...')
                query = query.replace("wikipedia", "")
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                print(results)
                speak(results)

            elif 'open youtube' in query:
                webbrowser.open("youtube.com")
                speak("YouTube Opened")

            elif 'open google' in query:
                webbrowser.open("google.com")
                speak("Google Opened")

            elif 'open stackoverflow' in query:
                webbrowser.open("stackoverflow.com")
                speak("Stack Owerflow Opened")

            elif 'play music' in query:
                music_dir = 'D:\\Songs'
                songs = os.listdir(music_dir)
                print(songs)
                os.startfile(os.path.join(music_dir, songs[0]))
                speak("Asking Windows Player to play  ")

            elif 'the time' in query:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                speak(f"Sir, the time is {strTime}")

            elif 'open code' in query:
                codePath = "C:\\Users\\Haris\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
                os.startfile(codePath)
            elif "tell me a joke" in query:
                joke = pyjokes.get_joke()
                speak(joke)
            elif "take screenshot" in query:
                speak("Sir, Please tell me the name for the screenshot file")
                name = takeCommand().lower()
                speak("Taking Screenshot...!")
                time.sleep(2)
                img = pyautogui.screenshot()
                img.save(f"{name}.png")
                speak("ScreenShot Saved...!")

            elif "tell me news" in query:
                speak("Please wait Sir,fetching the latest news")
                news()
            elif "how are you jarvis" in query or 'how are you' in query:
                stMsgs = ['Just doing my thing!', 'I am fine!', 'Nice!', 'I am nice and full of energy','i am okey ! How are you']
                ans_q = random.choice(stMsgs)
                speak(ans_q)  
                ans_take_from_user_how_are_you = takeCommand()
                if 'fine' in ans_take_from_user_how_are_you or 'happy' in ans_take_from_user_how_are_you or 'okey' in ans_take_from_user_how_are_you:
                    speak('okey..')  
                elif 'not' in ans_take_from_user_how_are_you or 'sad' in ans_take_from_user_how_are_you or 'upset' in ans_take_from_user_how_are_you:
                    speak('oh sorry..')  
            elif 'who make you' in query or 'created you' in query or 'develop you' in query:
                ans_m = " For your information Aabdul and Adnan Created me ! I give Lot of Thannks to Him "
                print(ans_m)
                speak(ans_m)
            elif "who are you" in query or "about you" in query or "your details" in query:
                about = "I am Acro an A I based computer program but i can help you lot like a your close friend ! i promise you ! Simple try me to give simple command ! like playing music or video from your directory i also play video and song from web or online ! i can also entain you i so think you Understand me ! ok Lets Start "
                print(about)
                speak(about)
            elif "hello" in query or "hello Acro" in query:
                hel = "Hello Sir ! How May i Help you.."
                print(hel)
                speak(hel)
            elif "your name" in query or "sweat name" in query:
                na_me = "Thanks for Asking my name my self ! jaarvis"  
                print(na_me)
                speak(na_me)
            elif "you feeling" in query:
                print("feeling Very sweet after meeting with you")
                speak("feeling Very sweet after meeting with you")
            elif 'search on google' in query:
                speak("What would you like to search on google?")
                cm = takeCommand().lower()
                webbrowser.open(cm)
            
            elif 'image' in query:
                speak("what should i genrate")
                if __name__ == "__main__":
                    image()


                speak("searching ")
            elif 'play' in query:
                song = query.replace('play', '')
                speak('playing ' + song)
                pywhatkit.playonyt(song)
            elif 'send message' in query:
                speak('Okay! Sending Message...')
                import datetime

                current_time = datetime.datetime.now()
                next_minute = (current_time.minute // 1) * 1 + 2
                next_hour = current_time.hour if next_minute == 0 else current_time.hour

                pywhatkit.sendwhatmsg("+919623946129", "Hello, this is a test message.", next_hour, next_minute)
            
            elif 'search' in query:
                speak('What do you want to search for?') #stop for a while to take the next cammand after the search
                search = takeCommand()
                url = 'https://google.com/search?q=' + search
                webbrowser.open_new_tab(url)
            elif 'change voice' in query:
                if 'female' in query:
                    engine.setProperty('voice', voices[0].id)
                else:
                    engine.setProperty('voice', voices[1].id)
                speak("Hello Sir, I have switched my voice. How is it?")

            elif 'website' in query:
                speak("what should i read")
                if __name__ == "__main__":
                    # Example usage:
                    web_url = input("Enter the URL of the web page: ")
                    page_text = read_text_from_webpage(web_url)

                    if page_text:
                        print("Text extracted from the web page:\n", page_text)
                        speak(page_text)
                    else:
                        print("Failed to extract text from the web page.")
            

            elif 'notepad' in query:
                speak("what should i write on notepad")
                if __name__ == "__main__":
                    main()

            elif 'give summery' in query:
                pdf_reader1()

            elif 'calculate' in query:
                speak("what sould i calculate")
                if __name__ == "__main__":
                    perform_math_operations()
            

            elif 'wordpad' in query:
                speak("what should i do")
                if __name__ == "__main__":
                    wordpad()


            elif 'vs code' in query:
                speak("what should i do")
                if __name__ == "__main__":
                    vscode()


        
        




            

                
            elif "remember that" in query:
                speak("What should I remember")
                data = takeCommand()
                speak("You said me to remember that" + data)
                print("You said me to remember that " + str(data))
                remember = open("data.txt", "w")
                remember.write(data)
                remember.close()
            elif "read pdf" in query:
                pdf_reader()

            elif "do you remember anything" in query:
                remember = open("data.txt", "r")
                speak("You told me to remember that" + remember.read())
                print("You told me to remember that " + str(remember))

                

            elif 'email to abdul' in query:
                try:
                    speak("What should I say?")
                    content = takeCommand()
                    to = "sarbanabdul@gmail.com"
                    sendEmail(to, content)
                    speak("Email has been sent!")
                except Exception as e:
                    print(e)
                    speak(
                        "Sorry my friend abdul bhai. I am not able to send this email")

v=time.time()
speak(v-u)
