
import moviepy.editor as vdE
import speech_recognition as sr
from pydub import AudioSegment
import os

def main() :
    to_search = input("Enter the video you want to search : \n")
    """___________________________Convert the video into audio__________________________________"""
    # Loading and Extracting audio
    video_clip = vdE.VideoFileClip("v1.mp4")
    audio_clip= video_clip.audio
    
    # Writing and making audio file
    audio_clip.write_audiofile("v1.mp3")
    video_clip.close()
    audio_clip.close()
    """___________________________Convert the mp3 into wav__________________________________"""
    audio_file = "v1.mp3"
    audio =  AudioSegment.from_mp3(audio_file)
    wav_file = "v1.wav"
    audio.export(wav_file, format="wav")
    os.system("rm v1.mp3")
    
    """___________________________Convert the audio into text__________________________________"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source :
        audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data)
    os.system("rm v1.wav")    
    with open("v1.txt", "w") as fw : 
        fw.write(text)
        
    """___________________________Checking searching str present in text__________________________________"""
    with open("v1.txt", "r") as fr :
        for lnum, line in enumerate(fr, start=1) :
            if to_search in line :
                print(to_search, "is present in the video")
            else :
                print(to_search, "is not present in the video")
if __name__ == '__main__' :
    main()