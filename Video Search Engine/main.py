
import moviepy.editor as vdE # for playing with the video
import speech_recognition as sr # for playing with the speech into text 
from pydub import AudioSegment # for playing with the mp3 files rather than wav 
import glob # for getting the name of the files whose are mp4 
import os 

def main() :
    """___________________________Get seaeched str and videos array__________________________________"""
    to_search = input("Enter the video you want to search : \n")
    videos = glob.glob("*.mp4")
    videosArr =[]
    searchedVideosArr=[]
    flag = False
    for value in videos :
        index = value.find(".mp4")
        videosArr.append(value[0:index])
        
        """___________________________Convert the video into audio__________________________________"""
    for key, value in enumerate(videosArr) :
        # Loading and Extracting audio
        video_clip = vdE.VideoFileClip(f"{value}.mp4")
        audio_clip= video_clip.audio
        
        # Writing and making audio file
        audio_clip.write_audiofile(f"{value}.mp3")
        video_clip.close()
        audio_clip.close()
        """___________________________Convert the mp3 into wav__________________________________"""
        audio_file = f"{value}.mp3"
        audio =  AudioSegment.from_mp3(audio_file)
        wav_file = f"{value}.wav"
        audio.export(wav_file, format="wav")
        os.system(f"rm {value}.mp3")
        
        """___________________________Convert the audio into text__________________________________"""
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file) as source :
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        os.system(f"rm {value}.wav")    
        with open(f"{value}.txt", "w") as fw : 
            fw.write(text)
            
        """___________________________Checking searching str present in text__________________________________"""
        with open(f"{value}.txt", "r") as fr :
            for lnum, line in enumerate(fr, start=1) :
                if to_search in line :
                    searchedVideosArr.append(1)
                else :
                    searchedVideosArr.append(0) 
            os.system(f"rm {value}.txt")

        """___________________________Printing all those videos whose contain that text__________________________________"""
    for k, v in enumerate(searchedVideosArr) : 
        if v == 1 :
            print(f"{videosArr[k]}.mp4")
            flag =True
        else : 
            pass
    if(not(flag)):
        print(f"{to_search} is not present any of the given videos")
    else : 
        print(f"{to_search} is present in these vidoes")
        
        
if __name__ == '__main__' :
    main()