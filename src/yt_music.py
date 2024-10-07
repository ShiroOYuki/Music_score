import yt_dlp
import os
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg") 

# https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#embedding-examples

class Downloader:
    @classmethod
    def download(cls, url):
        opts = {
            'format': 'm4a/bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
            'paths': {
                'home': './data/music/temp'
            },
            'outtmpl': {
                'default': '%(id)s.%(ext)s'
            }
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            print("Title:", info.get("title"))
            filepath = ydl.prepare_filename(info, outtmpl=opts['outtmpl']['default'])
            abs_path = os.path.abspath(filepath)
            cls.m4a_to_mp3(abs_path)
            
    @classmethod
    def m4a_to_mp3(cls, input_file: str):
        audio = AudioSegment.from_file(input_file, format="m4a")
        
        output_path = os.path.abspath("./data/music/download")
        filename = os.path.basename(input_file).removesuffix(".m4a") + ".mp3"
        
        output_path = os.path.join(output_path, filename)
        
        audio.export(output_path, format="mp3")
        print(f"[output] {output_path}")
        os.remove(input_file)
        return output_path

        
if __name__ == "__main__":
    # download("https://www.youtube.com/watch?v=t3kOeUsnocg")
    while True:
        try:
            Downloader.download(input("URL: "))
        except:
            break