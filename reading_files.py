import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('OPENAI_API_KEY')

import requests
from pytube import YouTube
import urllib.request
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
import yt_dlp
import re

# Define headers at the top level
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

# Define headers and environment variables
os.environ['USER_AGENT'] = headers['User-Agent']

# Reading From PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("/home/aboelmagd/Downloads/ITI/Django/Django - Getting Python in The Web - lecture 1.pdf")
pages = loader.load()
print(len(pages))
page = pages[0]
print(page.page_content[:500])
print(page.metadata)

# Progress hook for YouTube download
def progress_hook(d):
    if d['status'] == 'downloading':
        percent = d['_percent_str']
        speed = d.get('_speed_str', 'N/A')
        print(f'\rDownloading... {percent} at {speed}', end='', flush=True)
    elif d['status'] == 'finished':
        print('\nDownload completed. Starting audio extraction...')

def sanitize_filename(filename):
    # Replace special characters with regular ones
    filename = filename.replace('：', ':')  # Replace Chinese colon with regular colon
    # Remove any other potentially problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename

# Reading From Youtube
print("\nLoading YouTube content...")
try:
    url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir = os.path.join(os.path.dirname(__file__), "docs/youtube/")
    os.makedirs(save_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl': os.path.join(save_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'progress_hooks': [progress_hook],
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("\nStarting YouTube download...")
            info = ydl.extract_info(url, download=True)
            
            # Sanitize the title for the output filename
            title = sanitize_filename(info['title'])
            output_file = os.path.join(save_dir, f"{title}.mp3")
            
            # Get the actual downloaded file
            downloaded_files = [f for f in os.listdir(save_dir) if f.endswith('.mp3')]
            if downloaded_files:
                actual_file = os.path.join(save_dir, downloaded_files[0])
                print(f"\nProcessing audio file: {actual_file}")

                print("\nStarting transcription (this may take a while)...")
                loader = GenericLoader(
                    FileSystemBlobLoader(os.path.dirname(actual_file), glob="*.mp3"),
                    OpenAIWhisperParser()
                )
                docs = loader.load()
                print("\nTranscription completed!")
                if docs:
                    print("\nYouTube Content:", docs[0].page_content[0:500])
                else:
                    print("No content was processed from the YouTube video")
            else:
                raise Exception("No MP3 file found after download")

    except Exception as youtube_error:
        raise Exception(f"YouTube processing error: {str(youtube_error)}")

except Exception as e:
    print(f"Error processing YouTube content: {str(e)}")
    print("Try using a different YouTube video or check your internet connection")

# Reading From Github
print("\nLoading GitHub content...")
try:
    from langchain_community.document_loaders import WebBaseLoader
    
    loader = WebBaseLoader(
        "https://raw.githubusercontent.com/basecamp/handbook/master/titles-for-programmers.md",
        verify_ssl=True,
        requests_kwargs={'headers': headers}
    )
    docs = loader.load()
    if docs:
        print("GitHub Content:", docs[0].page_content[:500])
    else:
        print("No content was loaded from GitHub")
except Exception as e:
    print(f"Error loading GitHub content: {str(e)}")

# Reading From Notion
print("\nLoading Notion content...")
try:
    notion_dir = os.path.join(os.path.dirname(__file__), "docs/Notion_DB")
    os.makedirs(notion_dir, exist_ok=True)
    
    from langchain_community.document_loaders import NotionDirectoryLoader
    loader = NotionDirectoryLoader(notion_dir)
    docs = loader.load()
    
    if docs:
        print("Notion Content:", docs[0].page_content[0:200])
        print("Notion Metadata:", docs[0].metadata)
    else:
        print("No Notion documents found in directory:", notion_dir)
        print("Please export your Notion pages to the Notion_DB directory first")
        
except Exception as e:
    print(f"Error processing Notion content: {str(e)}")