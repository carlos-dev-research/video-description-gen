from transformers import pipeline
import ollama
from PIL import Image
import numpy as np
import cv2
import io

class Interpreter:
    def __init__(self,whisper_model="openai/whisper-medium",device='cuda'):
        if device == 'cuda':
            try:
                self.audio_pipe = pipeline("automatic-speech-recognition", model=whisper_model, device = 'cuda')
            except:
                print("No cuda found, falling back to cpu")
                self.audio_pipe = pipeline("automatic-speech-recognition", model=whisper_model, device = 'cpu')
        elif device=='cpu':
            self.audio_pipe = pipeline("automatic-speech-recognition", model=whisper_model, device = 'cpu')
        else:
            print(f"Device {device} not found, falling back to cpu")
    


    def generate_text(self,instruction, images):
        byte_images = []
        for img in images:
            if isinstance(img, Image.Image):
                byte_images.append(pil_to_byte_array(img))
            elif isinstance(img, np.ndarray):
                byte_images.append(pil_to_byte_array(numpy_to_pil(img)))
            else:
                raise ValueError("Unsupported image format. Use PIL or NumPy images.")
        
        result = ollama.generate(
            model='llava',
            prompt=instruction,
            images=byte_images,
            stream=False
        )['response']
            
        return result
    
    def analyze_video(self,video_path, fps, instruction):
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate / fps)
        frame_count = 0
        descriptions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                description = self.generate_text(instruction, [pil_img])
                descriptions.append(description)
            
            frame_count += 1
        
        cap.release()
        return " ".join(descriptions)

    def interpret_images(self,video_path):
        fps = 1/90
        instruction = 'Describe briefly what you see in the image'
        return  self.analyze_video(video_path, fps, instruction)
    
    # Define the transcription function
    def transcribe_audio(self,audio_file):
        transcription = self.audio_pipe(audio_file)
        return transcription['text']

    def gen_description(self,video_file):
        image_description = self.interpret_images(video_file)
        transcription = self.transcribe_audio(video_file)
        instruction = f"""Visual Description of Images: "{image_description}"

        Transcription of Audio: {transcription}

        You are an assistant who generates descriptions of videos to be used in Youtube, you use captures of the video and transcriptions,
        Create a description of the Video using the Visual Description of the Images and the Transcription of Audio
        Video Description: """
        return self.generate_text(instruction,[])
    
def pil_to_byte_array(pil_image):
    byte_array = io.BytesIO()
    pil_image.save(byte_array, format='PNG')
    return byte_array.getvalue()

def numpy_to_pil(np_image):
    if np_image.dtype == np.uint8:
        return Image.fromarray(np_image)
    elif np_image.dtype == np.float32 or np_image.dtype == np.float64:
        return Image.fromarray((np_image * 255).astype(np.uint8))