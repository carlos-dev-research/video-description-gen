from models.Interpreter import Interpreter
import gradio as gr

interpreter = Interpreter(whisper_model="openai/whisper-medium",device='cuda')

# Create the Gradio interface
iface = gr.Interface(
    fn=interpreter.gen_description,
    inputs=gr.Video(),
    outputs="text",
    title="Video Description Generator",
    description="Upload Video to Generate Description"
)

# Launch the interface
iface.launch(server_name="0.0.0.0")