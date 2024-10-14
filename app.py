import os
import time
from typing import Any
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline
import ssl
import httpx
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import time
import ollama
import random
import subprocess
from gtts import gTTS

from utils.custom import css_code

load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Azure OpenAI specific settings
# GENERATION_MODEL = os.getenv("GENERATION_MODEL")
# OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
# OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")

LOCAL_MODEL_PATH = "./models"

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def progress_bar(amount_of_time: int) -> None:
    """
    A simple progress bar that increases over time,
    then disappears when it reaches completion
    :param amount_of_time: time taken
    :return: None
    """
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

def generate_text_from_image(url: str) -> str:
    try:
        image_to_text: Any = pipeline("image-to-text", model=LOCAL_MODEL_PATH)
        print("Model loaded successfully")
        generated_text: str = image_to_text(url)[0]["generated_text"]
        generated_text = generated_text.lstrip("arafed").strip()
        generated_text = generated_text.capitalize()
    except Exception as e:
        st.error(f"An error occurred while generating text from the image: {str(e)}")
        return f"Error: Unable to generate text from image. {str(e)}"

    print(f"IMAGE INPUT: {url}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text

# def generate_story_from_text(scenario: str) -> str:
#     prompt_template: str = f"""
#     You are a talented story teller who can create a story from a simple narrative.
#     Create a story based on the following scenario, which describes an image. The story should:
#     - Be maximum 100 words long
#     - Be coherent and directly related to the given scenario
#     - Include at least one relevant quote
#     - Focus solely on the content described in the scenario
#     - Avoid any unnecessary information
#     CONTEXT: {scenario}
#     STORY: (including at least one quote):
#     """

#     try:
#         response = ollama.chat(model="llama2", messages=[
#             {
#                 "role": "user",
#                 "content": prompt_template
#             }
#         ])
#         generated_story = response['message']['content']
#     except Exception as e:
#         st.error(f"An error occurred while generating the story: {str(e)}")
#         return f"Error: Unable to generate story. {str(e)}"

#     print(f"TEXT INPUT: {scenario}")
#     print(f"GENERATED STORY OUTPUT: {generated_story}")
#     return generated_story


def generate_story_from_text(scenario: str) -> str:
    prompt_template: str = f"""
    You are a talented screenwriter who can create engaging dialogue-based stories with morals from simple scene descriptions.
    Based on the following scenario, which describes an image or a scene from a storyboard, create a short story told primarily through dialogue. The story should:
    - Be approximately 200-250 words long
    - Consist mainly of dialogue, with minimal narrative description
    - Be coherent and directly related to the given scenario
    - Include character names before each line of dialogue
    - Capture the mood and context of the scene(s)
    - Include brief action descriptions in parentheses where necessary
    - Focus on the content described in the scenario, but feel free to expand it into a short sequence if appropriate
    - Have a clear beginning, middle, and end, even if brief
    - Conclude with a clear moral or lesson learned
    - Format the dialogue so that each character's line starts with their name in ALL CAPS, followed by a colon and their dialogue
    
    SCENARIO: {scenario}
    
    DIALOGUE-BASED STORY WITH MORAL:
    """

    try:
        response = ollama.chat(model="llama2", messages=[
            {
                "role": "user",
                "content": prompt_template
            }
        ])
        generated_story = response['message']['content']
        
        # Post-process the story to ensure proper formatting
        lines = generated_story.split('\n')
        formatted_lines = []
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                formatted_lines.append(f"{parts[0].strip().upper()}: {parts[1].strip()}")
            else:
                formatted_lines.append(line)
        
        formatted_story = '\n'.join(formatted_lines)
        
        # Ensure there's a moral at the end
        # if "Moral:" not in formatted_story:
        #     formatted_story += "\n\nMoral: [The moral should be inferred from the story and added here.]"
        
    except Exception as e:
        st.error(f"An error occurred while generating the dialogue-based story: {str(e)}")
        return f"Error: Unable to generate dialogue-based story. {str(e)}"

    return formatted_story


def generate_speech_from_text(message: str, max_retries: int = 5, initial_delay: float = 1, max_delay: float = 60) -> None:
    API_URL: str = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers: dict[str, str] = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payloads: dict[str, Any] = {
        "inputs": message
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payloads, verify=False)
            response.raise_for_status()
            
            if response.headers.get('content-type') == 'audio/flac':
                with open("generated_audio.flac", "wb") as file:
                    file.write(response.content)
                print("Audio file generated successfully using LJ Speech model.")
                return
            else:
                print(f"Unexpected content type: {response.headers.get('content-type')}")
                print(f"Response content: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                delay = min(initial_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Falling back to gTTS.")
                use_gtts_fallback(message)
                return

    print(f"Failed to generate speech after {max_retries} attempts. Falling back to gTTS.")
    use_gtts_fallback(message)

def use_gtts_fallback(message: str) -> None:
    tts = gTTS(text=message, lang='en')
    tts.save("generated_audio.mp3")
    # Convert mp3 to flac if needed
    os.system("ffmpeg -i generated_audio.mp3 generated_audio.flac")
    os.remove("generated_audio.mp3")
    print("Audio file generated using gTTS fallback.")

def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")

    st.markdown(css_code, unsafe_allow_html=True)

    with st.sidebar:
        st.write("Story Generator App")

    st.header("Image-to-Story Converter")
    uploaded_file: Any = st.file_uploader("Please choose a file to upload", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data: Any = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image",
                 use_column_width=True)
        progress_bar(100)
        scenario: str = generate_text_from_image(uploaded_file.name)
        if not scenario.startswith("Error:"):
            story: str = generate_story_from_text(scenario)
            
            with st.spinner("Generating audio..."):
                generate_speech_from_text(story)

            with st.expander("Generated Image scenario"):
                st.write(scenario)
            with st.expander("Generated short story"):
                st.write(story)

            if os.path.exists("generated_audio.flac"):
                st.success("Audio generated successfully using LJ Speech model.")
                st.audio("generated_audio.flac")
            elif os.path.exists("generated_audio.mp3"):
                st.warning("Audio generated using fallback gTTS method.")
                st.audio("generated_audio.mp3")
            else:
                st.error("No audio file was generated.")
        else:
            st.error(scenario)

if __name__ == "__main__":
    main()