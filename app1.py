import os
import time
from typing import Any, List
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import ollama
import random
from gtts import gTTS
 
from utils.custom import css_code
 
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
LOCAL_MODEL_PATH = "./models"
 
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
 
def progress_bar(amount_of_time: int) -> None:
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
        generated_text: str = image_to_text(url)[0]["generated_text"]
        generated_text = generated_text.lstrip("arafed").strip()
        generated_text = generated_text.capitalize()
    except Exception as e:
        st.error(f"An error occurred while generating text from the image: {str(e)}")
        return f"Error: Unable to generate text from image. {str(e)}"
 
    print(f"IMAGE INPUT: {url}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text
 
def generate_story_from_text(scenarios: List[str]) -> str:
    combined_scenarios = " ".join(scenarios)
    prompt_template: str = f"""
    You are a talented screenwriter who can create engaging dialogue-based stories with morals from multiple scene descriptions.
    Based on the following scenarios, which describe multiple images or scenes, create a cohesive short story told primarily through dialogue. The story should:
    - Be approximately 200-250 words long
    - Consist mainly of dialogue, with minimal narrative description
    - Be coherent and incorporate elements from all provided scenarios
    - Include character names before each line of dialogue
    - Capture the mood and context of the scenes
    - Include brief action descriptions in parentheses where necessary
    - Have a clear beginning, middle, and end
    - Conclude with a clear moral or lesson learned that ties the scenes together
    - Format the dialogue so that each character's line starts with their name in ALL CAPS, followed by a colon and their dialogue
    SCENARIOS: {combined_scenarios}
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
        lines = generated_story.split('\n')
        formatted_lines = []
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                formatted_lines.append(f"{parts[0].strip().upper()}: {parts[1].strip()}")
            else:
                formatted_lines.append(line)
        formatted_story = '\n'.join(formatted_lines)
        # if "Moral:" not in formatted_story:
        #     formatted_story += "\n\nMoral: [The moral should be inferred from the story and added here.]"
    except Exception as e:
        st.error(f"An error occurred while generating the dialogue-based story: {str(e)}")
        return f"Error: Unable to generate dialogue-based story. {str(e)}"
 
    return formatted_story
 
def generate_speech_from_text(message: str, max_retries: int = 10, initial_delay: float = 1, max_delay: float = 60) -> None:
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
    os.system("ffmpeg -i generated_audio.mp3 generated_audio.flac")
    os.remove("generated_audio.mp3")
    print("Audio file generated using gTTS fallback.")
 
def main() -> None:
    st.set_page_config(page_title="MULTI-IMAGE TO STORY CONVERTER", page_icon="🖼️")
 
    st.markdown(css_code, unsafe_allow_html=True)
 
    with st.sidebar:
        st.write("Multi-Image Story Generator App")
 
    st.header("Multi-Image to Story Converter")
    uploaded_files = st.file_uploader("Please choose files to upload", type="jpg", accept_multiple_files=True)
 
    if uploaded_files:
        scenarios = []
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            bytes_data = uploaded_file.getvalue()
            with open(uploaded_file.name, "wb") as file:
                file.write(bytes_data)
            with st.spinner(f"Generating text for {uploaded_file.name}..."):
                scenario = generate_text_from_image(uploaded_file.name)
                if not scenario.startswith("Error:"):
                    scenarios.append(scenario)
                else:
                    st.error(scenario)
        if scenarios:
            with st.spinner("Generating story from all scenarios..."):
                story = generate_story_from_text(scenarios)
            with st.spinner("Generating audio..."):
                generate_speech_from_text(story)
 
            with st.expander("Generated Image scenarios"):
                for i, scenario in enumerate(scenarios, 1):
                    st.write(f"Image {i}: {scenario}")
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
            st.error("No valid scenarios were generated from the uploaded images.")
 
if __name__ == "__main__":
    main()