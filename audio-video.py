import requests
import numpy as np
from PIL import Image
import pyttsx3
from moviepy import *

# ---------------- SETTINGS ----------------

PEXELS_API_KEY = "samanyusingh"
video_output = "paris_itinerary.mp4"

itinerary = {
    "Day 1": {
        "places": ["Eiffel Tower", "Seine River", "Louvre Museum"],
        "text": "Day one in Paris. Visit the Eiffel Tower, enjoy a walk along the Seine River, and explore the Louvre Museum."
    },
    "Day 2": {
        "places": ["Montmartre", "Sacré-Cœur", "Paris Cafés"],
        "text": "Day two in Paris. Discover Montmartre, admire the Sacré-Cœur, and relax at charming Paris cafés."
    },
    "Day 3": {
        "places": ["Notre-Dame Cathedral", "Latin Quarter", "Luxembourg Gardens"],
        "text": "Day three in Paris. See Notre-Dame Cathedral, wander the Latin Quarter, and unwind in Luxembourg Gardens."
    },
    "Day 4": {
        "places": ["Versailles Palace", "Paris Streets", "Evening Cruise"],
        "text": "Day four in Paris. Explore the Palace of Versailles, stroll through Paris streets, and enjoy an evening cruise."
    }
}

headers = {"Authorization": PEXELS_API_KEY}

# ---------------- HELPERS ----------------

def fetch_image(query, filename):
    search_url = "https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": 1}

    r = requests.get(search_url, headers=headers, params=params)

    if r.status_code != 200:
        raise Exception(f"Pexels API Error: {r.status_code} → {r.text}")

    data = r.json()

    if not data["photos"]:
        raise Exception(f"No image found for {query}")

    image_url = data["photos"][0]["src"]["landscape"]

    img_data = requests.get(image_url).content

    with open(filename, "wb") as f:
        f.write(img_data)

    return filename

def generate_audio(text, filename):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# ---------------- BUILD VIDEO ----------------

day_clips = []

for day, content in itinerary.items():

    print(f"Processing {day}")

    audio_file = f"{day}_audio.wav"
    generate_audio(content["text"], audio_file)

    audio = AudioFileClip(audio_file)
    duration = audio.duration

    duration_per_image = duration / len(content["places"])

    image_clips = []

    for i, place in enumerate(content["places"]):

        image_file = f"{day}_{i}.jpg"
        fetch_image(place + " Paris", image_file)

        img = Image.open(image_file)
        frame = np.array(img)

        clip = ImageClip(frame).with_duration(duration_per_image)
        image_clips.append(clip)

    video = concatenate_videoclips(image_clips, method="compose")
    video = video.with_audio(audio)

    # ---------------- SUBTITLES ----------------

    subtitle = TextClip(
        text=content["text"],
        font_size=28,
        color="white",
        size=video.size,
        method="caption"
    ).with_duration(duration).with_position(("center", "bottom"))

    video = CompositeVideoClip([video, subtitle])

    day_clips.append(video)

# ---------------- FINAL MERGE ----------------

final_video = concatenate_videoclips(day_clips, method="compose")

final_video.write_videofile(video_output, fps=24, codec="libx264")

print("Itinerary video created successfully")
