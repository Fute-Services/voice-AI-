"""Generate the test fixture: a short two-speaker Hinglish "phone call".

Uses Gemini multi-speaker TTS so we have an exact reference transcript, then
degrades it to phone quality (8 kHz band-limited + background noise) with ffmpeg.

    python tests/fixtures/make_sample.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

HERE = Path(__file__).parent
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.8-flash-tts")

# Reference transcript (Roman-script Hinglish). Speaker names map to TTS voices.
DIALOGUE = [
    ("Rahul", "Hello, haan Priya, main Rahul bol raha hoon Sharma Logistics se. Kya aapke paas do minute hain?"),
    ("Priya", "Haan Rahul, boliye. Main abhi office se nikal rahi thi, but it's okay, batao."),
    ("Rahul", "Actually wo invoice number four five seven two ka payment abhi tak pending dikh raha hai. Amount hai thirty eight thousand five hundred rupees."),
    ("Priya", "Achha, wo wala. Mujhe laga accounts team ne last Friday ko hi transfer kar diya tha. Ek second, main check karti hoon."),
    ("Rahul", "Koi baat nahi, aap aaram se check kijiye. Agar UTR number mil jaaye toh please mujhe WhatsApp kar dena."),
    ("Priya", "Okay, mil gaya. Payment twenty second October ko initiate hua tha, but bank ne reject kar diya kyunki IFSC code galat tha."),
    ("Rahul", "Oh, that explains it. Sahi IFSC code hai HDFC zero zero zero one two three four. Main aapko email pe bhi bhej deta hoon."),
    ("Priya", "Perfect. Main kal subah tak dobara process karwa dungi. Kuch aur?"),
    ("Rahul", "Bas ek cheez, next month se hamara rate per kilo thoda revise ho raha hai, to main aapko new rate card bhej doonga."),
    ("Priya", "Theek hai, bhej dijiye, main apne manager Mr. Iyer ke saath discuss kar lungi. Thank you Rahul."),
    ("Rahul", "Thank you so much Priya, have a good evening. Bye."),
]
VOICES = {"Rahul": "Puck", "Priya": "Kore"}


def reference_text() -> str:
    return "\n".join(f"{who}: {line}" for who, line in DIALOGUE)


def main() -> None:
    load_dotenv(HERE.parents[1] / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY missing")
    client = genai.Client(api_key=key)

    style = "Natural Indian accent on a phone call, conversational pace, Hindi words pronounced as Hindi"
    parts = [
        types.Part(text=line, speech_metadata=types.SpeechMetadata(speaker=who, style=style))
        for who, line in DIALOGUE
    ]
    resp = client.models.generate_content(
        model=TTS_MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                            ),
                        )
                        for name, voice in VOICES.items()
                    ]
                )
            ),
        ),
    )
    pcm = resp.candidates[0].content.parts[0].inline_data.data  # 24 kHz s16le mono

    clean = HERE / "sample_clean.wav"
    with wave.open(str(clean), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)

    # Phone-call degradation: 300-3400 Hz band, 8 kHz, plus low-level brown noise.
    out = HERE / "sample_call.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(clean),
            "-f", "lavfi", "-i", "anoisesrc=color=brown:amplitude=0.02:sample_rate=24000",
            "-filter_complex",
            "[0:a]highpass=f=300,lowpass=f=3400[v];[v][1:a]amix=inputs=2:duration=first:weights=1 1[m];"
            "[m]aresample=8000",
            "-ac", "1", "-b:a", "32k", str(out),
        ],
        check=True,
    )
    clean.unlink()
    (HERE / "sample_call.reference.txt").write_text(reference_text() + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
