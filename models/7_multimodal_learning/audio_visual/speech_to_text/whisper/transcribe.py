import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import argparse
import os

def load_model(model_id="openai/whisper-large-v3-turbo"):
    print(f"Loading Whisper model: {model_id}...")
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def transcribe(audio_path, model_id="openai/whisper-large-v3-turbo", language=None):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return

    pipe = load_model(model_id)

    if language:
        print(f"Transcribing '{audio_path}' (target language: {language})...")
    else:
        print(f"Transcribing '{audio_path}' (language: auto-detect)...")

    # Whisper can auto-detect, but we force it if requested
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language

    result = pipe(audio_path, generate_kwargs=generate_kwargs)

    print("\n--- Transcription Result ---")
    print(result["text"])
    print("----------------------------\n")

    # Save to file
    output_path = audio_path + ".txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Transcription saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI Whisper (Hugging Face)")
    parser.add_argument("audio_path", type=str, help="Path to the audio file (mp3, wav, m4a)")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3-turbo", help="Model ID (default: large-v3-turbo)")
    parser.add_argument("--language", type=str, default=None, help="Target language (default: auto-detect). Example: 'english', 'german'")

    args = parser.parse_args()
    transcribe(args.audio_path, args.model, args.language)
