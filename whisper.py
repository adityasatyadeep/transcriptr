import os
import modal

MODEL_DIR = os.path.join(os.path.dirname(__file__), "/model")
MODEL_NAME = "openai/whisper-large-v3"
MODEL_REVISION = "afda370583db9c5359511ed5d989400a6199dfe1"

image = modal.Image.debian_slim().apt_install("ffmpeg").pip_install(
        "torch",
        "transformers",
        "hf-transfer",
        "huggingface_hub",
        "librosa",
        "soundfile",
        "accelerate",
        "datasets",
    )

app = modal.App(
    name="transcriptr",
    image=image,
    # secrets=[modal.Secret.from_name("OPENAI_API_KEY")],
)
# volume = modal.Volume.from_name("audio_samples", create_if_missing=True)



# build is run as container is being built
@modal.build()
def download_model(self):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        revision=MODEL_REVISION,
    )
    move_cache()

@app.cls(
    gpu="a10g",
    concurrency_limit=10,
    mounts=[modal.Mount.from_local_dir("./audio_samples", remote_path="/root/audio_samples")]
)
class Model:
    # enter is run right when container starts
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to("cuda")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            return_timestamps=True,
            device="cuda"
        )
    @modal.batched(max_batch_size=1, wait_ms=1000)
    def transcribe(self, audio):
        print("Processing: ", audio)
        import time
        start = time.time()
        result = self.pipe(audio)
        end = time.time()
        print(f"Time taken: {end - start}\n")
        return result


@app.function(mounts=[modal.Mount.from_local_dir("./audio_samples", remote_path="/root/audio_samples")])
@modal.web_endpoint()
def transcribe_audio():
    whisper = Model()
    print("📣 Transcribing")

    files = os.listdir("/root/audio_samples")
    file_paths = ["/root/audio_samples/" + f for f in files]

    responses = []

    for x in whisper.transcribe.map(file_paths):
        print(x["text"])
        responses.append(x['text'])

    return responses
