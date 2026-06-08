# AI Image Detector

Detects AI generated images using only EXIF and file metadata. This is completely local and no external APIs are required.

## How it works

The detector inspects the embedded metadata of the image for signals that indicate AI generation.

* **Software and Creator EXIF tag**: AI tools like Midjourney, DALL E, Adobe Firefly, and Stable Diffusion often write their name directly into the Software EXIF field.
* **PNG text chunks**: Stable Diffusion and ComfyUI embed full generation parameters like prompt, steps, sampler, and seed into PNG text chunks.
* **XMP metadata**: Adobe Firefly and DALL E embed AI provenance claims in XMP namespaces.
* **C2PA manifest**: DALL E and other modern tools embed cryptographic content credentials.
* **Camera EXIF richness**: Real photographs from cameras contain abundant EXIF fields. Examples include make, model, lens, ISO, shutter speed, and GPS. AI generated images typically have none of these.

Each signal is weighted and combined into a single Stage 1 AI Score ranging from 0.0 to 1.0.

## Score bands

* 0.00 to 0.30 is LIKELY_REAL
* 0.30 to 0.60 is UNCERTAIN
* 0.60 to 0.85 is PROBABLY_AI
* 0.85 to 1.00 is LIKELY_AI

## Project Structure

* `main.py` is the command line entry point.
* `app.py` provides a Streamlit web interface for easy drag and drop testing.
* `pipeline.py` orchestrates the extraction and scoring process.
* `stage1_metadata.py` contains the core extraction logic for EXIF, XMP, and PNG chunks.
* `logger.py` handles writing results to an audit log file.
* `config.py` holds the thresholds for the score bands.

## Run the Web App

Install the dependencies and start the Streamlit server.

```bash
pip install Pillow python_dotenv streamlit
streamlit run app.py
```
<img width="3014" height="1640" alt="image" src="https://github.com/user-attachments/assets/45fb3538-6ee8-4e6a-a025-9acd3e0f7477" />

## Command Line Usage

You can also run the detector via the command line interface.

```bash
python main.py photo.jpg
```

To see more information or get a full JSON output, append the verbose or json flags to your command.


## Extending

Every result includes a `next_stage_input` field. This is a prepackaged payload ready to pass to any external API. For example, you could pass it to a vision model or a cloud classifier as a future Stage 2. To wire in a future API, import your module, pass the payload to it, blend the scores, and rerun the verdict logic.
