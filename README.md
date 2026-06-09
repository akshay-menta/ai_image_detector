# AI Image Detector Pipeline

A smart, cost-effective tool to detect AI-generated images. This system uses a 3-stage pipeline to save API costs by checking free local metadata first, and only sending confusing or tricky images to a highly accurate AI detection model.

## How It Works: The 3-Stage Pipeline

### Stage 1: Local Metadata Scan (Free)
The script reads the hidden data inside the image file (like EXIF data, PNG chunks, and C2PA credentials) completely locally. 
* Real camera photos usually have rich metadata (like Shutter Speed, ISO, Lens Make).
* AI images often have specific software tags (like "Midjourney" or "Stable Diffusion") or no metadata at all.
If Stage 1 finds a definite answer (either clearly real or clearly AI), the pipeline **stops here** to save money.

### Stage 2: Winston AI Image Detection
If the image is in the "gray zone" (no clear metadata found in Stage 1), the image is sent to **Winston AI**, a purpose-built, highly accurate image detection model. 
* Winston AI visually analyzes the image for AI artifacts and returns a Human Score (0-100).
* We blend this score with the Stage 1 score (75% Winston, 25% Metadata) to get the most accurate final result.

### Stage 3: Confidence Router
Finally, the router looks at the final score and decides what should happen to the image:
* 🟢 **Auto-Approve**: The image is clearly real.
* 🟡 **Human Review**: The image is still tricky, send it to a human moderator.
* 🔴 **Auto-Reject**: The image is clearly AI-generated.

## Score Bands

The final AI probability score ranges from 0.00 to 1.00:
* **0.00 to 0.30**: LIKELY_REAL
* **0.30 to 0.60**: UNCERTAIN
* **0.60 to 0.85**: PROBABLY_AI
* **0.85 to 1.00**: LIKELY_AI

## Project Structure

* `poc_app.py`: The main Streamlit web application. Features a premium dark-themed UI and drag-and-drop testing.
* `pipeline.py`: Coordinates the full 3-stage execution.
* `stage1_metadata.py`: Core logic for reading EXIF, XMP, and PNG chunks locally.
* `stage2_winston.py`: Handles secure uploading and API communication with Winston AI.
* `stage3_router.py`: Handles final routing logic and builds the audit trail.
* `logger.py`: Writes all detection results to an audit log file (`audit_log.jsonl`).

## How to Run

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```
<img width="2946" height="1686" alt="image" src="https://github.com/user-attachments/assets/aa8aae32-8a32-43b3-a973-7370a8e69e8d" />

<img width="3014" height="1640" alt="image" src="https://github.com/user-attachments/assets/77eb2f6b-3409-4702-8570-90e17aef859e" />



2. **Add your Winston AI Key**:
Create a `.env` file in the main folder and add your API key from [Winston AI](https://gowinston.ai/):
```
WINSTON_API_KEY=your_key_here
```
*(You can also paste the key directly into the sidebar of the web app.)*

3. **Start the Web App**:
```bash
streamlit run poc_app.py
```
