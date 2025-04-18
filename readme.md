# Cloth Similarity Recommendation

**APP Link** : https://cloth-recommendation.streamlit.app/

## Overview
This project provides a clothing similarity recommendation system using a deep learning feature extractor and a FastAPI backend. A pre-trained MobileNet model is fine-tuned on a custom dataset to extract image features, which are stored in a SQLite database. Users can upload images to find visually similar clothing items.

## Prerequisites
- Python 3.8 or higher
- pip
- Virtual environment tool (optional but recommended)

## Setup
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd "Cloth Similarity"
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv cloth_env
   # Windows
   cloth_env\Scripts\activate.bat
   # macOS/Linux
   source cloth_env/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Feature Extractor
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook exp.ipynb
   ```
2. In `exp.ipynb`:
   - Run cells to load and preprocess data from `dataset/train` and `dataset/test`.
   - Perform data augmentation with `ImageDataGenerator`.
   - Load MobileNet base model (weights from ImageNet) and build a custom classifier on top.
   - Train the model for 10 epochs.
   - Fine-tune by unfreezing the base model and re-running training for another 10 epochs.
   - Save the feature extractor:
     ```python
     feature_extractor.save('feature_extractor.keras')
     ```

## API Backend
The FastAPI app (`api.py`) provides two endpoints:

- **POST** `/reset_and_update_db`
  - Deletes existing `textile.db` if present.
  - Crawls product images from the configured base URL.
  - Extracts features using the saved feature extractor and populates the SQLite database.
  - Returns a success message.

- **POST** `/upload`
  - Accepts an image file upload.
  - Preprocesses and extracts features.
  - Queries the database for the top 5 visually similar items using cosine similarity.
  - Returns the list of image URLs with similarity scores.

### Running the API
```bash
uvicorn api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

## Usage
1. Reset and populate the database:
   ```bash
   curl -X POST http://127.0.0.1:8000/reset_and_update_db
   ```
2. Upload an image to get recommendations:
   ```bash
   curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:8000/upload
   ```

## Dataset Structure
```
dataset/
  train/
    pants/
    shirt/
    shorts/
    t-shirt/
  test/
    pants/
    shirt/
    shorts/
    t-shirt/
```

## License
This project is licensed under the MIT License.