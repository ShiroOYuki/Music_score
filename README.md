## Environment Setup

1. **Requirements**:
   - Python version: `>= 3.12`

2. **Create a Virtual Environment**:
    
    Run the following command to create a virtual environment:
    ```bash
    virtualenv .venv
    ```

3. **Activate the Virtual Environment**:
    ```bash
    .venv\Scripts\activate
    ```

4. **Install Dependencies**
   
    Run the following command to install all required packages:
    ```bash
    pip install -r req.txt
    ```

---

## File Descriptions
1. `main.py`
    
    This is the main script for the project, serving as the central point for integrating and running all modules.

    - Usage:
    When deploying or testing the entire workflow, execute this script.

2. `use_model.py`

    Provides an example of using the pre-trained model to extract features from music files and compute similarity scores.

    - Usage:
    Helpful for developers testing feature extraction and similarity algorithms.

3. `yt_music.py`

    A utility for downloading music from YouTube and preprocessing it for further analysis.

    - Usage:
    Supports data collection for training models or analyzing music tracks.

--

## Testing the Application

1. **Run the Main Script**:
   
    Execute the following command to run the application:
    ```bash
    python main.py
    ```
