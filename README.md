# chinmukh.ai

Chinmukh.ai (*চিন মুখ*, lit. *recognize face*) is a simple, scalable and deployable multi-model architecture designed for gender classification and facial verification.

Created for the [COMSYS-2025 Hackathon](https://unstop.com/p/comsys-hackathon-5-2025-6th-international-conference-on-frontiers-in-computing-and-systems-comsys-2025-jadavpur-univer-1499220), this architecture has been trained on the [FACECOM dataset](https://drive.google.com/file/d/1KXx1CW-BM3muxzCtmyhyv9xtWco_nDbL/view), which consists of over 5,000 face images captured and simulated under varied real-world conditions.

The goal of this project is to develop a **generalizable** and **resilient** model capable of making **robust predictions**, even when under the influence of challenging environmental factors such as blur, occlusion, lighting changes, jitter, noise and rotation.

## Installation/Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/reacfen/chinmukh.ai
    cd chinmukh.ai
    ```

2. Download the pretrained weights from [this Google Drive link](https://drive.google.com/file/d/1ip1Tlf14yTRL_ZBcFjoCbmmjz_VtjM_O/view?usp=sharing) and extract it inside the `chinkmukh.ai` folder alongside the dataset.

3. Set up a Python virtual environment:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - Windows:

        ```bash
        venv\Scripts\activate.bat
        ```

    - macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

5. Install dependencies in the virtual environment:

    ```bash
    pip install -r requirements.txt
    ```

6. Run the model:

    - Gender Classifier:

        ```bash
        # Train/fine-tune the model
        python gender-classifier/src/train_and_validate.py
        ```

        ```bash
        # Visualize training and validation curves
        python gender-classifier/src/visualization.py
        ```

        ```bash
        # Run the model on a sample image (TTA disabled)
        python gender-classifier/src/inference.py
        ```

        ```bash
        # Run the model on a sample image (TTA enabled)
        python gender-classifier/src/inference.py --use-tta
        ```

        ```bash
        # Test the model (TTA disabled)
        python gender-classifier/src/test_and_evaluate.py
        ```

        ```bash
        # Test the model (TTA enabled)
        python gender-classifier/src/test_and_evaluate.py --use-tta
        ```

    - Face Verifier:

        ```bash
        # Train/fine-tune the model
        python face-verifier/src/train_and_validate.py
        ```

        ```bash
        # Visualize training and validation curves
        python face-verifier/src/visualization.py
        ```

        ```bash
        # Run the model on a pair of sample images (TTA disabled)
        python face-verifier/src/inference.py
        ```

        ```bash
        # Run the model on a pair of sample images (TTA enabled)
        python face-verifier/src/inference.py --use-tta
        ```

        ```bash
        # Test the model (TTA disabled)
        python face-verifier/src/test_and_evaluate.py
        ```

        ```bash
        # Test the model (TTA enabled)
        python face-verifier/src/test_and_evaluate.py --use-tta
        ```

        > If you want to train or run the model on the CPU instead of the GPU, just append the `--use-cpu` flag to the script. For example:
        >
        > ```bash
        > python gender-classifier/src/train_and_validate.py --use-cpu  # Use CPU for training
        >
        > # Works for the other scripts as well (except 'visualization.py', which doesn't use GPU compute)
        > ```

7. Once done, deactivate the virtual environment and exit the terminal session:

    - Windows:

        ```bash
        venv\Scripts\deactivate.bat
        ```

    - macOS/Linux:

        ```bash
        source venv/bin/deactivate
        ```

## Analysis

Here are the relevant links to each component of this multi-model architecture (includes block diagrams, analysis and training/validation results):

[» Task A: Gender Classifier (Binary Classification using Sigmoid)](./gender-classifier/)

[» Task B: Face Verifier (Facial Verification using Cosine Similarity)](./face-verifier/)
