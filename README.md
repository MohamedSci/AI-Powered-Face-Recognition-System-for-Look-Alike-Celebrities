# AI-Powered Face Recognition System for Look-Alike Celebrities

**Title:** AI-Powered Face Recognition System for Look-Alike Celebrities

**Description:**

This project implements an AI-based face recognition system that identifies celebrities who resemble the person in a given image. It utilizes a Convolutional Neural Network (CNN) trained on a dataset of celebrity images to compute similarity scores between the input image and known celebrities.

**Key Features:**

- **Image Preprocessing:** Performs necessary steps like resizing and normalization to prepare images for the CNN.
- **CNN Architecture:** Employs a convolutional neural network architecture to extract features from facial images for recognition.
- **Similarity Score Calculation:** Calculates a similarity score between the input image and known celebrities based on their extracted features.
- **Top Similar Celebrities:** Provides a list of the top N celebrities most similar to the person in the input image.
- **Model Training and Evaluation:** Trains the CNN model on a dataset of celebrity images and evaluates its performance using test data.

**Usage:**

1. **Prerequisites:**
   - Python 3.x (ensure you have the necessary libraries like TensorFlow, OpenCV, etc. installed)
   - A dataset of celebrity images (prepare your own or consider using an existing dataset)

2. **Data Preparation:**
   - Organize your celebrity images into separate folders, where each folder represents a celebrity.
   - Ensure all images are in the same format (e.g., JPG, PNG).

3. **Running the Code:**
   - Clone this repository to your local machine: `git clone https://github.com/your-username/face-recognition-project.git`
   - Navigate to the project directory: `cd face-recognition-project`
   - Modify the `data_dir` and `labels_dir` variables in the code to point to your dataset location.
   - Run the script: `python main.py` (Replace `main.py` with the actual filename if different)

4. **Predicting Similar Celebrities:**
   - After training is complete, the script prompts you for the path to an image containing a person's face.
   - The script will predict the top N (default is 5) most similar celebrities based on their facial features and their similarity scores.

**Additional Notes:**

- Experiment with different CNN architectures and hyperparameters (learning rate, epochs, etc.) to potentially improve performance.
- Explore advanced techniques like background removal or pre-trained models like VGGFace for more complex scenarios.
- Consider incorporating data augmentation techniques to improve generalization performance.

**Future Enhancements:**

- Implement real-time face recognition using techniques like webcam integration.
- Build a web application to make the recognition process more user-friendly.
- Explore face analysis capabilities beyond recognition, such as age and gender estimation.

**Contribution:**

We welcome your contributions! Feel free to fork this repository and make improvements to the code, documentation, or functionalities. We can build this project together into a robust and effective face recognition system.

