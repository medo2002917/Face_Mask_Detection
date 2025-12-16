# Face Mask Detection Using CNN

📌 Project Description

This project implements a binary image classification system using a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask or not.
The model is trained on face images and evaluated using accuracy, F1-score, and a confusion matrix.


---

📂 Dataset

Classes: With Mask, Without Mask

Source: Public Face Mask Dataset

Structure:


data/
├── with_mask/
└── without_mask/


---

⚙️ Installation

Install the required dependencies using:

pip install -r requirements.txt

requirements.txt

tensorflow
numpy
matplotlib
scikit-learn
opencv-python
pillow


---

▶️ Training the Model

Run the training script to train the CNN and save the best model:

python code/train.py

The best trained model will be saved automatically in:

saved_model/best_model.h5


---

📊 Evaluation

Evaluate the trained model and generate results:

python code/evaluate.py

This script generates:

Accuracy curve

Loss curve

Confusion matrix

Sample predictions


All outputs are saved in:

results/


---

🧪 Inference (Single Image Prediction)

To classify a single image from the command line:

python code/predict.py --image path/to/image.jpg

The script outputs the predicted class and confidence score.


---

📝 Notes

The dataset is not included in the repository due to size limitations.

Make sure the dataset folder structure is correct before running the code.

Early stopping is used to avoid overfitting.



---

👨‍🎓 Course Information

Course: CS417 – Neural Network
Submission Type: CNN Project


---
