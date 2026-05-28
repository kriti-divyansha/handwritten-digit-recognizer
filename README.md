# Handwritten Digit Recognizer using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

The project also includes an interactive GUI where users can draw digits and get real-time predictions using the trained AI model.

---

## Features

* Handwritten digit recognition using CNN
* Trained on the MNIST dataset
* Real-time digit prediction
* Interactive drawing canvas
* GUI built using Tkinter
* High accuracy (~99%)

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Tkinter
* Pillow (PIL)

---

## Dataset

This project uses the MNIST dataset, which contains:

* 70,000 handwritten digit images
* 28×28 grayscale images
* Digits from 0 to 9

---

## Project Structure

```bash
handwritten-digit-recognizer/
│
├── main.py              # CNN training script
├── app.py               # GUI application
├── digit_model.h5       # Trained model
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model Architecture

The CNN model consists of:

* Conv2D layers
* MaxPooling layers
* Flatten layer
* Dense layers
* Softmax activation for classification

The model achieves approximately 99% accuracy on the MNIST test dataset.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/handwritten-digit-recognizer.git
```

### 2. Navigate to Project Folder

```bash
cd handwritten-digit-recognizer
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python main.py
```

This will:

* Train the CNN model
* Evaluate accuracy
* Save the trained model as `digit_model.h5`

---

## Run the GUI Application

Run:

```bash
python app.py
```

Then:

1. Draw a digit using your mouse
2. Click the Predict button
3. The model will predict the digit

---

## Output

* Model Accuracy: ~99%
* Real-time handwritten digit prediction

---

## Future Improvements

* Web deployment using Flask or Streamlit
* Better UI/UX
* Support for custom uploaded images
* Real-time prediction while drawing
* Dark mode interface

---

## Author

Kriti Divyansha
