import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

# Create main window
root = tk.Tk()
root.title("Digit Recognizer")

# Canvas settings
canvas_width = 580
canvas_height = 580

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Create image for drawing
image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

# Draw on canvas
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)

    canvas.create_oval(x1, y1, x2, y2, fill="black")
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

# Predict digit
def predict_digit():
    # Resize image to 28x28
    img = image.resize((28, 28))

    # Invert colors
    img = ImageOps.invert(img)

    # Convert to numpy array
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Reshape for model
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)

    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    result_label.config(text="Draw a digit")

# Buttons
predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

# Result label
result_label = tk.Label(root, text="Draw a digit", font=("Arial", 18))
result_label.pack()

# Run app
root.mainloop()