# 📸 Wander-Lens

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![GitHub Stars](https://img.shields.io/github/stars/akshit40/Wander-Lens?style=social)
![GitHub Forks](https://img.shields.io/github/forks/akshit40/Wander-Lens?style=social)

> ⚡ *Wander-Lens* is an AI-powered Streamlit app that recognizes famous landmarks from images, even those taken from lazy, bad, or unconventional angles.

---

## 🌟 Features

- 🧠 **Advanced AI Model**: Utilizes a fine-tuned MobileNetV2 model for high-accuracy image classification.
- 🏞️ **24+ Landmark Classes**: Trained to recognize a wide variety of famous Indian monuments.
- 💪 **Robust Recognition**: Specially trained using data augmentation to handle blurry, off-center, and poorly lit "lazy" photos.
- 🖥️ **Interactive UI**: A clean and modern web interface built with Streamlit for easy image uploads and result visualization.
- 📊 **Detailed Predictions**: Provides the top 3 landmark predictions with confidence scores for a more informative result.
- 🚀 **Automated Setup**: Includes a script to automatically download and structure the required dataset from Kaggle.

---

## 📈 Model Performance

The model was trained for 15 epochs and achieved a final validation accuracy of **87.4%**. The training process involved a two-phase fine-tuning approach on a MobileNetV2 base.


---

## 🚀 Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/akshit40/Wander-Lens.git
    cd Wander-Lens
    ```

2. **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the dataset**
    - Follow the instructions [here](https://www.kaggle.com/docs/api) to get your `kaggle.json` API token.
    - Place the `kaggle.json` file in the correct directory:
        - **Windows**: `C:\Users\YourUsername\.kaggle\`
        - **macOS/Linux**: `~/.kaggle/`
    - Run the setup script:
    ```bash
    python setup_dataset.py
    ```

5. **Run the app**
    ```bash
    streamlit run app.py
    ```

---

## 💡 How to Use

1. Launch the Streamlit app.
2. Click the **"Upload your image here"** button and select an image file (`.jpg`, `.jpeg`, or `.png`).
3. The app will instantly analyze the image and display the results.
4. View the top prediction and its confidence score.
5. Check the **"See other possibilities"** expander to view the 2nd and 3rd best guesses.

---

## 🛠️ Tech Stack & Model

| Component       | Technology / Method                                        |
| --------------- | ---------------------------------------------------------- |
| **Framework**   | Streamlit                                                  |
| **AI/ML**       | TensorFlow, Keras                                           |
| **Model Arch**  | MobileNetV2 (Fine-Tuned)                                    |
| **Techniques**  | Transfer Learning, Data Augmentation, Dropout               |
| **Dataset**     | [Indian Monuments Image Dataset](https://www.kaggle.com/datasets/danushkumarv/indian-monuments-image-dataset) by Danush Kumar V |
| **Data Setup**  | Kaggle API                                                  |

---

## 🙏 Acknowledgments

- 🧠 TensorFlow Team – for creating the powerful deep learning framework.
- 🧩 Streamlit Team – for the amazing rapid development framework.
- 💾 Danush Kumar V – for providing the excellent dataset on Kaggle.

---

Built with ❤️ by Akshit

**Note:** This project was trained on a smaller dataset subset to avoid overheating my laptop CPU/GPU. In the future, I plan to enhance accuracy and scale to the full dataset when I have access to more powerful hardware.


## 🤝 Contributing

Contributions are welcome! If you'd like to improve Wander-Lens, please follow these steps:

```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/your-amazing-feature

# Make your changes and commit them
git commit -m "Add your amazing feature"
git push origin feature/your-amazing-feature

# Open a pull request 🚀


# Make your changes and commit them
git commit -m "Add your amazing feature"
git push origin feature/your-amazing-feature

# Open a pull request 🚀
