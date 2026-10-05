# 🧠 PyTorch Image Intelligence

An end-to-end deep learning image classification system built with **PyTorch**, **ResNet18**, **Streamlit**, and **Supabase**.

The project demonstrates the complete machine learning workflow from building a convolutional neural network from scratch to transfer learning, model evaluation, real-image inference, web deployment, prediction logging, and analytics.

---

## 📌 Project Overview

This project was developed as a practical implementation of PyTorch and computer vision concepts.

Instead of only training a model inside a notebook, the project follows a complete machine learning pipeline:

```text
Dataset
   ↓
PyTorch Data Pipeline
   ↓
CNN from Scratch
   ↓
Model Evaluation
   ↓
CNN Improvement
   ↓
Transfer Learning with ResNet18
   ↓
Real-Image Inference
   ↓
Streamlit Web Application
   ↓
Supabase Database
   ↓
Prediction Analytics Dashboard
```

The final model uses **ResNet18 transfer learning** for CIFAR-10 image classification and achieved **93.87% test accuracy**.

---


<img width="1915" height="926" alt="Image classifier" src="https://github.com/user-attachments/assets/1a092b6b-6e27-46b7-8440-fe83ff6613a5" />

# 🎯 Objectives

The main objectives of this project are to:

- Learn the fundamentals of PyTorch through practical implementation.
- Understand tensors, datasets, DataLoaders, and GPU computation.
- Build a convolutional neural network from scratch.
- Implement a complete PyTorch training loop.
- Understand loss functions, optimizers, gradients, and backpropagation.
- Evaluate a model using accuracy, precision, recall, F1-score, and confusion matrices.
- Reduce overfitting using data augmentation, Batch Normalization, Dropout, and weight decay.
- Apply transfer learning using a pretrained ResNet18 model.
- Perform inference on user-uploaded images.
- Deploy the trained model through Streamlit.
- Store prediction history using Supabase.
- Build an analytics dashboard for model usage.

---

# 📊 Dataset

The project uses the **CIFAR-10** dataset.

CIFAR-10 contains **60,000 RGB images** across 10 object categories.

### Dataset Distribution

| Split | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |
| Total | 60,000 |

Each original CIFAR-10 image has dimensions:

```text
3 × 32 × 32
```

where:

```text
3  = RGB channels
32 = image height
32 = image width
```

### Classes

The ten CIFAR-10 classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | Datasets, transformations, and pretrained models |
| ResNet18 | Transfer-learning architecture |
| Google Colab | GPU-based model training |
| NumPy | Numerical operations |
| Pandas | Analytics and data processing |
| Matplotlib | Training and evaluation visualization |
| Scikit-learn | Classification metrics and confusion matrix |
| Pillow | Image processing |
| Streamlit | Interactive web application |
| Supabase | Prediction database |
| GitHub | Source-code and project management |

---

# 🧠 Model Development

Three model stages were implemented and compared.

## Model 1 — CNN From Scratch

The first model was created directly with PyTorch using:

- Conv2D
- ReLU
- MaxPooling
- Fully connected layers

Architecture:

```text
Input
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool
  ↓
Flatten
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
10 Classes
```

### Result

```text
Training Accuracy: 81.21%
Test Accuracy:     70.65%
```

The difference between training and test performance indicated that the basic CNN was beginning to overfit.

---

# 🚀 Model 2 — Improved CNN

The CNN was improved using several regularization and generalization techniques.

### Improvements

- Data augmentation
- Random cropping
- Random horizontal flipping
- Batch Normalization
- Dropout
- Weight decay
- Additional convolutional layer

Architecture:

```text
Input
  ↓
Conv2D
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Flatten
  ↓
Linear
  ↓
ReLU
  ↓
Dropout
  ↓
Classifier
```

### Result

```text
Best Test Accuracy: 76.22%
```

This improved generalization compared with the original CNN.

---

# 🏆 Model 3 — ResNet18 Transfer Learning

The final model uses a pretrained **ResNet18**.

The network was initialized using ImageNet pretrained weights and then fine-tuned for CIFAR-10.

The original ResNet18 classifier was replaced with:

```python
model.fc = nn.Linear(512, 10)
```

The ten outputs correspond to the ten CIFAR-10 classes.

### Data Split

```text
50,000 CIFAR-10 training images
             │
             ├── 45,000 Training
             │
             └── 5,000 Validation

10,000 Test Images
```

### Training Results

| Epoch | Train Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 86.33% | 92.92% |
| 2 | 93.23% | 93.88% |
| 3 | 95.24% | 93.84% |
| 4 | 96.51% | 93.90% |
| 5 | 97.01% | **94.20%** |

### Final Performance

```text
Best Validation Accuracy: 94.20%
Final Test Accuracy:      93.87%
```

---

# 📈 Model Comparison

| Model | Test Accuracy |
|---|---:|
| Basic CNN | 70.65% |
| Improved CNN | 76.22% |
| **ResNet18 Transfer Learning** | **93.87%** |

The final ResNet18 model improved test accuracy by approximately **23.22 percentage points** compared with the original CNN.

---

# 🔥 PyTorch Training Pipeline

The core PyTorch training process used throughout the project is:

```python
optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

This represents:

```text
Clear previous gradients
        ↓
Forward propagation
        ↓
Calculate loss
        ↓
Backpropagation
        ↓
Update model parameters
```

---

# 📏 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Class-wise accuracy
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

For the original CNN, the strongest class was:

```text
Horse: 83.50%
```

and the weakest class was:

```text
Cat: 57.20%
```

These results motivated further model improvement and transfer learning.

---

# 🖼️ Real-Image Inference

The final application accepts user-uploaded images.

The inference pipeline is:

```text
Uploaded Image
       ↓
Convert to RGB
       ↓
Resize to 128 × 128
       ↓
Convert to Tensor
       ↓
ImageNet Normalization
       ↓
Add Batch Dimension
       ↓
ResNet18
       ↓
Softmax
       ↓
Top-3 Predictions
```

The application displays:

- Predicted class
- Prediction confidence
- Top-3 predicted classes
- Confidence score for each prediction

---

# 🌐 Streamlit Web Application

A Streamlit interface provides an interactive frontend for the trained model.

The application contains three sections:

### Classifier

Users can:

- Upload JPG, JPEG, or PNG images.
- Run PyTorch inference.
- View the predicted class.
- View model confidence.
- View the top-3 predictions.

### Analytics

The analytics dashboard displays:

- Total number of predictions
- Average prediction confidence
- Most frequently predicted class
- Prediction distribution
- Average confidence by class
- Recent prediction history

### About

The About page describes:

- Model development
- Model accuracy
- AI pipeline
- Technology stack

---

# 🗄️ Supabase Database

Supabase is used to store application prediction history.

The database stores:

```text
filename
predicted_class
confidence
second_prediction
second_confidence
third_prediction
third_confidence
created_at
```

This allows the application to maintain prediction history and generate usage analytics.

---

# 🏗️ Complete System Architecture

```text
                     USER
                       │
                       ▼
               ┌──────────────┐
               │  STREAMLIT   │
               └──────┬───────┘
                      │
                Image Upload
                      │
                      ▼
               Preprocessing
                      │
                      ▼
               ┌──────────────┐
               │   PYTORCH    │
               │   ResNet18   │
               └──────┬───────┘
                      │
                      ▼
                    Softmax
                      │
                      ▼
              Top-3 Predictions
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
      User Interface       Supabase
                               │
                               ▼
                      Prediction History
                               │
                               ▼
                     Analytics Dashboard
```

---

# 📂 Project Structure

```text
pytorch-image-intelligence/
│
├── PyTorch_Image_Intelligence_Training.ipynb
│
├── app.py
│
├── cifar10_resnet18.pth
│
├── requirements.txt
│
├── supabase_setup.sql
│
├── .gitignore
│
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `PyTorch_Image_Intelligence_Training.ipynb` | Complete model development and training |
| `app.py` | Streamlit application |
| `cifar10_resnet18.pth` | Trained ResNet18 weights |
| `requirements.txt` | Python dependencies |
| `supabase_setup.sql` | Database table creation |
| `.gitignore` | Prevents private/unnecessary files from GitHub |
| `README.md` | Complete project documentation |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
```

Move into the project directory:

```bash
cd pytorch-image-intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local Streamlit address displayed in the terminal.

---

# 🔐 Supabase Configuration

Create a Supabase project and execute the SQL contained in:

```text
supabase_setup.sql
```

Configure the Streamlit secrets:

```text
.streamlit/secrets.toml
```

Example:

```toml
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
```

> **Important:** Never commit `secrets.toml`, API keys, passwords, or database credentials to GitHub.

The `.gitignore` file should contain:

```text
.streamlit/secrets.toml
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

# ☁️ Deployment

The Streamlit application can be deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
Google Colab
     ↓
Train Model
     ↓
Save Model Weights
     ↓
GitHub Repository
     ↓
Streamlit Community Cloud
     ↓
Configure Secrets
     ↓
Live Application
```

---

# 📚 PyTorch Concepts Learned

This project demonstrates practical understanding of:

- PyTorch tensors
- Tensor shapes
- CPU/GPU devices
- Dataset
- DataLoader
- Batch processing
- Image transformations
- Normalization
- `nn.Module`
- `forward()`
- Conv2D
- ReLU
- MaxPooling
- Fully connected layers
- CrossEntropyLoss
- Adam optimizer
- Learning rate
- Epochs
- Forward propagation
- Backpropagation
- Gradients
- Model parameter updates
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- Data augmentation
- Batch Normalization
- Dropout
- Weight decay
- Transfer learning
- ResNet18
- ImageNet pretrained weights
- Fine-tuning
- Model checkpointing
- Softmax
- Top-K prediction
- Model inference
- Saving/loading PyTorch weights

---

# ⚠️ Limitations

The model was trained and evaluated on CIFAR-10.

CIFAR-10 contains small 32×32 images, so performance on arbitrary high-resolution real-world photographs may differ from the reported CIFAR-10 test accuracy.

The reported **93.87% accuracy refers specifically to the CIFAR-10 test set** and should not be interpreted as guaranteed accuracy for arbitrary uploaded images.

The application currently supports only the ten CIFAR-10 categories.

---

# 🔮 Future Improvements

Possible future extensions include:

- Grad-CAM explainability
- Model confidence calibration
- Unknown/out-of-distribution image detection
- More advanced data augmentation
- Additional pretrained architectures
- Model comparison dashboard
- Inference latency measurement
- Model size comparison
- REST API integration
- Docker deployment
- Custom real-world dataset training
- Experiment tracking

---

# 🎓 Learning Outcome

This project demonstrates progression from understanding basic PyTorch operations to developing a complete deep learning application.

```text
PyTorch Fundamentals
        ↓
CNN From Scratch
        ↓
Training & Backpropagation
        ↓
Model Evaluation
        ↓
Regularization
        ↓
Transfer Learning
        ↓
Real-Image Inference
        ↓
Web Application
        ↓
Database Integration
        ↓
Analytics
        ↓
Deployment
```

Rather than using PyTorch only for model training, the project demonstrates how a trained deep learning model can become part of an end-to-end usable software system.

---

# 👩‍💻 Author

**Fatima Noor**

Computer Science | Artificial Intelligence | Computer Vision | Deep Learning

---

## ⭐ Project Status

```text
PyTorch Fundamentals        ✅
CNN From Scratch            ✅
Training Pipeline           ✅
Model Evaluation            ✅
Improved CNN                ✅
Transfer Learning           ✅
ResNet18                    ✅
Real-Image Inference        ✅
Streamlit Application       ✅
Supabase Integration        ✅
Analytics Dashboard         ✅
GitHub Documentation        ✅
```

---

If you find this project useful, consider giving the repository a ⭐.
