# MNIST Classification: From Naive Bayes to CNN Improvements

This project explores handwritten digit classification on the MNIST dataset. It starts with a **Gaussian Naive Bayes** baseline using two handcrafted features for binary classification (digits 0 and 1) and extends to a **Convolutional Neural Network (CNN)** for multi-class (0–9) classification. Controlled experiments show how adjusting kernel size and the number of feature maps improves accuracy.

---

## 📂 Repository Structure

```

├── Project_Part_1.py                  # Naive Bayes (binary: 0 vs 1)
├── Project_Part_2.ipynb               # CNN (10-class classification with experiments)
├── train_0_img-2.mat                  # Training data for digit 0
├── train_1_img-2.mat                  # Training data for digit 1
├── test_0_img-2.mat                   # Testing data for digit 0
├── test_1_img-2.mat                   # Testing data for digit 1
├── Report.pdf                         # Full project report with results & discussion
└── README.md                          # Project documentation
```

---

## ⚙️ Setup

### Create Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib scikit-learn tensorflow notebook
```

Or create a `requirements.txt` with:
```
numpy
scipy
matplotlib
scikit-learn
tensorflow
notebook
```
Then install with:
```bash
pip install -r requirements.txt
```

---

## How to Run

### **1. Naive Bayes (Part 1)**
Run the Python script:
```bash
python Project_Part_1.py
```

This:
- Extracts **mean** and **standard deviation** pixel features for digits 0 and 1.
- Fits a Gaussian Naive Bayes model.
- Reports individual and overall accuracies.

**Reported Accuracy:**
- Digit `0`: 91.43%  
- Digit `1`: 92.42%  
- **Overall:** 91.93%

---

### **2. CNN (Part 2)**
Open the notebook:
```bash
jupyter notebook Project_Part_2.ipynb
```

**Experiments performed:**
1. **Baseline CNN** – 3×3 kernels, 6 and 16 filters in the first two conv layers.
2. **Increased kernel size** – switched to 5×5 kernels.
3. **Increased feature maps** – expanded conv layers to 12 and 20 filters.

**Performance Summary:**

| Model                        | Accuracy (%) | Error (%) |
|-----------------------------|-------------|-----------|
| Naive Bayes (2 features)    | 91.93       | 8.07      |
| CNN Baseline (3×3 kernels)  | 97.88       | 2.12      |
| CNN (5×5 kernels)           | 98.19       | 1.81      |
| CNN (more feature maps)     | **98.40**   | **1.60**  |

---

## Key Insights

- **Baselines matter:** Even simple features provide a transparent reference point.
- **More capacity helps — up to a point:** Larger kernels and more filters boost performance but with diminishing returns.
- **Good generalization:** Validation loss stayed lower than training loss, indicating minimal overfitting.

---

## 📄 Report

For a detailed explanation of the methodology, equations, results, and discussion, see the full [project report](./Report.pdf).

---

## Acknowledgments

- **Dataset:** [MNIST Handwritten Digit Dataset](http://yann.lecun.com/exdb/mnist/) by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- Libraries used: TensorFlow, NumPy, Matplotlib, and Scikit-learn.

---


