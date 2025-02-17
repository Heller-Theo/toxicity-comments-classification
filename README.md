# **Toxicity Comments Classification**  

## **Project Overview**

This project aims to classify the toxicity level of YouTube comments using a **Recurrent Neural Network (RNN)**. The dataset is sourced from the [Kaggle Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) competition and contains **~2 million comments** labeled based on their toxicity level.  

## **Project Structure**

```plaintext
toxicity-comments-classification/
│── data/                   # Dataset files
│   ├── all_data.csv
│   ├── cleaned_dataset.pickle
│── notebooks/              # Jupyter notebooks for data processing & model training
│   ├── Preprocessing.ipynb
│   ├── Model_training.ipynb
│── src/                    # Source code
│   ├── tokenizer.pickle
│   ├── model/
│   │   ├── toxicity_model_0.keras
│   ├── tests/
│   │   ├── test_model.py   # Python script to test the trained model via CLI
│── .gitignore              # Files to ignore in Git
│── README.md               # Project documentation
│── requirements.txt        # Python package requirements
```

## **Project Files Overview**  

### **1. Data Files (`data/`)**

- **`all_data.csv`** – The original dataset containing ~2 million YouTube comments with toxicity labels.  
- **`all_data_sample.csv`** – A smaller subset of the dataset for quick testing.  
- **`cleaned_dataset.pickle`** – Preprocessed and tokenized dataset saved for training.  
- **`cleaned_dataset_sample.pickle`** – A sample of the cleaned dataset.  

### **2. Jupyter Notebooks (`notebooks/`)**

- **`Preprocessing.ipynb`** – Contains data cleaning steps such as text normalization, tokenization, and lemmatization.  
- **`Model_training.ipynb`** – Includes model training using an RNN, as well as evaluation and performance visualization.  

### **3. Source Code (`src/`)**

- **`tokenizer.pickle`** – A saved tokenizer used to convert text into numerical sequences for the model.  
- **`tokenizer_sample.pickle`** – A smaller version of the tokenizer for testing.  
- **`model/`**  
  - **`toxicity_model_0.keras`** – The trained RNN model used for predicting toxicity.  

### **4. Testing (`src/tests/`)**

- **`test_model.py`** – A Python script for testing the trained model using a command-line interface.  

### **5. Other Files**

- **`.gitignore`** – Specifies which files should be ignored by Git (e.g., dataset files).  
- **`README.md`** – The documentation file containing project details and instructions.
- **`requirements.txt`** – Lists the Python packages required for the project.

## **Dataset**

- **Source**: [Kaggle - Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- **Format**: CSV  
- **Columns of interest**:  
  - `comment_text` – The actual comment text  
  - `toxicity` – Toxicity score (0 to 1)  
  - `toxicity_annotator_count` – Number of annotators who rated toxicity  

## **Methodology**

1. **Preprocessing**

   - Convert text to lowercase  
   - Remove HTML tags and special characters  
   - Tokenize and lemmatize words  
   - Prepare sequences for the neural network  

2. **Model Training**  
   - Use an **RNN model** built with **TensorFlow/Keras**  
   - Train the model on cleaned comment data  
   - Evaluate performance with accuracy and loss metrics  

3. **Testing the Model**  
   - A **command-line interface (CLI) script** (`test_model.py`) is provided to test the model on custom comments.  

## **How to Run the Model Test**

### **Installation**

Ensure you have **Python 3.11+** and install the required dependencies:  

```bash
pip install -r requirements.txt
```

### **Testing a Comment via CLI**  

Run the following command:  

```bash
python src/tests/test_model.py --comment "I hate you" --tokenizer_path src/tokenizer.pickle --model_path src/model/toxicity_model_0.keras
```

### **Expected Output:**  

```bash
Comment: I hate you  
Predicted Toxicity: Slightly Toxic (Score: 0.58)  
```

## **Author**  

### Theo Heller
