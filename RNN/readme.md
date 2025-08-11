# IMDB Movie Reviews Sentiment Analysis using Bi-LSTM

## 📌 Project Description
This project demonstrates how to build and train a **Bidirectional Long Short-Term Memory (Bi-LSTM)** neural network to classify IMDB movie reviews as **positive** or **negative**.  
The notebook walks through data loading, preprocessing, model training, evaluation, and making predictions on custom inputs.

## 📂 Dataset
We use the **IMDB dataset** from Keras, which contains 50,000 highly polarized movie reviews (25k for training, 25k for testing). Reviews are already preprocessed into integer sequences representing words.

## 🛠 Steps in the Notebook
1. **Load and Preprocess Data**  
   - Load integer-encoded IMDB reviews.  
   - Limit vocabulary size.  
   - Pad/truncate reviews to a fixed length for LSTM input.

2. **Model Architecture (Bi-LSTM)**  
   - Embedding layer to convert integers into dense vectors.  
   - Bidirectional LSTM layers to capture context from both directions.  
   - Dense layer with sigmoid activation for binary classification.

3. **Training and Validation**  
   - Early stopping to prevent overfitting.  
   - Training with accuracy and loss monitoring.

4. **Threshold Tuning**  
   - Adjust decision threshold to optimize precision-recall balance.

5. **Evaluation**  
   - Test set accuracy and loss.  
   - Precision-Recall curve plotting.

6. **Custom Predictions**  
   - Decode numeric sequences back to text.  
   - Predict sentiment for custom reviews.

