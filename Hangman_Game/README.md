# Hangman Game – Machine Learning Solver

This project implements a machine learning–based solver for the classic Hangman game.  
The goal is to improve upon the standard frequency-based guessing strategy by learning how letters appear in words under different game states.

The solver was developed as part of the TrexQuant Hangman challenge.

---

## Project Overview

In the Hangman game, a hidden word must be guessed one letter at a time before a fixed number of incorrect guesses is reached.  
A naive frequency-based approach achieves a win rate of about 18%.

This project builds a structured ML pipeline that learns from simulated Hangman games and improves the win rate to around **40%** in practice games.

---

## Core Idea

The problem is framed as a supervised learning task, where the model learns which letters are likely to be correct given a partial word and past guesses.

Instead of using a single model, the solver combines **two complementary machine learning models**:

### 1. Presence Model (Binary Classification)
Predicts whether a given letter is likely to appear anywhere in the word.

This model is most useful in the early stages of the game, where broad filtering of bad guesses is important.

### 2. Masked Letter Model (Multiclass Classification)
Predicts which letter best fits a specific blank position in the word.

This model becomes dominant in the endgame, especially when only one or two blanks remain.

Both models are implemented using logistic regression for stability, interpretability, and minimal hyperparameter tuning.

---

## Feature Engineering

The performance of the solver relies heavily on feature design.  
The features fall into three main categories:

- **Basic statistics**  
  Word length, number of blanks, number of wrong guesses, completion rate, etc.

- **Context-based features**  
  Neighboring letters, n-gram statistics, and positional information around blanks.

- **Letter frequency features**  
  Global letter frequencies, word-length–specific frequencies, and position-based frequencies.

---

## Training Strategy

Since real Hangman game logs are not available, the models are trained on **simulated games** generated from a large dictionary.

- Presence model:
  - ~60k words
  - ~6 game states per word
  - ~360k training examples

- Masked letter model:
  - ~100k words
  - Multiple masked positions per word
  - ~300k training examples

Training is done offline, and the trained models are saved and later loaded by the game-playing agent.

---

## Guessing Strategy

The solver adapts its behavior depending on the stage of the game:

- **Early phase**: Presence model dominates
- **Middle phase**: Weighted combination of both models
- **End phase**: Masked letter model dominates

This phased strategy significantly improves performance compared to a single static policy.

---

## Results

- Win rate in practice games: **~40%**
- Baseline frequency strategy: **~18%**

The solver performs best on medium to long words (6–12 letters).  
Performance on very short words remains challenging.

---

## Repository Structure

Hangman_Game/
├── src/
│ └── hangman/
│ ├── features/
│ ├── models/
│ ├── training/
│ └── api/
├── notebooks/
│ └── hangman_agent.ipynb
├── data/
│ └── (dictionary files not tracked)
└── README.md
