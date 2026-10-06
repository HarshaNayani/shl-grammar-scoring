# SHL Grammar Scoring Engine

Predicts a 0-5 grammar score from 45-60 sec spoken audio clips (SHL Hiring Assessment 2026, Kaggle).

## Approach
1. Transcribe audio with Whisper (faster-whisper, small model)
2. Extract text and audio features
3. Train regression models (Ridge baseline, then SVR / LightGBM) with 5-fold CV
4. Evaluate with RMSE and Pearson correlation

## Files
- `notebook.ipynb`: full pipeline, visualizations, and report
- `submission.csv`: final predictions
