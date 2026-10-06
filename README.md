# SHL Grammar Scoring Engine

Predicts a 0-5 grammar score from 45-60 sec spoken audio clips
(SHL Hiring Assessment 2026, Kaggle).

**Results:** Public LB RMSE 0.3739 (rank 30) | 5-fold CV RMSE 0.484 | CV Pearson 0.921

## Approach
1. Transcribe the audio with Whisper (faster-whisper).
2. Extract features from three views:
   - **Audio embeddings:** wav2vec2, WavLM (base and large), Whisper (small and medium)
   - **Text:** mpnet sentence embeddings plus handcrafted features; grammar
     acceptability from CoLA models and GPT-2 perplexity
   - **Fluency:** pauses, speech rate, energy, MFCC and pitch (librosa)
3. Train one SVR or Ridge model per feature block with 5-fold CV and keep
   the out-of-fold predictions.
4. Stack the out-of-fold predictions: 0.7 x LinearRegression(positive=True)
   + 0.3 x HistGradientBoostingRegressor, clipped to 0-5.
5. Evaluate with RMSE and Pearson correlation.

Tried but not used (no CV gain): WavLM-large layer pooling, Whisper
ASR-confidence features, a small neural-net head.

## Files
- `shl-grammar-scoring-pipeline.ipynb`: full pipeline
- `submission.csv`: final predictions (v13)
- `report.md`: model details and metrics
- `requirements.txt`: dependencies

## Note
The competition audio and CSVs are not included (private challenge).
Pre-computed features were saved to a Kaggle dataset (`shl-features`).
