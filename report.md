**# v13 Report
- Final model: 0.7 x LinearRegression(positive=True) stack + 0.3 x HistGradientBoostingRegressor, clipped to 0-5
- LB RMSE 0.3739 (rank 30) | CV RMSE 0.484 | CV Pearson 0.921
- Level-1 blocks: text (mpnet + handcrafted), w2v, wavlm, whisper, wavlm_large, whisper_med (SVR + Ridge); gram, gram2 (Ridge); flu, flu2 (SVR)
- Rejected (no CV gain): wl_layers (blend CV 0.485), Whisper ASR-confidence (blend CV 0.486)**
