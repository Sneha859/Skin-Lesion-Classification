@echo off
REM Run preprocessing (create splits)
python src\preprocess\split_dataset.py

REM Train model
python src\models\train.py

REM Evaluate best model
python src\evaluate\evaluate_model.py

pause
