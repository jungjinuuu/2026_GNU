conda activate comp

python src/models/baseline.py
python src/models/tree_models.py
python src/models/optuna.py
python src/models/catboost_optuna.py --n-trials 40
python src/models/stacking.py

python src/models/train_final_model.py
python src/frontend/app.py --host 0.0.0.0 --port 8501