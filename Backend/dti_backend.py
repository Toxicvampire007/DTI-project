from DeepPurpose import DTI as models
from DeepPurpose import utils
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
import itertools
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_hyperparameter_tuning.log'),
        logging.StreamHandler()
    ]
)

class DTIHyperparameterTuner:
    def __init__(self, data_path):
        self.data_path = data_path
        self.best_config = None
        self.best_score = -np.inf
        self.results = []
        self.scaler = None

    def load_and_preprocess_data(self):
        logging.info("Loading and preprocessing dataset...")
        df = pd.read_csv(self.data_path)

        # Remove negative values for log1p
        df = df[df['LABEL'] > -1]
        df['LABEL'] = np.log1p(df['LABEL'])

        # Filter out label outliers
        q1 = df['LABEL'].quantile(0.05)
        q3 = df['LABEL'].quantile(0.95)
        df = df[(df['LABEL'] >= q1) & (df['LABEL'] <= q3)]

        # Normalize with RobustScaler
        self.scaler = RobustScaler()
        df['LABEL_NORMALIZED'] = self.scaler.fit_transform(df[['LABEL']])

        with open('best_label_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        X = utils.data_process(
            X_drug=df['SMILES'],
            X_target=df['TARGET'],
            y=df['LABEL_NORMALIZED'],
            drug_encoding='DGL_AttentiveFP',
            target_encoding='Transformer',
            split_method='cold_drug',
            frac=[0.6, 0.2, 0.2],
            random_seed=42
        )

        self.train, self.val, self.test = X
        logging.info(f"Data processed: Train={len(self.train)}, Val={len(self.val)}, Test={len(self.test)}")

    def define_hyperparameter_grid(self):
        return {
            'cls_hidden_dims': [[1024, 512, 256], [512, 512, 256], [2048, 1024, 512]],
            'train_epoch': [50, 75, 100],
            'LR': [1e-5, 5e-5, 1e-4],
            'batch_size': [16, 32, 64]
        }

    def evaluate_model(self, model, config_info):
        y_pred_test = model.predict(self.test)
        y_true_test = np.array(self.test['Label'].values)
        test_r2 = r2_score(y_true_test, y_pred_test)
        test_mse = mean_squared_error(y_true_test, y_pred_test)
        score = 0.7 * test_r2 + 0.3 * (1 - min(test_mse, 0.05))
        return {
            'test_r2': test_r2,
            'test_mse': test_mse,
            'composite_score': score,
            'config': config_info
        }

    def train_single_configuration(self, config):
        try:
            model_config = models.generate_config(
                drug_encoding='DGL_AttentiveFP',
                target_encoding='Transformer',
                cls_hidden_dims=config['cls_hidden_dims'],
                train_epoch=config['train_epoch'],
                LR=config['LR'],
                batch_size=config['batch_size']
            )
            model = models.model_initialize(**model_config)
            model.train(self.train, self.val, self.test)
            metrics = self.evaluate_model(model, config)
            if metrics['composite_score'] > self.best_score:
                self.best_score = metrics['composite_score']
                self.best_config = config.copy()
                model.save_model('best_model_enhanced')
                logging.info(f"🌟 New best model saved! Score: {self.best_score:.4f}")
            return metrics
        except Exception as e:
            logging.error(f"❌ Error training configuration {config}: {str(e)}")
            return None

    def smart_grid_search(self, max_trials=50):
        param_grid = self.define_hyperparameter_grid()
        keys, values = list(param_grid.keys()), list(param_grid.values())
        all_combos = list(itertools.product(*values))
        np.random.seed(42)
        np.random.shuffle(all_combos)
        combos = all_combos[:max_trials]

        for idx, combo in enumerate(combos):
            config = dict(zip(keys, combo))
            logging.info(f"\n🔁 Trial {idx+1}/{len(combos)}: {config}")
            metrics = self.train_single_configuration(config)
            if metrics:
                self.results.append(metrics)
                logging.info(f"✅ Test R²: {metrics['test_r2']:.4f}, MSE: {metrics['test_mse']:.6f}, Score: {metrics['composite_score']:.4f}")

        self.results.sort(key=lambda x: x['composite_score'], reverse=True)

    def generate_report(self):
        if not self.results:
            print("No results found.")
            return
        best = self.results[0]
        print("\n🔬 Best Configuration and Performance:")
        for k, v in best['config'].items():
            print(f"{k:20}: {v}")
        print(f"{'Test R²':20}: {best['test_r2']:.4f}")
        print(f"{'Test MSE':20}: {best['test_mse']:.6f}")
        print(f"{'Composite Score':20}: {best['composite_score']:.4f}")

        with open('enhanced_tuning_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("📁 Results saved to 'enhanced_tuning_results.json'")

def main():
    tuner = DTIHyperparameterTuner("D:/Combined Project/dataset.csv")
    tuner.load_and_preprocess_data()
    tuner.smart_grid_search(max_trials=50)
    tuner.generate_report()

if __name__ == "__main__":
    main()
