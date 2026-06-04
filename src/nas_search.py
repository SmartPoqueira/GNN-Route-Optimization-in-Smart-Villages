# nas_search.py

import os
import ast
import json
import time
import random
import yaml
import numpy as np
import pandas as pd
import torch

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc, f1_score
)

from src.model import ImprovedMPNN_NAS

class GraphClassifierNAS:
    """
    Manages the full NAS pipeline:
    1. Data loading and preprocessing (CSV -> Graphs).
    2. Dataset creation (train, val, test splits).
    3. Multi-objective NAS with Optuna (maximize F1, minimize time) over ImprovedMPNN_NAS.
    4. Saves results to a .json file and prints the Pareto front.
    """

    def __init__(self, config_path: str):
        # Load YAML config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get('seed', 42)
        self.set_seed(self.seed)

        self.csv_path = self.config['csv_path']
        if not os.path.exists(self.csv_path):
            print(f"[WARNING] Database path '{self.csv_path}' not found. Generating synthetic route dataset...")
            self.generate_synthetic_data()

        self.df = pd.read_csv(self.csv_path)

        # Undersampling parameters
        self.undersampling = self.config.get('undersampling', False)
        self.undersampling_ratio = self.config.get('undersampling_ratio', [70, 30])

        # Modelling parameters
        self.training_params = self.config['training_params']

        os.makedirs('results', exist_ok=True)
        self.preprocess_labels()
        self.node_mapping = self.create_node_mapping()
        self.num_node_features = len(self.node_mapping) + 1  # +1 for direction
        self.num_edge_features = 1  # travel times

        self.graph_objects = self.create_graph_objects()
        self.train_dataset, self.test_dataset, self.validation_dataset = self.split_data()

        if self.undersampling:
            self.train_dataset = self.undersample_data(self.train_dataset, self.undersampling_ratio)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # NAS parameters
        self.n_trials = self.training_params.get('n_trials_for_nas', 5)
        self.epochs = self.training_params.get('epochs', 30)
        self.patience = self.training_params.get('patience', 10)
        self.output_json = "nas_results.json"

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate_synthetic_data(self):
        """
        Generates a synthetic route dataset to run the experiments without the real database.
        """
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        import random
        random.seed(self.seed)
        
        data = []
        cameras = ['PAM1', 'PAM2', 'BUB', 'CAP']
        
        for i in range(1000):
            route_len = random.randint(2, 5)
            route = [random.choice(cameras) for _ in range(route_len)]
            times = [round(random.uniform(5.0, 120.0), 2) for _ in range(route_len - 1)]
            directions = [random.choice([0, 1]) for _ in range(route_len)]
            
            # Simple rule for repeat visitor (label = 1)
            repeater = 1 if ('CAP' in route or sum(times) < 150.0) and random.random() > 0.3 else 0
            
            data.append({
                'num_plate': f"VEH{i:04d}",
                'route': str(route),
                'times': str(times),
                'directions': str(directions),
                'repeater': repeater
            })
            
        df_syn = pd.DataFrame(data)
        df_syn.to_csv(self.csv_path, index=False)
        print(f"[INFO] Generated 1000 synthetic routes and saved to '{self.csv_path}'.")

    def preprocess_labels(self):
        """Preprocesses the label column."""
        self.df['label'] = self.df['repeater'].astype(int)
        class_counts = self.df['label'].value_counts()
        plt.figure()
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks([0,1], ['Non-Repeater', 'Repeater'])
        plt.savefig('results/class_distribution.png')
        plt.close()

    def create_node_mapping(self):
        """Creates a node-to-index mapping."""
        node_set = set()
        for route_str in self.df['route']:
            route = ast.literal_eval(route_str)
            for node in route:
                node_set.add(node)
        node_mapping = {node: idx for idx, node in enumerate(sorted(node_set))}
        return node_mapping

    def row_to_graph(self, row):
        route = ast.literal_eval(row['route'])
        times = ast.literal_eval(row['times'])
        directions = ast.literal_eval(row['directions'])
        num_nodes = len(route)

        edge_index, edge_attr = [], []
        for i in range(num_nodes - 1):
            source = i
            target = i + 1
            edge_index.append([source, target])
            edge_attr.append([times[i]])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr = torch.empty((0,1), dtype=torch.float)

        # Node features
        num_unique_nodes = len(self.node_mapping)
        node_type_features = torch.zeros((num_nodes, num_unique_nodes), dtype=torch.float)
        direction_features = torch.zeros((num_nodes,1), dtype=torch.float)

        for i, node in enumerate(route):
            idx_map = self.node_mapping[node]
            node_type_features[i, idx_map] = 1.0
            direction_features[i] = directions[i]

        x = torch.cat([node_type_features, direction_features], dim=1)
        y = torch.tensor([row['label']], dtype=torch.float)
        data_g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data_g

    def create_graph_objects(self):
        """Converts each DataFrame row into a PyG graph object."""
        graphs = []
        for _, row in self.df.iterrows():
            g = self.row_to_graph(row)
            graphs.append(g)
        return graphs

    def split_data(self):
        """Splits dataset into train, test, val with 70/15/15 proportions."""
        train_val_data, test_data = train_test_split(
            self.graph_objects,
            test_size=0.15,
            random_state=self.seed,
            stratify=[g.y.item() for g in self.graph_objects]
        )

        train_data, val_data = train_test_split(
            train_val_data,
            test_size=0.1765,
            random_state=self.seed,
            stratify=[g.y.item() for g in train_val_data]
        )

        return train_data, test_data, val_data

    def undersample_data(self, dataset, ratio):
        """
        Performs undersampling on 'dataset' assuming two classes.
        ratio = [majority%, minority%], e.g. [70, 30].
        """
        if len(ratio) != 2:
            raise ValueError("undersampling_ratio must be [majority_percent, minority_percent].")
        labels = [g.y.item() for g in dataset]
        class_counts = pd.Series(labels).value_counts()
        if len(class_counts) != 2:
            raise ValueError("Expected exactly 2 classes for undersampling.")

        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        majority_data = [g for g in dataset if g.y.item() == majority_class]
        minority_data = [g for g in dataset if g.y.item() == minority_class]

        n_minority = len(minority_data)
        desired_majority = int((ratio[0]/ratio[1]) * n_minority)
        if desired_majority > len(majority_data):
            # Scale down the minority class instead to maintain the requested ratio
            desired_majority = len(majority_data)
            n_minority = int((ratio[1]/ratio[0]) * desired_majority)
            minority_data = random.sample(minority_data, n_minority)
        majority_downsampled = random.sample(majority_data, desired_majority)
        balanced_dataset = majority_downsampled + minority_data
        random.shuffle(balanced_dataset)
        print(f"[DEBUG] Undersampling: majority={len(majority_data)}->{desired_majority}, minority={len(minority_data)}.")
        return balanced_dataset

    def run_multiobjective_experiment(self):
        """
        Runs multi-objective NAS with Optuna and saves results to .json.
        """
        def objective(trial: Trial):
            return self.objective_multi(trial)

        sampler = TPESampler()
        pruner = HyperbandPruner(min_resource=1, max_resource=self.epochs)

        study = optuna.create_study(
            directions=["maximize", "minimize"],  # (F1 Score, Time)
            sampler=sampler,
            pruner=pruner
        )
        study.optimize(objective, n_trials=self.n_trials)

        all_trials = []
        for t in study.trials:
            best_threshold_stored = t.user_attrs.get("best_threshold", None)

            trial_info = {
                "number": t.number,
                "values": t.values,
                "params": t.params,
                "best_threshold": best_threshold_stored,
                "state": str(t.state)
            }
            all_trials.append(trial_info)

        with open(self.output_json, "w") as f:
            json.dump(all_trials, f, indent=2)

        print("\n=== Pareto Front (best trials) ===")
        for bt in study.best_trials:
            threshold_bt = bt.user_attrs.get("best_threshold", None)
            print(f" Trial #{bt.number}")
            print(f"   F1 Score: {bt.values[0]:.4f}")
            print(f"   Time (s): {bt.values[1]:.2f}")
            print(f"   Params:   {bt.params}")
            print(f"   Best Threshold: {threshold_bt}\n")

        print(f"File '{self.output_json}' saved with all trial results.")
        return study

    def objective_multi(self, trial: Trial):
        """
        Returns (F1_on_val, TrainingTime) for the sampled architecture.
        """
        node_mlp_layers = trial.suggest_int('node_mlp_layers', 1, 3)
        edge_mlp_layers = trial.suggest_int('edge_mlp_layers', 1, 3)
        gnn_layers = trial.suggest_int('gnn_layers', 1, 4)
        msg_mlp_layers = trial.suggest_int('message_mlp_layers', 0, 3)
        final_mlp_layers = trial.suggest_int('final_mlp_layers', 1, 3)

        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
        aggregator = trial.suggest_categorical('aggregator', ['mean', 'add', 'max'])
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        activation = trial.suggest_categorical('activation', ['ReLU', 'GELU', 'LeakyReLU'])
        use_residual = trial.suggest_categorical('use_residual', [True, False])

        # Undersampling as a hyperparameter
        use_undersampling = trial.suggest_categorical('use_undersampling', [True, False])

        model_params = {
            'model_name': 'E-GAT',
            'hidden_dim': hidden_dim,
            'dropout': dropout_rate,
            'use_batchnorm': use_batchnorm,
            'aggregation': aggregator,
            'node_mlp_layers': node_mlp_layers,
            'edge_mlp_layers': edge_mlp_layers,
            'gnn_layers': gnn_layers,
            'message_mlp_layers': msg_mlp_layers,
            'final_mlp_layers': final_mlp_layers,
            'use_residual': use_residual,
            'activation': activation
        }

        model = ImprovedMPNN_NAS(
            num_node_features=self.num_node_features,
            num_edge_features=self.num_edge_features,
            model_params=model_params
        ).to(self.device)

        # Apply undersampling if enabled
        if use_undersampling:
            train_dataset = self.undersample_data(self.train_dataset, [70, 30])
        else:
            train_dataset = self.train_dataset

        g = torch.Generator()
        g.manual_seed(self.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_params['batch_size'],
            shuffle=True,
            generator=g
        )
        val_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.training_params['batch_size'],
            shuffle=False
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        start_time = time.time()
        best_val_f1, best_threshold = self.train_and_evaluate_for_trial(
            model, optimizer, criterion, train_loader, val_loader
        )
        end_time = time.time()
        train_time = end_time - start_time

        trial.set_user_attr("best_threshold", float(best_threshold))

        return (best_val_f1, train_time)

    def train_and_evaluate_for_trial(self, model, optimizer, criterion, train_loader, val_loader):
        """
        Trains and evaluates the model for 'self.epochs' epochs.
        Logs Train Loss, Validation Loss, and Best Threshold per epoch.
        Returns (best_val_f1, best_threshold_on_val).
        """
        best_val_f1 = 0.0
        best_threshold_overall = 0.5
        trigger_times = 0

        for epoch in range(self.epochs):
            # ----- TRAINING PHASE -----
            model.train()
            total_loss = 0.0
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out.view(-1), data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            # ----- VALIDATION PHASE -----
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    val_data = val_data.to(self.device)
                    val_out = model(val_data.x, val_data.edge_index, val_data.edge_attr, val_data.batch)
                    val_loss_batch = criterion(val_out.view(-1), val_data.y)
                    val_loss += val_loss_batch.item()
            avg_val_loss = val_loss / len(val_loader)

            # ----- FIND THRESHOLD THAT MAXIMISES F1 ON VALIDATION -----
            threshold_this_epoch, val_f1 = self.evaluate_best_threshold_and_f1(model, val_loader)

            # ----- LOG EPOCH RESULTS -----
            print(f"Epoch {epoch+1}, "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}, "
                  f"Best Threshold: {threshold_this_epoch:.4f}")

            # ----- EARLY STOPPING ON F1 -----
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_threshold_overall = threshold_this_epoch
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print("Early stopping: patience reached.")
                    break

        return best_val_f1, best_threshold_overall

    def evaluate_best_threshold_and_f1(self, model, loader):
        """
        Iterates over the validation loader, collects probabilities,
        computes the precision-recall curve, and extracts the threshold
        that maximises F1. Returns (best_threshold, f1_at_best_threshold).
        """
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                probs = torch.sigmoid(out.view(-1)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(data.y.cpu().numpy())

        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        if len(f1_scores) == 0:
            return 0.5, 0.0

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        return best_threshold, best_f1

    def run_all_models_comparison(self):
        """
        Trains and evaluates all 10 GNN models on the dataset to print Table V.
        """
        models = ['GCN', 'E-GCN', 'GraphSAGE', 'E-GraphSAGE', 'GGNN', 'E-GGNN', 'EdgeConv', 'MPNN', 'GAT', 'E-GAT']
        results = {}

        model_params = {
            'hidden_dim': 128,
            'dropout': 0.3,
            'use_batchnorm': True,
            'aggregation': 'mean',
            'node_mlp_layers': 2,
            'edge_mlp_layers': 2,
            'gnn_layers': 2,
            'message_mlp_layers': 2,
            'final_mlp_layers': 2,
            'use_residual': True,
            'activation': 'ReLU'
        }

        print("\n=== Training and Evaluating All 10 GNN Models (Table V) ===")

        for m_name in models:
            print(f"\nTraining Model: {m_name}...")
            model = ImprovedMPNN_NAS(
                num_node_features=self.num_node_features,
                num_edge_features=self.num_edge_features,
                model_params={**model_params, 'model_name': m_name}
            ).to(self.device)

            train_loader = DataLoader(self.train_dataset, batch_size=self.training_params['batch_size'], shuffle=True)
            val_loader = DataLoader(self.validation_dataset, batch_size=self.training_params['batch_size'], shuffle=False)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001)

            best_f1 = 0.0
            best_thresh = 0.5
            # Train for 5 epochs for comparison
            for epoch in range(5):
                model.train()
                for data in train_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    loss = criterion(out.view(-1), data.y)
                    loss.backward()
                    optimizer.step()

                thresh, val_f1 = self.evaluate_best_threshold_and_f1(model, val_loader)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_thresh = thresh

            print(f"Model {m_name} - Best Validation F1: {best_f1:.4f} (threshold: {best_thresh:.2f})")
            results[m_name] = {'F1': best_f1, 'Threshold': best_thresh}

        print("\n=== Model Comparison Summary ===")
        for m_name, res in results.items():
            print(f"{m_name:<15}: F1={res['F1']:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--compare', action='store_true', help='Compare all 10 models instead of running NAS')
    args = parser.parse_args()

    nas = GraphClassifierNAS(args.config)
    if args.compare:
        nas.run_all_models_comparison()
    else:
        nas.run_multiobjective_experiment()

if __name__ == "__main__":
    main()
