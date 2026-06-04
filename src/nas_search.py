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

from improvedMPNN_nas import ImprovedMPNN_NAS

class GraphClassifierNAS:
    """
    Clase que gestiona:
    1. Carga y preprocesamiento de datos (CSV -> Grafos).
    2. Creación del dataset (train, val, test).
    3. NAS multiobjetivo con Optuna (max F1, min tiempo) sobre ImprovedMPNN_NAS.
    4. Guarda resultados en un .json y muestra el frente de Pareto en consola.
    """

    def __init__(self, config_path: str):
        # Leer archivo YAML
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get('seed', 42)
        self.set_seed(self.seed)

        self.csv_path = self.config['csv_path']
        self.df = pd.read_csv(self.csv_path)

        # Parámetros undersampling
        self.undersampling = self.config.get('undersampling', False)
        self.undersampling_ratio = self.config.get('undersampling_ratio', [70, 30])

        # Parámetros de modelado
        self.training_params = self.config['training_params']

        self.preprocess_labels()
        self.node_mapping = self.create_node_mapping()
        self.num_node_features = len(self.node_mapping) + 1  # +1 para direction
        self.num_edge_features = 1  # times

        self.graph_objects = self.create_graph_objects()
        self.train_dataset, self.test_dataset, self.validation_dataset = self.split_data()

        if self.undersampling:
            self.train_dataset = self.undersample_data(self.train_dataset, self.undersampling_ratio)

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Parámetros de NAS
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

    def preprocess_labels(self):
        """ Preprocesa la columna de etiquetas. """
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
        """ Crea un mapeo node->indice. """
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

        # Features de nodos
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
        """ Convierte cada fila del DF en un objeto grafo PyG. """
        graphs = []
        for _, row in self.df.iterrows():
            g = self.row_to_graph(row)
            graphs.append(g)
        return graphs

    def split_data(self):
        """ Divide dataset en train, test, val con proporciones 70%, 15%, 15%. """
        # Primera división: 85% para entrenamiento + validación, 15% para prueba
        train_val_data, test_data = train_test_split(
            self.graph_objects,
            test_size=0.15,  # Cambiado de 0.2 a 0.15
            random_state=self.seed,
            stratify=[g.y.item() for g in self.graph_objects]
        )
        
        # Segunda división: aproximadamente 82.35% de train_val para entrenamiento, 17.65% para validación
        # Esto resulta en 70% entrenamiento y 15% validación del total original
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=0.1765,  # Cambiado de 0.1 a 0.1765
            random_state=self.seed,
            stratify=[g.y.item() for g in train_val_data]
        )
        
        return train_data, test_data, val_data


    def undersample_data(self, dataset, ratio):
        """
        Realiza undersampling en 'dataset' asumiendo dos clases.
        ratio = [majority%, minority%], p. ej. [70, 30].
        """
        if len(ratio) != 2:
            raise ValueError("undersampling_ratio debe ser [majority_percent, minority_percent].")
        labels = [g.y.item() for g in dataset]
        class_counts = pd.Series(labels).value_counts()
        if len(class_counts) != 2:
            raise ValueError("Se esperaban exactamente 2 clases para undersampling.")
        
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        majority_data = [g for g in dataset if g.y.item() == majority_class]
        minority_data = [g for g in dataset if g.y.item() == minority_class]

        n_minority = len(minority_data)
        desired_majority = int((ratio[0]/ratio[1]) * n_minority)
        if desired_majority > len(majority_data):
            raise ValueError("La proporción deseada excede las muestras disponibles de la clase mayoritaria.")
        majority_downsampled = random.sample(majority_data, desired_majority)
        balanced_dataset = majority_downsampled + minority_data
        random.shuffle(balanced_dataset)
        print(f"[DEBUG] Undersampling: majority={len(majority_data)}-> {desired_majority}, minority={len(minority_data)}.")
        return balanced_dataset

    def run_multiobjective_experiment(self):
        """
        Ejecuta la búsqueda multiobjetivo con Optuna y guarda resultados en .json.
        """
        def objective(trial: Trial):
            return self.objective_multi(trial)

        sampler = TPESampler()
        pruner = HyperbandPruner(min_resource=1, max_resource=self.epochs)

        study = optuna.create_study(
            directions=["maximize", "minimize"],  # (F1 Score, Tiempo)
            sampler=sampler,
            pruner=pruner
        )
        study.optimize(objective, n_trials=self.n_trials)

        all_trials = []
        for t in study.trials:
            # Recuperamos el threshold guardado en user_attrs
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

        print("\n=== Pareto Front (mejores) ===")
        for bt in study.best_trials:
            # También tomamos el threshold del best_trial
            threshold_bt = bt.user_attrs.get("best_threshold", None)
            print(f" Trial #{bt.number}")
            print(f"   F1 Score: {bt.values[0]:.4f}")
            print(f"   Time (s): {bt.values[1]:.2f}")
            print(f"   Params:   {bt.params}")
            print(f"   Best Threshold: {threshold_bt}\n")

        print(f"Archivo '{self.output_json}' guardado con las ejecuciones.")
        return study

    def objective_multi(self, trial: Trial):
        """
        Retorna (F1enVal, TiempoEntrenamiento) para la arquitectura generada.
        """
        node_mlp_layers = trial.suggest_int('node_mlp_layers', 1, 3)
        edge_mlp_layers = trial.suggest_int('edge_mlp_layers', 1, 3)
        msg_mlp_layers  = trial.suggest_int('message_mlp_layers', 1, 3)
        final_mlp_layers= trial.suggest_int('final_mlp_layers', 1, 3)

        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
        aggregator = trial.suggest_categorical('aggregator', ['mean','add','max'])
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        # Nuevos hiperparámetros
        use_attention = trial.suggest_categorical('use_attention', [True, False])
        residual_connection = trial.suggest_categorical('residual_connection', [True, False])
        
        # **Nuevo Hiperparámetro: Uso de Undersampling**
        use_undersampling = trial.suggest_categorical('use_undersampling', [True, False])

        model_params = {
            'hidden_dim': hidden_dim,
            'dropout': dropout_rate,
            'use_batchnorm': use_batchnorm,
            'aggregation': aggregator,
            'node_mlp_layers': node_mlp_layers,
            'edge_mlp_layers': edge_mlp_layers,
            'message_mlp_layers': msg_mlp_layers,
            'final_mlp_layers': final_mlp_layers,
            'use_attention': use_attention,
            'residual_connection': residual_connection
        }

        model = ImprovedMPNN_NAS(
            num_node_features=self.num_node_features,
            num_edge_features=self.num_edge_features,
            model_params=model_params
        ).to(self.device)

        # **Aplicar Undersampling si está habilitado**
        if use_undersampling:
            # Aplicar undersampling con la proporción fija [70, 30]
            train_dataset = self.undersample_data(self.train_dataset, [70, 30])
            print("[INFO] Undersampling aplicado al conjunto de entrenamiento.")
        else:
            train_dataset = self.train_dataset

        # Crear DataLoaders con el dataset balanceado o original
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

        # Guardamos el threshold como atributo del trial
        trial.set_user_attr("best_threshold", float(best_threshold))
        
        return (best_val_f1, train_time)



    def train_and_evaluate_for_trial(self, model, optimizer, criterion, train_loader, val_loader):
        """
        Entrena y evalúa el modelo por 'self.epochs' épocas.
        Muestra Train Loss, Validation Loss, y el Best Threshold.
        Retorna (best_val_f1, best_threshold_en_val).
        """
        best_val_f1 = 0.0
        best_threshold_overall = 0.5
        trigger_times = 0

        for epoch in range(self.epochs):
            # ----- FASE DE ENTRENAMIENTO -----
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

            # ----- FASE DE VALIDACIÓN: calculamos val_loss -----
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    val_data = val_data.to(self.device)
                    val_out = model(val_data.x, val_data.edge_index, val_data.edge_attr, val_data.batch)
                    val_loss_batch = criterion(val_out.view(-1), val_data.y)
                    val_loss += val_loss_batch.item()
            avg_val_loss = val_loss / len(val_loader)

            # ----- HALLAR EL UMBRAL QUE MAXIMIZA F1 EN VALIDACIÓN -----
            threshold_this_epoch, val_f1 = self.evaluate_best_threshold_and_f1(model, val_loader)

            # ----- IMPRIMIR LOS RESULTADOS DE ESTA ÉPOCA -----
            print(f"Epoch {epoch+1}, "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}, "
                  f"Best Threshold: {threshold_this_epoch:.4f}")

            # ----- EARLY STOPPING SEGÚN F1 (OPCIONAL) -----
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_threshold_overall = threshold_this_epoch
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print("Early stopping por paciencia alcanzada.")
                    break

        return best_val_f1, best_threshold_overall


    def evaluate_best_threshold_and_f1(self, model, loader):
        """
        Recorre el loader (validación), obtiene probabilidades,
        calcula la curva precision-recall, extrae el threshold que maximiza F1.
        Retorna (best_threshold, f1_al_mejor_threshold).
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

        # Curva precision-recall
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        # Evitamos division por cero
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        if len(f1_scores) == 0:  # En caso de no poder calcular
            return 0.5, 0.0

        best_idx = np.argmax(f1_scores)
        # Si best_idx coincide con el último índice (sin threshold), usar 0.5 por defecto
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        return best_threshold, best_f1

