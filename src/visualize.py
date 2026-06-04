# visualize_pareto.py

import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Importamos mplcursors
import mplcursors

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def identify_pareto_front(trials):
    """
    Dado trials: lista de diccionarios con 'values' = [f1, time]
    Retorna una lista de (time, f1) que pertenecen al frente de Pareto.
    """
    pareto_front = []
    for i, trial_i in enumerate(trials):
        f1_i, time_i = trial_i["values"]
        dominated = False
        for j, trial_j in enumerate(trials):
            if i == j:
                continue
            f1_j, time_j = trial_j["values"]
            # trial_j domina a trial_i si f1_j >= f1_i y time_j <= time_i
            # con al menos una de las dos > / < estricto
            if (f1_j >= f1_i and time_j <= time_i) and (f1_j > f1_i or time_j < time_i):
                dominated = True
                break
        if not dominated:
            pareto_front.append((time_i, f1_i))
    return pareto_front

def visualize_pareto(json_file):
    """
    Visualiza el frente de Pareto a partir de un archivo JSON que contiene
    los trials: [ {number, values:[f1, time], params, best_threshold, state}, ... ]
    """
    # 2) Cargamos los datos
    data = load_json(json_file)

    # Extraemos F1/tiempo de todos los trials y construimos una lista de etiquetas
    all_time = []
    all_f1 = []
    labels = []  # Guardará el texto a mostrar al pasar el ratón

    for trial in data:
        f1_val, time_val = trial["values"]
        all_time.append(time_val)
        all_f1.append(f1_val)

        # Construimos la etiqueta para el tooltip.
        # Mostramos algunos parámetros relevantes, limitando decimales.
        trial_number = trial["number"]
        params = trial.get("params", {})
        aggregator = params.get("aggregator", "NA")
        hidden_dim = params.get("hidden_dim", "NA")
        dropout = params.get("dropout_rate", "NA")
        lr = params.get("lr", "NA")

        # Formatear los parámetros para una mejor legibilidad
        label_str = (
            f"Trial #{trial_number}\n"
            f"F1 = {f1_val:.3f}, Time = {time_val:.2f}s\n"
            f"Aggregator = {aggregator}\n"
            f"Hidden Dim = {hidden_dim}\n"
            f"Dropout = {dropout:.3f}\n"
            f"LR = {lr:.5f}"
        )
        labels.append(label_str)

    # Calculamos el frente de Pareto
    pareto_front = identify_pareto_front(data)
    if pareto_front:
        pareto_front_sorted = sorted(pareto_front, key=lambda x: x[0])
        pareto_time, pareto_f1 = zip(*pareto_front_sorted)
    else:
        pareto_time, pareto_f1 = [], []

    # 3) Configuramos estilo de Seaborn
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
    plt.figure(figsize=(12,8))

    # 4) Graficamos todos los trials
    scatter_all = plt.scatter(
        all_time, all_f1,
        c='blue', alpha=0.6, s=120, edgecolors='w', label='Trials'
    )

    # 5) Graficamos Pareto Front
    if pareto_front:
        scatter_pareto = plt.scatter(
            pareto_time, pareto_f1,
            c='red', marker='*', s=300, edgecolors='k', label='Pareto Front'
        )
        plt.plot(
            pareto_time, pareto_f1,
            'r--', linewidth=2
        )

    # 6) Añadimos interacción con mplcursors para los puntos de scatter_all
    cursor = mplcursors.cursor(scatter_all, hover=True)

    # Evento que se lanza al situar el cursor sobre un punto
    @cursor.connect("add")
    def on_add(sel):
        # sel.index indica el índice del punto en scatter_all
        i = sel.index
        sel.annotation.set_text(labels[i])
        sel.annotation.set_fontsize(11)
        # Ajustamos la posición del tooltip
        sel.annotation.xy = (all_time[i], all_f1[i])

    plt.xlabel("Tiempo de Entrenamiento (s)", fontsize=16)
    plt.ylabel("F1 Score (Validación)", fontsize=16)
    plt.title("Frente de Pareto: F1 vs. Tiempo (Hover para detalles)", fontsize=18, fontweight='bold')
    plt.legend(loc='lower right', frameon=False, fontsize=14)
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1)  # asumiendo F1 en [0,1]
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Uso: python visualize_pareto.py <nas_results.json>")
        sys.exit(1)
    json_file = sys.argv[1]
    visualize_pareto(json_file)

if __name__ == "__main__":
    main()

