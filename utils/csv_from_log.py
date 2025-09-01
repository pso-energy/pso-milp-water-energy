import re
import csv


def csv_from_log(scenario_log_path: str, log_fn: str, csv_fn: str):
    """
    Esegue il parsing di un file di log per estrarre i dati di costo e posizione
    di ogni particella e li salva in un file CSV.

    Args:
        scenario_log_path (str): Il percorso della cartella contenente il file di log.
        log_file_path (str): Il nome del file di log da analizzare.
        csv_file_path (str): Il nome del file CSV di output.
    """

    log_file_path = f"{scenario_log_path}/{log_fn}"
    csv_file_path = f"{scenario_log_path}/{csv_fn}"

    # Espressione regolare per trovare l'ID della particella e il costo
    # NOTA: Gestisce numeri interi, decimali e anche il valore 'inf'.
    fitness_pattern = re.compile(
        r"\[(?P<particle>\d+)\]: Fitness:(?P<cost>[\d\.\+\-inf]+)"
    )

    # Espressione regolare per trovare l'ID della particella e il vettore di posizione
    position_pattern = re.compile(
        r"\[(?P<particle>\d+)\]: Position: \[(?P<x_values>.*?)\]"
    )

    last_positions = [0] * 28
    output_data = []
    iteration_counter = [0] * 28
    max_x_len = 0

    print(f"Analisi del file di log: {log_file_path}...")

    with open(log_file_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            # Cerca prima una riga di posizione per aggiornare lo stato della particella
            pos_match = position_pattern.search(line)
            if pos_match:
                particle_id = pos_match.group("particle")
                x_values_str = pos_match.group("x_values")
                # Pulisce e memorizza i valori di x come lista di stringhe
                x_values = [val.strip() for val in x_values_str.split(",")]
                last_positions[int(particle_id)] = x_values
                continue  # Passa alla riga successiva

            # Se non è una riga di posizione, cerca una riga di costo (Fitness)
            fit_match = fitness_pattern.search(line)
            if fit_match:
                particle_id = fit_match.group("particle")
                cost = fit_match.group("cost")

                # Crea una riga per il csv
                iteration_counter[int(particle_id)] += 1
                position = last_positions[int(particle_id)]
                row = [
                    iteration_counter[int(particle_id)],
                    particle_id,
                    cost,
                ] + position
                output_data.append(row)

                # Aggiorna la lunghezza massima del vettore x per l'header
                if len(position) > max_x_len:
                    max_x_len = len(position)

    if not output_data:
        print(
            "Nessun dato valido trovato. Controlla il formato del log e le espressioni regolari."
        )
        return

    # --- MODIFICATION START ---
    # Sort the data by the first column (iteration) and then the second (particle)
    # The key specifies sorting by the first element (row[0]) then the integer value of the second (row[1])
    print("Ordinamento dei dati...")
    output_data.sort(key=lambda row: (row[0], int(row[1])))
    # --- MODIFICATION END ---

    # Crea l'header del CSV in modo dinamico
    header = ["iteration", "particle", "fitness"] + [f"x_{i}" for i in range(max_x_len)]

    # Scrive i dati nel file CSV
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(output_data)

    print(
        f"✅ File CSV '{csv_file_path}' creato e ordinato con successo. Contiene {len(output_data)} righe."
    )


# --- Esecuzione dello script ---
if __name__ == "__main__":
    scenario_log_path = "simulations_unzipped/2024-12-07T21-09_FULL_x1/log"
    log_file_name = "main.log"
    csv_file_name = "particles_log.csv"

    # Esegui la funzione di parsing
    csv_from_log(scenario_log_path, log_file_name, csv_file_name)
