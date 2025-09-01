import re
from datetime import datetime
import statistics
import argparse


def analyze_log(log_file_path):
    """
    Parses a log file to calculate the time taken for each particle iteration to
    reach a result (either success or failure). It then computes the mean and
    standard deviation for the successful iterations.

    Args:
        log_file_path (str): The path to the log file.
    """
    # Regex to capture the timestamp, particle ID, and the log message.
    log_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?\[(\d+)\]: (.*)"
    )

    # Dictionary to store the start time of the most recent iteration for each particle.
    iteration_start_times = {}

    # List to store the calculated durations in seconds for successful iterations.
    successful_durations = []

    # Counter for failed (non-promising or infeasible) iterations.
    failure_count = 0

    print("--- Particle Iteration Durations ---")

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    timestamp_str, particle_id_str, message = match.groups()
                    particle_id = int(particle_id_str)

                    # Convert the timestamp string to a datetime object.
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

                    # A "Position" line marks the beginning of a new iteration for a particle.
                    if "Position:" in message:
                        iteration_start_times[particle_id] = timestamp

                    # Check for a successful "fitness improved" message.
                    elif "fitness improved" in message:
                        if particle_id in iteration_start_times:
                            end_time = timestamp
                            start_time = iteration_start_times[particle_id]

                            duration = (end_time - start_time).total_seconds()
                            successful_durations.append(duration)

                            print(
                                f"Particle {particle_id}: SUCCESS - Fitness improved in {duration:.2f} seconds"
                            )
                            # Remove the start time to signify this iteration is complete.
                            del iteration_start_times[particle_id]

                    # Check for non-promising or infeasible particle messages.
                    elif (
                        "Non-promising particle" in message
                        or "Infeasible problem" in message
                        or "Fitness:inf" in line
                    ):
                        if particle_id in iteration_start_times:
                            end_time = timestamp
                            start_time = iteration_start_times[particle_id]

                            duration = (end_time - start_time).total_seconds()
                            failure_count += 1

                            print(
                                f"Particle {particle_id}: FAILURE - Non-promising/Infeasible after {duration:.2f} seconds"
                            )
                            # Remove the start time to signify this iteration is complete.
                            del iteration_start_times[particle_id]

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # --- Synthesize and Print Results ---
    print("\n--- Synthesized Results ---")
    if successful_durations:
        # Calculate mean for successful durations
        mean_duration = statistics.mean(successful_durations)
        print(
            f"Mean time for successful fitness improvement: {mean_duration:.2f} seconds"
        )

        # Calculate standard deviation (requires at least 2 data points)
        if len(successful_durations) > 1:
            std_dev_duration = statistics.stdev(successful_durations)
            print(
                f"Standard Deviation of successful times: {std_dev_duration:.2f} seconds"
            )
        else:
            print("Standard Deviation cannot be calculated with only one data point.")

        print(f"Total successful iterations: {len(successful_durations)}")
    else:
        print("No successful fitness improvement events were found in the log file.")

    print(f"Total failed (non-promising/infeasible) iterations: {failure_count}")


if __name__ == "__main__":
    # Set up argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description="Analyze particle fitness logs.")

    # Add an argument for the log file path.
    # It's a positional argument, meaning it's required.
    parser.add_argument("log_file", help="Path to the log file to be analyzed.")

    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Call the analysis function with the provided file path.
    analyze_log(args.log_file)
