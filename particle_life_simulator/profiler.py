import cProfile
import pstats
import os

from main import main

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    profiling_dir = "profiling"
    os.makedirs(profiling_dir, exist_ok=True)

    text_file_path = os.path.join(profiling_dir, "profiling_results.txt")
    with open(text_file_path, "w") as text_file:
        stats = pstats.Stats(profiler, stream=text_file)
        stats.strip_dirs()  # Remove unnecessary path information
        stats.sort_stats("cumulative")  # Sort by cumulative execution time
        stats.print_stats()  # Save stats to the text file

    binary_file_path = os.path.join(profiling_dir, "profiling_results.prof")
    profiler.dump_stats(binary_file_path)

    print("Profiling completed. Results:")
    print(f"- Readable text file: {text_file_path}")
    print(f"- Binary profile file: {binary_file_path} (for SnakeViz or other tools)")