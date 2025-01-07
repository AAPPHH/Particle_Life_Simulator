import cProfile
import pstats

from main import main

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    with open("profiling_results.txt", "w") as text_file:
        stats = pstats.Stats(profiler, stream=text_file)
        stats.strip_dirs()  # Remove extraneous path information
        stats.sort_stats("cumulative")  # Sort by cumulative time
        stats.print_stats()  # Print statistics to the file

    # Save profiling data in binary format for further analysis
    profiler.dump_stats("profiling_results.prof")

    print("Profiling completed. Results:")
    print("- Readable text file: profiling_results.txt")
    print("- Binary profile file: profiling_results.prof (for SnakeViz or other tools)")
