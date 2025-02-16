import os
import sys
import cProfile
import pstats
import io
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "particle_life_simulator")))

from particle_life_simulator.main import main

# Define paths for profiling results
profiling_dir = "profiling"
text_file_path = os.path.join(profiling_dir, "profiling_results.txt")
binary_file_path = os.path.join(profiling_dir, "profiling_results.prof")


@pytest.fixture
def run_profiling():
    """Runs profiling before tests to generate output files."""
    os.makedirs(profiling_dir, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    # Save profiling results in a text file
    with open(text_file_path, "w") as text_file:
        stats = pstats.Stats(profiler, stream=text_file)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()

    # Save profiling results in a binary file
    profiler.dump_stats(binary_file_path)
    return profiler



def test_profiling_directory(run_profiling):
    """Checks if the profiling directory is created automatically."""
    assert os.path.exists(profiling_dir), "Profiling directory was not created"


def test_profiling_file_content(run_profiling):
    """Ensures that the profiling output files contain data."""
    assert os.path.exists(text_file_path) and os.path.getsize(text_file_path) > 0, "Profiling text file is missing or empty"
    assert os.path.exists(binary_file_path) and os.path.getsize(binary_file_path) > 0, "Profiling binary file is missing or empty"


def test_profiling_execution():
    """Ensures that profiling correctly captures function execution."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    def dummy_function():
        return sum(range(1000))

    dummy_function()
    
    profiler.disable()

    # Save profiling results in a string
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.print_stats()

    assert "dummy_function" in s.getvalue(), "Profiling results do not contain dummy_function"
