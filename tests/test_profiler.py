import os
import sys
import cProfile
import pstats
import io
import shutil
import pytest

# Ensure `main()` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "particle_life_simulator")))

try:
    from particle_life_simulator.main import main  # Import the main function
except ModuleNotFoundError as e:
    print(f"Module import failed: {e}")
    sys.exit(1)

# Define paths for profiling results
profiling_dir = "profiling"
text_file_path = os.path.join(profiling_dir, "profiling_results.txt")
binary_file_path = os.path.join(profiling_dir, "profiling_results.prof")

def run_profiling():
    """Runs profiling before tests and generates output files."""
    os.makedirs(profiling_dir, exist_ok=True)

    # Backup old profiling results
    if os.path.exists(text_file_path):
        shutil.move(text_file_path, text_file_path + ".backup")
    if os.path.exists(binary_file_path):
        shutil.move(binary_file_path, binary_file_path + ".backup")

    profiler = cProfile.Profile()
    try:
        profiler.enable()
    except ValueError:
        pytest.skip("Skipping test: Another profiling tool is already active")

    main() 

    profiler.disable()

    with open(text_file_path, "w") as text_file: 
        text_file.write("\n==== NEW PROFILING RUN ====\n")
        stats = pstats.Stats(profiler, stream=text_file)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()

    # Save profiling results in a binary file 
    profiler.dump_stats(binary_file_path)

def test_tkinter_availability():
    """Checks if Tkinter is available to prevent `_tkinter.TclError`."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
    except Exception:
        pytest.skip("Skipping test: Tkinter is not available or misconfigured.")

def test_profiling_directory():
    """Checks if the profiling directory is created."""
    run_profiling()
    assert os.path.exists(profiling_dir), "Profiling directory was not created"

def test_profiling_file_content():
    """Ensures that the profiling output files exist and contain data."""
    run_profiling()
    assert os.path.exists(text_file_path) and os.path.getsize(text_file_path) > 0, "Profiling text file is missing or empty"
    assert os.path.exists(binary_file_path) and os.path.getsize(binary_file_path) > 0, "Profiling binary file is missing or empty"

def test_profiling_execution():
    """Ensures that profiling correctly captures function execution."""
    profiler = cProfile.Profile()
    
    try:
        profiler.enable()  
    except ValueError:
        pytest.skip("Skipping test: Another profiling tool is already active")

    # Simple function for testing
    def dummy_function():
        return sum(range(1000))

    dummy_function()

    profiler.disable()

    # Save profiling results in memory and check function presence
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.print_stats()

    assert "dummy_function" in s.getvalue(), "Profiling results do not contain dummy_function"
