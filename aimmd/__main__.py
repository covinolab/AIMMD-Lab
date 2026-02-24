"""
AIMMD multiprocessing entry point.

This exists so that multiprocessing.spawn can safely
re-execute the main module.
"""

def main():
    # nothing to do here – real work happens elsewhere
    pass

if __name__ == "__main__":
    main()
