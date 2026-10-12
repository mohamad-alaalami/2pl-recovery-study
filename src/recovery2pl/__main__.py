import sys

from recovery2pl.cli import main

if __name__ == "__main__":      # guard: sampler worker processes re-import the main module
    sys.exit(main())
