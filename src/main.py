import os
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UI module
from ui.app import main

if __name__ == "__main__":
    main()
