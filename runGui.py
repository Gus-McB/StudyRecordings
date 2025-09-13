#!/usr/bin/env python3
"""
StudyRecordings GUI Launcher
Simple launcher script for the transcript analysis GUI
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from gui_app import main
    main()
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install -r requirements_gui.txt")
except Exception as e:
    print(f"Error starting application: {e}")
    input("Press Enter to exit...")