# script for google drive image upload performance test.
import time
import os
import sys
sys.path.append("/Users/changbeankang/Claw_For_Humanity/HOS_II/Google_Gspread_Drive_Manager")
import gspread_drive_manager as gsmanager


current_ws = os.getcwd()
gsmanager.initialize.__init__(f'{current_ws}')