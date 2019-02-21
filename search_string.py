# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:47:14 2019

@author: kkrao
"""

import os
import glob

# Sets the main directory
main_path = "D:\Krishna\Project\codes"
string = "summer: "

# Gets a list of everything in the main directory including folders
main_directory = os.listdir(main_path)

# This list will hold all of the folders to search through, including the main folder
sub_directories = []

# Adds the main folder to to the list of folders
sub_directories.append(main_path)

# Loops through everthing in the main folder, searching for sub folders
for item in main_directory:
    # Creates the full path to each item)
    item_path = os.path.join(main_path, item)

    # Checks each item to see if it is a directory
    if os.path.isdir(item_path) == True:
        # If it is a folder it is added to the list
        sub_directories.append(item_path)

for directory in sub_directories:
    for files in glob.glob(os.path.join(directory,"*.py")):
        f = open( files, 'r' )
        file_contents = f.read()
        if string in file_contents:
            print f.name