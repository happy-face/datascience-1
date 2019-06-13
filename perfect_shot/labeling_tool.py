

"""
 One-click image sorting/labelling script. Copies or moves images from a folder into subfolders.
 This script launches a GUI which displays one image after the other and lets the user give different labels
 from a list provided as input to the script. In contrast to original version, version 2 allows for
 relabelling and keeping track of the labels.
 Provides also short-cuts - press "1" to put into "label 1", press "2" to put into "label 2" a.s.o.

 USAGE:
 run 'python sort_folder_vers2.py' or copy the script in a jupyter notebook and run then

 you need also to provide your specific input (source folder, labels and other) in the preamble
 original Author: Christian Baumgartner (c.baumgartner@imperial.ac.uk)
 changes, version 2: Nestor Arsenov (nestorarsenov_AT_gmail_DOT_com)
 Date: 24. Dec 2018
"""

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to the input folder")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset CSV file path. If this file doesn't exist it will get created. If it exists we will load it and contiue labeling.")
    parser.add_argument("-mw", "--max-width", type=int, default=1700, help="Max width for image so it fits the screen")
    parser.add_argument("-mh", "--max-height", type=int, default=900, help="Max height for image so it fits the screen")
    return parser.parse_args()


# Define global variables, which are to be changed by user:

# In[5]:


##### added in version 2

# the folder in which the pictures that are to be sorted are stored
# don't forget to end it with the sign '/' !
input_folder = '/media/julia/stovariste/dataset/'

# the different folders into which you want to sort the images, e.g. ['cars', 'bikes', 'cats', 'horses', 'shoes']
labels = ["discard", "keep"]
colors = ["yellow", "green"]
df_path = 'dataset_out.csv'

img_max_width = 1700
img_max_height = 900

# a selection of what file-types to be sorted, anything else will be excluded
file_extensions = ['.JPG', '.jpg', '.png', '.whatever']
#####


# In[8]:


import pandas as pd
import os
import numpy as np

import argparse
import tkinter as tk
import os
from shutil import copyfile, move
from PIL import ImageTk, Image

class ImageGui:
    """
    GUI for iFind1 image sorting. This draws the GUI and handles all the events.
    Useful, for sorting views into sub views or for removing outliers from the data.
    """

    def __init__(self, master, labels, colors, paths):
        """
        Initialise GUI
        :param master: The parent window
        :param labels: A list of labels that are associated with the images
        :param paths: A list of file paths to images
        :return:
        """

        # So we can quit the window from within the functions
        self.master = master

        # Extract the frame so we can draw stuff on it
        frame = tk.Frame(master)

        # Initialise grid
        frame.grid()

        # Start at the first file name
        self.index = 0
        self.paths = paths
        self.labels = labels
        self.colors = colors
        #### added in version 2
        self.sorting_label = 'unsorted'
        ####

        # Number of labels and paths
        self.n_labels = len(labels)
        self.n_paths = len(paths)

        # Set empty image container
        self.image_raw = None
        self.image = None
        self.image_panel = tk.Label(frame)

        # set image container to first image
        self.set_image(paths[self.index])

        # Make buttons
        self.buttons = []
        for label in labels:
            self.buttons.append(
                    tk.Button(frame, text=label, width=10, height=2, fg='blue', command=lambda l=label: self.vote(l))
            )

        ### added in version 2
        self.buttons.append(tk.Button(frame, text="prev im", width=10, height=2, fg="green", command=lambda l=label: self.move_prev_image()))
        self.buttons.append(tk.Button(frame, text="next im", width=10, height=2, fg='green', command=lambda l=label: self.move_next_image()))
        ###

        # Add progress label
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label = tk.Label(frame, text=progress_string, width=10)

        # Place buttons in grid
        for ll, button in enumerate(self.buttons):
            button.grid(row=0, column=ll, sticky='we')
            #frame.grid_columnconfigure(ll, weight=1)

        # Place progress label in grid
        self.progress_label.grid(row=1, column=0, sticky='we') # +2, since progress_label is placed after
                                                                            # and the additional 2 buttons "next im", "prev im"

        #### added in version 2
        # Add sorting label
        self.sorting_label = tk.Label(frame, text=("label: %s" % (df.label[self.index])), width=15)
        # Place sorting label in grid
        self.sorting_label.grid(row=1, column=1, sticky='we') # +2, since progress_label is placed after
                                                                            # and the additional 2 buttons "next im", "prev im"

        # Place typing input in grid, in case the mode is 'copy'
        tk.Label(frame, text="go to #pic:").grid(row=1, column=2)

        self.return_ = tk.IntVar() # return_-> self.index
        self.return_entry = tk.Entry(frame, width=6, textvariable=self.return_)
        self.return_entry.grid(row=1, column=3, sticky='we')
        master.bind('<Return>', self.num_pic_type)


        # Place the image in grid
        self.image_panel.grid(row=2, column=0, columnspan=max(4, self.n_labels+1), sticky='we')

        # key bindings (so number pad can be used as shortcut)
        # make it not work for 'copy', so there is no conflict between typing a picture to go to and choosing a label with a number-key
        for key in range(self.n_labels):
            master.bind(str(key+1), self.vote_key)
        master.bind('<Right>', self.right_key)
        master.bind('<Left>', self.left_key)
        master.bind('t', self.toggle_vote_key)


    ### added in version 2
    def move_prev_image(self):
        """
        Displays the prev image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        self.index -= 1
        self.update_panel()

    ### added in version 2
    def move_next_image(self):
        """
        Displays the next image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        self.index += 1
        self.update_panel()

    def update_panel(self):
        progress_string = "%s %d/%d" % (df.set_name[self.index], self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)

        label = df.label[self.index]
        label_index = self.labels.index(label)
        self.sorting_label.configure(text=("label: %s" % (label)), bg=self.colors[label_index])

        if self.index < self.n_paths:
            self.set_image(df.full_path[self.index])
        else:
            self.master.quit()


    def set_image(self, path):
        """
        Helper function which sets a new image in the image view
        :param path: path to that image
        """
        image = self._load_image(path)
        self.image_raw = image
        self.image_panel.update()
        self.image = ImageTk.PhotoImage(image)
        self.image_panel.configure(image=self.image)

    def vote(self, label):
        """
        Processes a vote for a label: Initiates the file copying and shows the next image
        :param label: The label that the user voted for
        """

        df.label[self.index] = label
        df.to_csv(df_path)
        self.update_panel()

    def vote_key(self, event):
        """
        Processes voting via the number key bindings.
        :param event: The event contains information about which key was pressed
        """
        pressed_key = int(event.char)
        label = self.labels[pressed_key-1]
        self.vote(label)

    def toggle_vote_key(self, event):
        """
        Processes voting via the number key bindings.
        :param event: The event contains information about which key was pressed
        """
        label = df.label[self.index]
        label_index = self.labels.index(label)
        new_label = self.labels[(label_index + 1) % len(self.labels)]
        self.vote(label)

    def left_key(self, event):
        """
        Processes voting via the number key bindings.
        :param event: The event contains information about which key was pressed
        """
        self.move_prev_image()

    def right_key(self, event):
        """
        Processes voting via the number key bindings.
        :param event: The event contains information about which key was pressed
        """
        self.move_next_image()

    #### added in version 2
    def num_pic_type(self, event):
        """Function that allows for typing to what picture the user wants to go.
        Works only in mode 'copy'."""
        # -1 in line below, because we want images bo be counted from 1 on, not from 0
        self.index = self.return_.get() - 1
        self.update_panel()

    @staticmethod
    def _load_image(path, size=(400,700)):
        """
        Loads and resizes an image from a given path using the Pillow library
        :param path: Path to image
        :param size: Size of display image
        :return: Resized image
        """
        image = Image.open(path)
        width, height = image.size
        if width > img_max_width:
            new_width = img_max_width;
            new_height = int(float(new_width) * (float(height) / width))
            width = new_width
            height = new_height

        if height > img_max_height:
            new_height = img_max_height
            new_width = int(float(new_height) * (float(width) / float(height)))
            width = new_width
            height = new_height

        image = image.resize((width, height), Image.ANTIALIAS)
        return image


def get_path_recursive(input_folder, file_extensions, paths):
    for root, subdirs, files in os.walk(input_folder):
        for file in files:
            if os.path.splitext(file)[1] in file_extensions:
                paths.append(os.path.join(root, file))
#        for subdir in subdirs:
#            get_path_recursive(os.path.join(root, subdir), file_extensions, paths)


# The main bit of the script only gets exectured if it is directly called
if __name__ == "__main__":

###### Commenting out the initial input and puting input into preamble
#     # Make input arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', '--folder', help='Input folder where the *tif images should be', required=True)
#     parser.add_argument('-l', '--labels', nargs='+', help='Possible labels in the images', required=True)
#     args = parser.parse_args()

#     # grab input arguments from args structure
#     input_folder = args.folder
#     labels = args.labels

    args = parse_args()
    input_folder = args.input
    df_path = args.dataset
    img_max_width = args.max_width
    img_max_height = args.max_height
    

    # Put all image file paths into a list
    paths = []
    get_path_recursive(input_folder, file_extensions, paths)
    paths = sorted(set(paths))
    try:
        df = pd.read_csv(df_path, header=0)
        # Store configuration file values
    except FileNotFoundError:
        df = pd.DataFrame(columns=["full_path", "im_path", 'set_name', 'label'])
        for i in range(len(paths)):
            set_name = os.path.split(os.path.dirname(paths[i]))[-1]
            image_path = os.path.join(set_name, os.path.basename(paths[i]))
            df.loc[i] = [paths[i], image_path, set_name, labels[0]]


# Start the GUI
root = tk.Tk()
app = ImageGui(root, labels, colors, paths)
root.mainloop()
