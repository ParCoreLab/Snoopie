import sys
if sys.version_info[0] == 3:
    import tkinter as tk
else:
    import Tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(initialdir=os.getcwd())
if folder_path:
    print(folder_path)
    root.destroy()
else:
    print("-1")
    root.destroy()
    exit(-1)

root.mainloop()
