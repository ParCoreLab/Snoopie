import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
if folder_path:
    print(folder_path)
    root.destroy()
else:
    print("-1")
    root.destroy()
    exit(-1)

root.mainloop()
