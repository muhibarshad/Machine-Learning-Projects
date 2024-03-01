import tkinter as tk
import os

def main() :
    window = tk.Tk() # For the external window
    header = tk.Label(text="Hy Sweet, Enter what you want to say : ") # Header of the window
    told = tk.Text(window) # for the inner text
    header.pack() # Attach the text box to the window
    told.pack()  # Attach the text box to the window
    
    def speak(): # nested function 
        said = told.get("1.0", tk.END) # Get the text from start to end
        command = " espeak "+ f"\"{said}\"" # making the command to execute 
        os.system(command) # exexute the command 
        
    button = tk.Button(window, text="Speak", command=speak) # button with the command function to execute 
    button.pack()
    
    window.mainloop() # event listener to handle the buttton clicks 
    
if __name__ == '__main__': # main line of the scripte
    main()