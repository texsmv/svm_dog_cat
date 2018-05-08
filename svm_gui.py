import tkinter as tk
from tkinter import *
from tkinter import ttk
import sys
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from funciones import *
from svm_model import *

root = tk.Tk()

root.title("SVM")

width = 500
height = 400
x_offset = 50
y_offset = 100

root.geometry("%dx%d+%d+%d" % (width, height,x_offset,y_offset))

def OpenFile(label):
    global name
    name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                           filetypes =(("All Files","*.*"),("Text File", "*.txt")),
                           title = "Choose a file.")
    image2 = Image.open(name)
    image2 = image2.resize((250, 250), Image.ANTIALIAS)
    displayImage = ImageTk.PhotoImage(image2)
    label.configure(image = displayImage)
    label.image = displayImage



def predict():
    image_t = cv2.imread(name)
    image_t = cv2.resize(image_t,(256,256))
    image_t = resize(image_t)
    image_t_fv = [feature_vector_4_img(process_image(image_t))]
    out_put = clf.predict(image_t_fv)
    textBoxResult.config(state=NORMAL)
    textBoxResult.delete("1.0",END)
    textBoxResult.insert(END,str(out_put[0]))
    textBoxResult.config(state=DISABLED)



frame = tk.Frame(root)
frame.pack(fill='both',expand='yes')

#Load image
#image = tk.PhotoImage(file="D:/charizard.png")
label = tk.Label(image="")
label.place(x=100,y=50)
#label.pack()

#Load Button
loadButton = tk.Button(frame,height=1, width=10,text="...",command = lambda: OpenFile(label))
loadButton.place(x=100,y=350)

#Predict Button
predictButton = tk.Button(frame,height=1, width=10,text="predict",command = predict)
predictButton.place(x=300,y=350)

#TextBoxResult
textBoxResult = tk.Text(frame,height = 1, width=8)
textBoxResult.place(x=400,y=110)




root.mainloop()
