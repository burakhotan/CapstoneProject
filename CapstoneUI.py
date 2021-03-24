from tkinter import *

master = Tk()

canvas = Canvas(master,height=720,width=1280)
canvas.pack()

frame_left= Frame(master,bg='#002401')
frame_left.place(relx=0,rely=0,relwidth=0.3,relheight=1)

frame_right = Frame(master,bg='#dedede')
frame_right.place(relx=0.3,rely=0,relwidth=0.7,relheight=1)

master.mainloop()