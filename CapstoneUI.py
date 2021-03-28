from tkinter import *

master = Tk()

canvas = Canvas(master,height=720,width=1280)
canvas.pack()

frame_left= Frame(master,bg='#002401')
frame_left.place(relx=0,rely=0,relwidth=0.3,relheight=1)

frame_right = Frame(master,bg='#dedede')
frame_right.place(relx=0.3,rely=0,relwidth=0.7,relheight=1)

lbl_featureNum= Label(frame_left,bg="#002401",text="Feature",font="Verdana 12 bold",fg='white',padx=30,pady=50)
lbl_featureNum.place(relx=0,rely=0,anchor='nw')

var=IntVar()

R1= Radiobutton(frame_left,text="7",variable=var,value=1,bg="#002401",fg='white',font="Verdana 10",selectcolor="#002401")
R1.grid(row=0,column=0,padx=26,pady=75)

R2= Radiobutton(frame_left,text="9",variable=var,value=2,bg="#002401",fg='white',font="Verdana 10",selectcolor="#002401")
R2.grid(row=0,column=1,padx=26,pady=75)


R3= Radiobutton(frame_left,text="12",variable=var,value=3,bg="#002401",fg='white',font="Verdana 10",selectcolor="#002401")
R3.grid(row=0,column=2,padx=26,pady=75)

R3= Radiobutton(frame_left,text="15",variable=var,value=4,bg="#002401",fg='white',font="Verdana 10",selectcolor="#002401")
R3.grid(row=0,column=3,padx=26,pady=75)

master.mainloop()