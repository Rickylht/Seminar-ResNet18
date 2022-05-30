from tkinter import *
from train_procedure_0 import get_results 
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import os
import cv2 as io
import random
import matplotlib.pyplot as plt
import math
from PIL import ImageGrab

class MY_GUI(): 
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name
        self.img = None
        self.img_old01 = None
        self.img_old02 = None
        self.img_custom = None
        
        self.drawing_line_count = 0

        self.new_x0 = 0.0
        self.new_y0 = 0.0
        self.new_x1 = 0.0
        self.new_y1 = 0.0
        self.new_x2 = 0.0
        self.new_y2 = 0.0
        self.new_x3 = 0.0
        self.new_y3 = 0.0 

    def set_init_window(self):
        self.init_window_name.title("Victory")
        self.init_window_name.geometry('800x700')

        self.name_label = Label(self.init_window_name, 
                                text = "Victory", 
                                font = ('Arial', 20))
        self.name_label.grid(row = 0, column = 0)

        self.ImportImage_button = Button(self.init_window_name, 
                                         text = "Import Image", 
                                         bg = 'lightblue',
                                         width = 20,
                                         command = self.import_and_compute)
        self.ImportImage_button.grid(row = 1, column = 0)

        self.status_label = Label(self.init_window_name, 
                                 text = "Status: ", 
                                 font = ('Arial', 20))
        self.status_label.grid(row = 2, column = 10)

        self.status_text = Text(self.init_window_name, width = 10, height = 1, font = ('Arial', 20))
        self.status_text.grid(row = 3, column = 10)

        self.BohlerAngle_label = Label(self.init_window_name, 
                                       text = "Böhler Angle: ", 
                                       font = ('Arial', 20))
        self.BohlerAngle_label.grid(row = 5, column = 10)

        self.BohlerAngle_text = Text(self.init_window_name, width = 10, height = 1, font = ('Arial', 20))                            
        self.BohlerAngle_text.grid(row = 6, column = 10)

        self.BohlerAngle_mark = Label(self.init_window_name, text = "", font = ('Arial', 12), fg = 'green')
        self.BohlerAngle_mark.grid(row = 6, column = 11)

        self.MoreExample_button = Button(self.init_window_name, 
                                         text = "Examples with \nfull recovery", 
                                         bg = 'lightgreen',
                                         font = ('Arial', 15),
                                         command = self.show_more_example)
        self.MoreExample_button.grid(row = 8, column = 10)

        self.Hide_Contour_button = Button(self.init_window_name, 
                                          text = "Hide Lines",
                                          font = ('Arial', 15),
                                          command =  self.decide)
        self.Hide_Contour_button.grid(row = 12, column = 0)

        self.CustomDrawing_button = Button(self.init_window_name, 
                                         text = "Custom Drawing", 
                                         bg = 'lightblue',
                                         font = ('Arial', 15),
                                         command = self.drawing)
        self.CustomDrawing_button.grid(row = 12, column = 1) 

        self.Compute_New_Angle_button = Button(self.init_window_name, 
                                         text = "Compute", 
                                         bg = 'blue',
                                         font = ('Arial', 15),
                                         command = self.compute_new_angle)
                                        
        self.Reset_button = Button(self.init_window_name, 
                                   text = "Reset ⟳", 
                                   bg = 'red',
                                   font = ('Arial', 15),
                                   command = self.reset)                                

        self.line_info_label1 = Label(self.init_window_name, text = "———  Wrist Joint line", font = ('Arial', 12), fg = 'blue')
        self.line_info_label1.grid(row = 10, column = 10, sticky = W)

        self.line_info_label2 = Label(self.init_window_name, text = "———  Radius Center line", font = ('Arial', 12), fg = 'red')
        self.line_info_label2.grid(row = 11, column = 10, sticky = W)
        #photo canvas
        self.photo_canvas = Canvas(self.init_window_name, 
                                   width = 450,
                                   height = 550,
                                   bg = "lightgrey")
        self.photo_canvas.create_line(5, 5, 5, 550, fill='red')
        self.photo_canvas.create_line(5, 550, 449, 550, fill='red')
        self.photo_canvas.create_line(449, 550, 449, 5, fill='red')
        self.photo_canvas.create_line(449, 5, 5, 5, fill='red')                     
        self.photo_canvas.grid(row = 2, column = 0, rowspan = 10, columnspan = 10)

        

    def import_and_compute(self):
        
        photo_path = askopenfilename()
        #print(photo_path)

        #functions of computing bohler angle
        angle, status = get_results(photo_path)

        #show image
        self.img = ImageTk.PhotoImage(Image.open('./images/image_contour.png'))
        self.photo_canvas.create_image(20, 20, image = self.img, anchor = NW)

        Status = status                         #need to be changed
        self.status_text.delete(1.0, END)
        self.status_text.insert(END, Status)

        Angle = angle                           #need to be changed
        self.BohlerAngle_text.delete(1.0, END)
        self.BohlerAngle_text.insert(END, Angle)

        if Angle <= 15 and Angle >= 10:
            self.BohlerAngle_mark['text'] = 'OK'
            self.BohlerAngle_mark['fg'] = 'green'
        else:
            self.BohlerAngle_mark['text'] = 'Out of range'
            self.BohlerAngle_mark['fg'] = 'red'
        

        #create directory to store images
        folder = os.path.exists("images")
        if not folder:
            os.makedirs("images")
        
        image_name = os.path.join(photo_path)
        image_old = io.imread(image_name)
        image_old = io.resize(image_old, (400,500), interpolation= io.INTER_AREA)
        io.imwrite('./images/image_old.png', image_old)
        
    
    def show_more_example(self):
        window_example = Toplevel(self.init_window_name)
        window_example.title("More Examples")
        window_example.geometry("900x600")

        canvas = Canvas(window_example, width = 1000, height = 600, bg = "lightgrey")
        
        #insert images
        
        idx = random.randint(1,10)
        self.img_old01 = ImageTk.PhotoImage(Image.open("./image_negative/{}.png".format(idx)).resize((400,500)))
        canvas.create_image(20, 20, image = self.img_old01, anchor = NW)

        idx = random.randint(11,20)
        self.img_old02 = ImageTk.PhotoImage(Image.open("./image_negative/{}.png".format(idx)).resize((400,500)))
        canvas.create_image(20 + 400, 20, image = self.img_old02, anchor = NW)
        
        canvas.pack()

    
    def decide(self):
        if self.Hide_Contour_button['text'] == "Hide Lines":
            self.img = ImageTk.PhotoImage(Image.open('./images/image_old.png'))
            self.photo_canvas.create_image(20, 20, image = self.img, anchor = NW)
            self.Hide_Contour_button['text'] = "Show Lines"
        elif self.Hide_Contour_button['text'] == "Show Lines":
            self.img = ImageTk.PhotoImage(Image.open('./images/image_contour.png'))
            self.photo_canvas.create_image(20, 20, image = self.img, anchor = NW)
            self.Hide_Contour_button['text'] = "Hide Lines"    

    
    def drawing(self):
        if self.photo_canvas['bg'] == 'lightgrey':
            self.drawing_line_count = 0
            self.img = ImageTk.PhotoImage(Image.open('./images/image_old.png'))
            self.photo_canvas.create_image(20, 20, image = self.img, anchor = NW)

            window_example = Toplevel(self.init_window_name)
            window_example.title("Instruction")
            window_example.geometry("250x50+{}+{}".format(self.init_window_name.winfo_rootx() + 200,self.init_window_name.winfo_rooty() + 300))
            label = Label(window_example, text = "Step 1, draw Wrist Joint Line\n Step 2, draw Radius Center Line", font = ('Arial', 12), fg = "black")
            label.pack()

            self.photo_canvas.bind('<ButtonPress-1>', self.custom_drawing)
            self.photo_canvas.bind('<ButtonRelease-1>', self.custom_drawing)
            
            self.photo_canvas['bg'] = 'lightgreen'

            self.Compute_New_Angle_button.grid(row = 12, column = 3)
            self.Reset_button.grid(row = 12, column = 2)
            self.Hide_Contour_button.grid_forget()
            self.CustomDrawing_button.grid(row = 12, column = 0)

            self.CustomDrawing_button['text'] = 'Quit & Save'
        else:
            if self.drawing_line_count == 2:
                self.photo_canvas.update()
                x=self.init_window_name.winfo_rootx()+self.photo_canvas.winfo_x()
                y=self.init_window_name.winfo_rooty()+self.photo_canvas.winfo_y()
                x1=x+self.photo_canvas.winfo_width()
                y1=y+self.photo_canvas.winfo_height()
                ImageGrab.grab().crop((x+20,y+20,x1-34,y1-34)).save("./images/image_contour.png")
                window_example = Toplevel(self.init_window_name)
                window_example.title("Save successfully")
                window_example.geometry("150x50+{}+{}".format(self.init_window_name.winfo_rootx() + 200,self.init_window_name.winfo_rooty() + 300))
                label = Label(window_example, text = "Save successfully", font = ('Arial', 10, "bold"), fg = "green")
                label.pack()

            else:
                window_example = Toplevel(self.init_window_name)
                window_example.title("Not saved")
                window_example.geometry("150x50+{}+{}".format(self.init_window_name.winfo_rootx() + 200,self.init_window_name.winfo_rooty() + 300))
                label = Label(window_example, text = "Lines not saved", font = ('Arial', 10, "bold"), fg = "red")
                label.pack()


            self.img = ImageTk.PhotoImage(Image.open('./images/image_contour.png'))
            self.photo_canvas.create_image(20, 20, image = self.img, anchor = NW)

            self.photo_canvas.unbind('<ButtonPress-1>')
            self.photo_canvas.unbind('<ButtonRelease-1>')

            self.photo_canvas['bg'] = 'lightgrey'

            self.Compute_New_Angle_button.grid_forget()
            self.Reset_button.grid_forget()
            self.Hide_Contour_button.grid(row = 12, column = 0)
            self.CustomDrawing_button.grid(row = 12, column = 1)

            self.CustomDrawing_button['text'] = 'Custom Drawing'
            self.new_x0 = 0.0
            self.new_y0 = 0.0
            self.new_x1 = 0.0
            self.new_y1 = 0.0
            self.new_x2 = 0.0
            self.new_y2 = 0.0
            self.new_x3 = 0.0
            self.new_y3 = 0.0 
        
        
    def compute_new_angle(self):
        AB = [self.new_x0, self.new_y0, self.new_x1, self.new_y1]
        CD = [self.new_x2, self.new_y2, self.new_x3, self.new_y3]

        def compute_angle(v1, v2):
            dx1 = v1[2] - v1[0]
            dy1 = v1[3] - v1[1]
            dx2 = v2[2] - v2[0]
            dy2 = v2[3] - v2[1]
            angle1 = math.atan2(dy1, dx1)
            angle1 = int(angle1 * 180/math.pi)
            # print(angle1)
            angle2 = math.atan2(dy2, dx2)
            angle2 = int(angle2 * 180/math.pi)
            # print(angle2)
            if angle1*angle2 >= 0:
                included_angle = abs(angle1-angle2)
            else:
                included_angle = abs(angle1) + abs(angle2)
                if included_angle > 180:
                    included_angle = 360 - included_angle
            return included_angle

        angle = compute_angle(AB, CD)
        angle = abs(90 - angle)
        self.BohlerAngle_text.delete(1.0, END)
        self.BohlerAngle_text.insert(END, angle)

        if angle <= 15 and angle >= 10:
            self.BohlerAngle_mark['text'] = 'OK'
            self.BohlerAngle_mark['fg'] = 'green'
        else:
            self.BohlerAngle_mark['text'] = 'Out of range'
            self.BohlerAngle_mark['fg'] = 'red'


    def custom_drawing(self,event):
        
        if str(event.type) == 'ButtonPress' and event.x <= 450 and event.x >= 50 and event.y <= 550 and event.y >= 50:  #ensure point recorded in canvas
            self.photo_canvas.old_coords = event.x, event.y
            # print(event.x)
            # print(event.y)

        elif str(event.type) == 'ButtonRelease'and event.x <= 450 and event.x >= 50 and event.y <= 550 and event.y >= 50:
            x, y = event.x, event.y
            x1, y1 = self.photo_canvas.old_coords
            if self.drawing_line_count == 0:
                self.photo_canvas.create_line(x, y, x1, y1, fill='blue')
                self.new_x0 = x1        #for wrist joint line
                self.new_y0 = y1
                self.new_x1 = event.x
                self.new_y1 = event.y   
                self.drawing_line_count = self.drawing_line_count + 1
            elif self.drawing_line_count == 1:
                self.photo_canvas.create_line(x, y, x1, y1, fill='red')
                self.new_x2 = x1        #for center line
                self.new_y2 = y1
                self.new_x3 = event.x
                self.new_y3 = event.y
                self.drawing_line_count = self.drawing_line_count + 1
    
    def reset(self):
        self.img = ImageTk.PhotoImage(Image.open('./images/image_old.png'))
        self.photo_canvas.create_image(20, 20, image = self.img, anchor = NW)
        self.drawing_line_count = 0
        self.new_x0 = 0.0
        self.new_y0 = 0.0
        self.new_x1 = 0.0
        self.new_y1 = 0.0
        self.new_x2 = 0.0
        self.new_y2 = 0.0
        self.new_x3 = 0.0
        self.new_y3 = 0.0 
        

def gui_start():
    init_window = Tk()
    PORTAL = MY_GUI(init_window)
    PORTAL.set_init_window()
    init_window.mainloop()

