from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import geometry

tk = Tk()


# TODO offload global variables
path_to_image_f = ""
path_to_image_s = ""
path_to_coordinates = ""

# TODO: reformat to named tuple
# Why different point_coordinates_f and ..._s
first_reference_point_coordinates_f = [-695.0, 25.0]
second_reference_point_coordinates_f = [397.0, 166.0]
target_coordinates_f = [-695.0, 25.0]
first_reference_point_coordinates_s = [-712.0, 53.0]
second_reference_point_coordinates_s = [709.0, 164.0]
target_coordinates_s = [-712.0, 55.0]

first_reference_point_coordinates = ['-0.47', '3.31', '-0.17']
second_reference_point_coordinates = ['1.16', '5.56', '-0.06']
first_camera_coordinates = ['0', '0', '0']
second_camera_coordinates = ['1.16', '0', '0']


# TODO: destroy extra functions
def choose_first_point_f():
    img = ImageBoard("image_f", "first_reference_point_coordinates_f")


def choose_first_point_s():
    img = ImageBoard("image_f", "second_reference_point_coordinates_f")


def choose_target_f():
    img = ImageBoard("image_f", "target_coordinates_f")


def choose_target_s():
    img = ImageBoard("image_s", "target_coordinates_s")


def choose_second_point_f():
    img = ImageBoard("image_s", "first_reference_point_coordinates_s")


def choose_second_point_s():
    img = ImageBoard("image_s", "second_reference_point_coordinates_s")


def choose_image_1():
    filename = askopenfilename()
    global path_to_image_f
    path_to_image_f = filename


def choose_image_2():
    filename = askopenfilename()
    global path_to_image_s
    path_to_image_s = filename


def choose_file():
    filename = askopenfilename()
    global path_to_coordinates
    path_to_coordinates = filename
    save_coordinates(filename)


def save_coordinates(path_to_coordinates):
    global first_camera_coordinates, second_camera_coordinates, first_reference_point_coordinates,\
        second_reference_point_coordinates
    with open(path_to_coordinates, "r") as f:
        text = f.read()
    if text.count(f"\n") != 3 or text.count("16"):
        assert EOFError
    objects = text.split(f"\n")
    first_camera_coordinates = objects[0].split(";")
    second_camera_coordinates = objects[1].split(";")
    first_reference_point_coordinates = objects[2].split(";")
    second_reference_point_coordinates = objects[3].split(";")


def calculate_coordinates():
    global first_reference_point_coordinates_f, second_reference_point_coordinates_f, target_coordinates_f, \
        first_reference_point_coordinates_s, second_reference_point_coordinates_s, target_coordinates_s,\
        first_reference_point_coordinates, second_reference_point_coordinates, first_camera_coordinates,\
        second_camera_coordinates
    #print(str(first_reference_point_coordinates_f)+f"\n"+str(second_reference_point_coordinates_f)+f"\n"+
    #     str(target_coordinates_f)+f"\n"+str(first_reference_point_coordinates_s)+f"\n"+
    #     str(second_reference_point_coordinates_s)+f"\n"+str(target_coordinates_s)+f"\n"+
    #     str(first_reference_point_coordinates)+f"\n"+str(second_reference_point_coordinates)+f"\n"+
    #    str(first_camera_coordinates)+f"\n"+str(second_camera_coordinates))
    # create two cameras and calculate
    first_point = geometry.KnownPoint(*first_reference_point_coordinates)
    second_point = geometry.KnownPoint(*second_reference_point_coordinates)
    camera_f = geometry.Camera(*first_camera_coordinates, *first_reference_point_coordinates_f,
                               *second_reference_point_coordinates_f, *target_coordinates_f)
    camera_s = geometry.Camera(*second_camera_coordinates, *first_reference_point_coordinates_s,
                               *second_reference_point_coordinates_s, *target_coordinates_s)

    camera_f.find_point_on_camera(first_point, second_point)
    camera_s.find_point_on_camera(first_point, second_point)

    #camera_f.get_coordinates()

    target_coordinates = geometry.find_coordinates_of_target(camera_f, camera_s)
    with open("target.txt", "w") as f:
        f.write(str(target_coordinates))



class ImageBoard:
    def __init__(self, image_to_show: str, coordinates_to_remember: str):
        self.cord_to_remember = coordinates_to_remember
        filename = ""
        if image_to_show == "image_f":
            global path_to_image_f
            filename = path_to_image_f
        elif image_to_show == "image_s":
            global path_to_image_s
            filename = path_to_image_s
        self.root = Toplevel(tk)

        img = Image.open(filename)
        photo_from_camera = ImageTk.PhotoImage(img)

        self.height_of_image = photo_from_camera.height()
        self.width_of_image = photo_from_camera.width()

        canvas_photo = Canvas(self.root, width=self.width_of_image, height=self.height_of_image)

        photo = canvas_photo.create_image(0, 0, anchor=NW, image=photo_from_camera)
        canvas_photo.pack()
        self.root.update()
        canvas_photo.bind("<Button-1>", self.on_wasd)
        self.root.mainloop()

    def on_wasd(self, event):
        if self.cord_to_remember == "first_reference_point_coordinates_f":
            global first_reference_point_coordinates_f
            first_reference_point_coordinates_f = [event.x - self.width_of_image/2, -event.y + self.height_of_image / 2]
        elif self.cord_to_remember == "second_reference_point_coordinates_f":
            global second_reference_point_coordinates_f
            second_reference_point_coordinates_f = [event.x - self.width_of_image/2,
                                                    -event.y + self.height_of_image / 2]
        elif self.cord_to_remember == "target_coordinates_f":
            global target_coordinates_f
            target_coordinates_f = [event.x - self.width_of_image/2, -event.y + self.height_of_image / 2]
        elif self.cord_to_remember == "first_reference_point_coordinates_s":
            global first_reference_point_coordinates_s
            first_reference_point_coordinates_s = [event.x - self.width_of_image/2, -event.y + self.height_of_image / 2]
        elif self.cord_to_remember == "second_reference_point_coordinates_s":
            global second_reference_point_coordinates_s
            second_reference_point_coordinates_s = [event.x - self.width_of_image/2,
                                                    -event.y + self.height_of_image / 2]
        elif self.cord_to_remember == "target_coordinates_s":
            global target_coordinates_s
            target_coordinates_s = [event.x - self.width_of_image/2, -event.y + self.height_of_image / 2]
        self.root.destroy()


if __name__ == "__main__":

    canvas = Canvas(tk, width=500, height=500)
    canvas.pack()

    choose_image_f_button = Button(tk, text="choose image 1", command=choose_image_1)
    choose_image_f_button.place(x=50, y=100)

    tk.update()

    show_image_f_button_f = Button(tk, text="choose first point on image 1", command=choose_first_point_f)
    show_image_f_button_f.place(x=50, y=150)

    show_image_f_button_s = Button(tk, text="choose second point on image 1", command=choose_first_point_s)
    show_image_f_button_s.place(x=50, y=200)

    show_image_f_button_target = Button(tk, text="choose target on image 1", command=choose_target_f)
    show_image_f_button_target.place(x=50, y=250)

    choose_image_s_button = Button(tk, text="choose image 2", command=choose_image_2)
    choose_image_s_button.place(x=300, y=100)

    show_image_s_button_f = Button(tk, text="choose first point on image 2", command=choose_second_point_f)
    show_image_s_button_f.place(x=300, y=150)

    show_image_s_button_s = Button(tk, text="choose second point on image 2", command=choose_second_point_s)
    show_image_s_button_s.place(x=300, y=200)

    show_image_s_button_target = Button(tk, text="choose target on image 2", command=choose_target_s)
    show_image_s_button_target.place(x=300, y=250)

    choose_coordinates = Button(tk, text="choose file with coordinates", command=choose_file)
    choose_coordinates.place(x=50, y=300)

    calculate_coordinates_button = Button(tk, text="calculate coordinates", command=calculate_coordinates)
    calculate_coordinates_button.place(x=300, y=300)


    tk.mainloop()