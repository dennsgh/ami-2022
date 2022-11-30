from tracemalloc import start
from PIL import ImageTk, Image, ImageDraw, ImageOps
import numpy as np
import tkinter as tk
import threading
from pathlib import Path

from pyrsistent import b
from json_handler import JSON_Handler


################################################################################################################################
#   You might want to fix paths in json_handler.py for your env
#   You could use https://jsoneditoronline.org/ to look at the json file structure if it helps to get an overview
#   The program simply goes through the index you provide and will show you the image and ask if you want to relabel
#   Valid inputs are only (y/else) and integers of categories in the prelabeled.json
################################################################################################################################
def crop_region(image: np.array, bounding_box) -> np.array:
    '''Crops an image based on the bounding box
    Args:
        image: The image in np.array format
        bounding_box: The bounding box of the image presumably from the json
    Returns: 
        np.array cropped image
    '''

    x_min, y_min, x_range, y_range = bounding_box

    return image[x_min:x_min + x_range, y_min:y_min + y_range]


class App(threading.Thread):
    '''App to help relabel images
    Initialize the path to the jason using the JSON_Handler object
    '''

    def __init__(self, tk_root, json_handler: JSON_Handler) -> None:
        '''Initializes the application
        Args:
            tk_root: root window of the application(tk.Tk())
        Returns:
            None
        '''
        self.thumbnail_size = (200, 200)
        self.source_size = (720, 480)
        self.root = tk_root
        self.jh = json_handler
        self.data = self.jh.data
        self.root.geometry("1280x720")  # (optional)
        self.root.attributes('-topmost', True)
        self.LABEL = tk.Label(self.root, text="Relabeling App")
        self.LABEL.pack()
        threading.Thread.__init__(self)
        self.start()

    def run(self) -> None:
        '''Runs the relabel app
        Args:
            None
        Returns:
            None
        '''
        print("=====================================")
        print("1 : Default with starting/stopping index")
        print("2 : Go through all data of a specific label")
        print("3 : Default with no (y/n)")
        print("4 : Check from a .txt file ")
        print("=====================================")
        mode = int(input("Relabel mode: >> "))
        if (mode == 1):
            start_index = int(input("Starting index >> "))
            stop_index = int(input("Stopping index >> "))
            self.root.deiconify()
            if (self.relabel_service(start_index, stop_index, mode)):
                print("Program Terminated.")
                self.root.quit()
                self.root.update()

        if (mode == 2):
            while (1):
                try:
                    print(self.jh.data["categories"])
                    arg1 = int(input("Which label to verify? >> "))
                    if (self.jh.is_label(arg1)):
                        break
                except:
                    print("Not a label, please enter a valid label!")
            self.root.deiconify()
            arg2 = int(input("Starting from index >> "))

            if (self.relabel_service(arg1, arg2, mode)):
                print("Program Terminated.")
                self.root.quit()
                self.root.update()

        if (mode == 3):
            start_index = int(input("Starting index >> "))
            stop_index = int(input("Stopping index >> "))
            self.root.deiconify()
            if (self.relabel_service(start_index, stop_index, mode)):
                print("Program Terminated.")
                self.root.quit()
                self.root.update()

        if (mode == 4):
            cwd = Path().cwd()
            print(f"Current working directory : {cwd.as_posix()}")
            fp = Path(cwd, str(input("Path to text file >> ")))
            self.root.deiconify()
            if (self.relabel_service(fp, -1, mode)):
                print("Program Terminated.")
                self.root.quit()
                self.root.update()

    # Relabels from starting index to end index
    def relabel_service(self, arg1: int, arg2: int, mode: int) -> bool:
        '''Relabel service from starting index to end index, the run function will call this function
        Args:
            arg1: passed from run as a mode argument
            arg2: passed from run as a mode argument
            mode: the mode of relabel
        Returns:
            True on success
        '''
        if mode == 1:
            for i in range(arg1, arg2 + 1):
                #Access data per order
                damage_type = self.jh.get_label_from_id(
                    self.jh.data['annotations'][i]['category_id'])
                filename = self.jh.data["annotations"][i]['filepath']
                print("================================")
                print(f"Index : {i}")
                print(f"Damage Label : {damage_type}")
                print(f"Filename : {filename}")
                _ = Path("data", self.jh.data["annotations"][i]['filepath'])
                im = Image.open(_)
                im = ImageOps.contain(im, self.thumbnail_size)
                img = ImageTk.PhotoImage(im)
                self.LABEL = tk.Label(self.root, image=img)
                self.LABEL.pack()
                print("Is the image correctly labeled? (y/n) ")

                bool_replace = str(input(">> "))
                if (bool_replace.lower() == 'n'):
                    _ = Path("data", self.jh.data['annotations'][i]['source'])
                    print(_.is_file())
                    ori_img = Image.open(_)
                    bbox = self.jh.data['annotations'][i]['bbox']

                    x_min, y_min, x_range, y_range = bbox
                    shape = [(x_min, y_min), (x_min + x_range, y_min + y_range)]
                    boxer = ImageDraw.Draw(ori_img)
                    boxer.rectangle(shape, outline="red", width=5)
                    ori_img = ImageOps.contain(ori_img, self.source_size)
                    ori_img_tk = ImageTk.PhotoImage(ori_img)
                    ori_tk = tk.Label(self.root, image=ori_img_tk)
                    ori_tk.pack()

                    if (self.jh.do_relabel_routine(i)):
                        print(
                            f"Index {i} was relabelled to {self.jh.data['annotations'][i]['category_id']}:{self.jh.get_label_from_id(self.jh.data['annotations'][i]['category_id'])} successfully.."
                        )
                    ori_tk.destroy()
                print("================================")
                self.LABEL.destroy()
                self.root.update()

            return True

        if mode == 2:
            i = 0
            for entry in self.jh.data['annotations']:

                if entry["category_id"] == arg1 and i >= arg2:
                    #Access data per order
                    damage_type = self.jh.get_label_from_id(entry["category_id"])
                    filename = entry['filepath']
                    print("================================")
                    print(f"Index : {i}")
                    print(f"Damage Label : {damage_type}")
                    print(f"Filename : {filename}")
                    _ = Path("data", entry['filepath'])
                    im = Image.open(_)
                    im = ImageOps.contain(im, self.thumbnail_size)

                    img = ImageTk.PhotoImage(im)
                    self.LABEL = tk.Label(self.root, image=img)
                    self.LABEL.pack()

                    _ = Path("data", entry['source'])
                    print(_.is_file())
                    ori_img = Image.open(_)
                    bbox = entry['bbox']

                    x_min, y_min, x_range, y_range = bbox
                    shape = [(x_min, y_min), (x_min + x_range, y_min + y_range)]
                    boxer = ImageDraw.Draw(ori_img)
                    boxer.rectangle(shape, outline="red", width=5)
                    ori_img = ImageOps.contain(ori_img, self.source_size)
                    ori_img_tk = ImageTk.PhotoImage(ori_img)
                    ori_tk = tk.Label(self.root, image=ori_img_tk)
                    ori_tk.pack()

                    if (self.jh.do_relabel_routine(i)):
                        print(
                            f"Image {entry['id']} was relabelled to {entry['category_id']}:{self.jh.get_label_from_id(entry['category_id'])} successfully.."
                        )
                    ori_tk.destroy()
                    print("================================")
                    self.LABEL.destroy()
                    self.root.update()

                i += 1

            return True

        if mode == 3:
            for i in range(arg1, arg2 + 1):
                #Access data per order
                damage_type = self.jh.get_label_from_id(
                    self.jh.data['annotations'][i]['category_id'])
                filename = self.jh.data["annotations"][i]['filepath']
                print("================================")
                print(f"Index : {i}")
                print(f"Damage Label : {damage_type}")
                print(f"Filename : {filename}")
                _ = Path("data", self.jh.data["annotations"][i]['filepath'])
                im = Image.open(_)
                im = ImageOps.contain(im, self.thumbnail_size)
                img = ImageTk.PhotoImage(im)
                self.LABEL = tk.Label(self.root, image=img)
                self.LABEL.pack()

                _ = Path("data", self.jh.data['annotations'][i]['source'])
                print(_.is_file())
                ori_img = Image.open(_)
                bbox = self.jh.data['annotations'][i]['bbox']

                x_min, y_min, x_range, y_range = bbox
                shape = [(x_min, y_min), (x_min + x_range, y_min + y_range)]
                boxer = ImageDraw.Draw(ori_img)
                boxer.rectangle(shape, outline="red", width=5)
                ori_img = ImageOps.contain(ori_img, self.source_size)
                ori_img_tk = ImageTk.PhotoImage(ori_img)
                ori_tk = tk.Label(self.root, image=ori_img_tk)
                ori_tk.pack()

                if (self.jh.do_relabel_routine(i)):
                    print(
                        f"Index {i} was relabelled to {self.jh.data['annotations'][i]['category_id']}:{self.jh.get_label_from_id(self.jh.data['annotations'][i]['category_id'])} successfully.."
                    )
                ori_tk.destroy()
                print("================================")
                self.LABEL.destroy()
                self.root.update()

            return True

        if mode == 3:
            for i in range(arg1, arg2 + 1):
                #Access data per order
                damage_type = self.jh.get_label_from_id(
                    self.jh.data['annotations'][i]['category_id'])
                filename = self.jh.data["annotations"][i]['filepath']
                print("================================")
                print(f"Index : {i}")
                print(f"Damage Label : {damage_type}")
                print(f"Filename : {filename}")
                _ = Path("data", self.jh.data["annotations"][i]['filepath'])
                im = Image.open(_)
                im = ImageOps.contain(im, self.thumbnail_size)
                img = ImageTk.PhotoImage(im)
                self.LABEL = tk.Label(self.root, image=img)
                self.LABEL.pack()

                _ = Path("data", self.jh.data['annotations'][i]['source'])
                print(_.is_file())
                ori_img = Image.open(_)
                bbox = self.jh.data['annotations'][i]['bbox']

                x_min, y_min, x_range, y_range = bbox
                shape = [(x_min, y_min), (x_min + x_range, y_min + y_range)]
                boxer = ImageDraw.Draw(ori_img)
                boxer.rectangle(shape, outline="red", width=5)
                ori_img = ImageOps.contain(ori_img, self.source_size)
                ori_img_tk = ImageTk.PhotoImage(ori_img)
                ori_tk = tk.Label(self.root, image=ori_img_tk)
                ori_tk.pack()

                if (self.jh.do_relabel_routine(i)):
                    print(
                        f"Index {i} was relabelled to {self.jh.data['annotations'][i]['category_id']}:{self.jh.get_label_from_id(self.jh.data['annotations'][i]['category_id'])} successfully.."
                    )
                ori_tk.destroy()
                print("================================")
                self.LABEL.destroy()
                self.root.update()

            return True

        if mode == 4:
            problem_set = []
            with open(arg1, 'r') as fp:
                for line in fp:
                    for token in line.split():
                        problem_set.append(int(token))

            print(problem_set)
            i = 0
            for entry in self.jh.data['annotations']:
                if len(problem_set) <= 0:
                    break

                if entry['id'] in problem_set:
                    problem_set.remove(entry['id'])
                    #Access data per order
                    damage_type = self.jh.get_label_from_id(
                        self.jh.data['annotations'][i]['category_id'])
                    filename = self.jh.data["annotations"][i]['filepath']
                    print("================================")
                    print(f"Index : {i}")
                    print(f"Damage Label : {damage_type}")
                    print(f"Filename : {filename}")
                    _ = Path("data", self.jh.data["annotations"][i]['filepath'])
                    im = Image.open(_)
                    im = ImageOps.contain(im, self.thumbnail_size)
                    img = ImageTk.PhotoImage(im)
                    self.LABEL = tk.Label(self.root, image=img)
                    self.LABEL.pack()

                    _ = Path("data", self.jh.data['annotations'][i]['source'])
                    print(_.is_file())
                    ori_img = Image.open(_)
                    bbox = self.jh.data['annotations'][i]['bbox']

                    x_min, y_min, x_range, y_range = bbox
                    shape = [(x_min, y_min), (x_min + x_range, y_min + y_range)]
                    boxer = ImageDraw.Draw(ori_img)
                    boxer.rectangle(shape, outline="red", width=5)
                    ori_img = ImageOps.contain(ori_img, self.source_size)
                    ori_img_tk = ImageTk.PhotoImage(ori_img)
                    ori_tk = tk.Label(self.root, image=ori_img_tk)
                    ori_tk.pack()

                    if (self.jh.do_relabel_routine(i)):
                        print(
                            f"Index {i} was relabelled to {self.jh.data['annotations'][i]['category_id']}:{self.jh.get_label_from_id(self.jh.data['annotations'][i]['category_id'])} successfully.."
                        )
                    ori_tk.destroy()
                    print("================================")
                    self.LABEL.destroy()
                    self.root.update()

                i += 1
            return True


def main():
    #Spawns GUI and json backend
    jh = JSON_Handler()
    ROOT = tk.Tk()
    ROOT.withdraw()

    APP = App(ROOT, jh)
    ROOT.mainloop()


if __name__ == "__main__":
    main()