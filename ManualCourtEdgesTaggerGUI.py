import pickle
import re

import cv2
import numpy as np
from matplotlib import pyplot as plt

import argparse
import os, time, random
import tkinter as tk
from tkinter import messagebox

from PIL import ImageTk, Image

GAME_IMG_RATIO = 0.45
COURT_IMG_RATIO = 0.4
COLORS = [(0, 0, 255), (28, 255, 111), (255, 128, 100), (192, 128, 255), (255, 55, 55), (200, 255, 90), (208, 50, 64)]
GAUSSIAN_BLUR_FILTER_SIZE = 41
GAUSSIAN_BLUR_FILTER_SIGMA = 35
WELCOME_MESSAGE = "Welcome to NBA court tagger\n\n" \
                  "You should match points of landmarks on the court,\n" \
                  "for example: the corner of the court, corner of the free throw line etc.\n" \
                  "Between the nba court black and white image (top) and the\n" \
                  "NBA game image (bottom) by clicking on the pictures.\n" \
                  "Click on the pictures to try!\n\n\n" \
                  "Some Guidelines:\n" \
                  "- You should create at least 4 pairs, aim for 5-6\n" \
                  "- Try to spread the points far away from each other as possible\n" \
                  "- You can UNDO your last click by right clicking your mouse while it's OVER the image\n" \
                  "- Click on Generate to check if the generated picture fits the angle of the NBA game image, \n" \
                  "if not - pair more points to get a better picture.\n" \
                  "- If you can't detect at least 4 points between the pictures, you can delete the picture and move to the next one"


def get_rand_gaussian_filter_size():
    random.seed(time.time())
    num = random.randint(31, 51)
    num += 1 - num % 2
    return num, num


def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim == 3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended


def center(win, win_size):
    width = win_size[0]
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win_size[1]
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2 - 50
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()


class GUI(tk.Frame):
    def __init__(self, court_2d_filename, game_filename, master=None, show_welcome_message = False):

        tk.Frame.__init__(self, master)
        self.master = master
        master.title("Manual NBA court tagger")
        self.W, self.H = master.winfo_screenwidth() - 550, master.winfo_screenheight() - 150
        self.game_filename = game_filename
        self.path = os.path.dirname(game_filename) + os.path.sep
        self.coords_list, self.nba_court_pts,self.game_pts  = [], [], []
        self.nba_court_cv_image = cv2.imread(court_2d_filename)
        self.game_cv_image = cv2.imread(game_filename)
        self.resize_images()

        self.game_label = tk.Label(borderwidth=2, relief="groove", text="Frame from an NBA game")
        self.nba_court_label = tk.Label(borderwidth=2, relief="groove", text="NBA court 2d illustration")
        self.load_images_widgets()

        self.update_images_widgets()
        self.buttons_frame = tk.LabelFrame(self.master)
        self.buttons_frame.pack()
        self.generate_button = tk.Button(self.buttons_frame, text='Generate', width=8, pady=2,
                                         command=self.generate_transformed_image)
        self.generate_button.pack(padx=20, pady=7, side='right')
        self.delete_button = tk.Button(self.buttons_frame, text="Delete Image", command=self.delete_image)
        self.delete_button.pack(padx=20, side='left')

        master.protocol('WM_DELETE_WINDOW', self.close_gui)
        master.bind("<Escape>", lambda event: self.close_gui())

        center(master, (self.W, self.H))
        master.resizable(width=False, height=False)
        # welcome message
        if show_welcome_message:
            tk.messagebox.showinfo("Welcome to NBA court tagger", WELCOME_MESSAGE)

        self.newWindow = None

    def mouse_hover(self, event, pts_list):
        pts_list.append((event.x, event.y))
        self.update_images_widgets()
        pts_list.pop()

    def update_images_widgets(self):
        court_img = self.nba_court_cv_image.copy()
        game_img = self.game_cv_image.copy()
        for i, (x, y) in enumerate(self.nba_court_pts):
            cv2.circle(court_img, (x, y), 1, COLORS[min(i, len(COLORS) - 1)], 2)
            cv2.putText(court_img, str(i + 1), (x + 2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLORS[min(i, len(COLORS) - 1)], 2)

        for i, (x, y) in enumerate(self.game_pts):
            cv2.circle(game_img, (x, y), 1, COLORS[min(i, len(COLORS) - 1)], 2)
            cv2.putText(game_img, str(i + 1), (x + 2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLORS[min(i, len(COLORS) - 1)], 2)
        nba_court_tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)))
        self.nba_court_label.configure(image=nba_court_tk_image)
        self.nba_court_label.image = nba_court_tk_image

        game_tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(game_img, cv2.COLOR_BGR2RGB)))
        self.game_label.configure(image=game_tk_image)
        self.game_label.image = game_tk_image

    def court_label_click(self, event):
        self.nba_court_pts.append((event.x, event.y))
        self.update_images_widgets()

    def game_label_click(self, event):
        self.game_pts.append((event.x, event.y))
        self.update_images_widgets()

    def undo_click(self, event, pts_list):
        if len(pts_list) > 0:
            pts_list.pop()
            self.update_images_widgets()

    def generate_transformed_image(self):
        if len(self.nba_court_pts) != len(self.game_pts) or len(self.nba_court_pts) < 4:
            tk.messagebox.showerror("Error",
                                    "You need to tag at least 4 pairs of points to generate a transformed image.\n"
                                    "In addition the number of points needs to be the same in both images")
        else:
            h, mask = cv2.findHomography(np.float32(self.nba_court_pts), np.float32(self.game_pts), method=cv2.RANSAC)
            nba_court_gray = cv2.imread(self.path + 'nba_court_gray.jpg')
            nba_court_gray = cv2.resize(nba_court_gray, (0, 0), fx=(self.H * COURT_IMG_RATIO) / nba_court_gray.shape[0],
                                                 fy=(self.H * COURT_IMG_RATIO) / nba_court_gray.shape[0])
            court_pattern_transformed = cv2.warpPerspective(nba_court_gray, h,
                                                            (self.game_cv_image.shape[1], self.game_cv_image.shape[0]))
            courts_horizontal = np.hstack((court_pattern_transformed, self.game_cv_image))
            court_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(courts_horizontal, cv2.COLOR_BGR2RGB)))
            self.tagged_image_win(court_tk)

    def delete_image(self):
        res = tk.messagebox.askquestion('Delete current image', 'Are you sure that the image is untaggable\n'
                                                                'and you want to remove it from the disk?')
        if res == 'yes':
            print("image deleted: " + self.game_filename)
            self.master.setvar(name='result', value="delete")
            self.master.destroy()
            if os.path.exists(self.game_filename):
                os.remove(self.game_filename)

    def close_gui(self):
        res = tk.messagebox.askquestion('Exit Application', 'Do you really want to exit')
        if res == 'yes':
            self.master.setvar(name='result', value="escape")
            self.master.destroy()

    def load_images_widgets(self):
        self.nba_court_label.bind("<Button 1>", self.court_label_click)
        self.nba_court_label.bind("<Button 2>", lambda event: self.undo_click(event, self.nba_court_pts))
        self.nba_court_label.bind("<Button 3>", lambda event: self.undo_click(event, self.nba_court_pts))
        self.nba_court_label.bind("<Motion>", lambda event: self.mouse_hover(event, self.nba_court_pts))
        self.nba_court_label.bind("<Leave>", lambda event: self.update_images_widgets())

        self.nba_court_label.pack(pady=10)

        self.game_label.bind("<Button 1>", self.game_label_click)
        self.game_label.bind("<Button 2>", lambda event: self.undo_click(event, self.game_pts))
        self.game_label.bind("<Button 3>", lambda event: self.undo_click(event, self.game_pts))
        self.game_label.bind("<Motion>", lambda event: self.mouse_hover(event, self.game_pts))
        self.game_label.bind("<Leave>", lambda event: self.update_images_widgets())
        self.game_label.pack(pady=10)

    def resize_images(self):
        court_size = self.nba_court_cv_image.shape
        court_downscale_ratio = (self.H * COURT_IMG_RATIO) / court_size[0]
        self.nba_court_cv_image = cv2.resize(self.nba_court_cv_image, (0, 0), fx=court_downscale_ratio,
                                             fy=court_downscale_ratio)
        game_size = self.game_cv_image.shape
        game_downscale_ratio = (self.H * GAME_IMG_RATIO) / game_size[0]
        self.game_cv_image = cv2.resize(self.game_cv_image, (0, 0), fx=game_downscale_ratio, fy=game_downscale_ratio)

    def exit_popup_window(self, save_image):
        if save_image:
            self.newWindow.destroy()
            self.create_and_save_dataset_images()
            self.master.setvar(name='result', value="tagged")
            self.master.destroy()
        else:
            self.newWindow.destroy()
            self.master.deiconify()

    def create_and_save_dataset_images(self):
        h, mask = cv2.findHomography(np.float32(self.nba_court_pts), np.float32(self.game_pts), method=cv2.RANSAC)
        nba_court_gray = cv2.imread(self.path + 'nba_court_gray_bitmap.bmp')
        nba_court_gray = cv2.resize(nba_court_gray, (0, 0), fx=(self.H * COURT_IMG_RATIO) / nba_court_gray.shape[0],
                                    fy=(self.H * COURT_IMG_RATIO) / nba_court_gray.shape[0])
        court_pattern_transformed = cv2.warpPerspective(nba_court_gray, h,
                                                        (self.game_cv_image.shape[1], self.game_cv_image.shape[0]))

        phase1_mask_mold = cv2.imread(self.path + 'nba_court_phase1_bitmap.bmp')
        phase1_mask_mold = cv2.resize(phase1_mask_mold, (0, 0), fx=(self.H * COURT_IMG_RATIO) / phase1_mask_mold.shape[0],
                                      fy=(self.H * COURT_IMG_RATIO) / phase1_mask_mold.shape[0])
        phase1_mask_transformed = cv2.warpPerspective(phase1_mask_mold, h,
                                                      (self.game_cv_image.shape[1], self.game_cv_image.shape[0]))
        filter_size = (get_rand_gaussian_filter_size())
        phase1_blur = cv2.GaussianBlur(phase1_mask_transformed, filter_size, 0)
        phase1_horizontal = np.hstack((self.game_cv_image, phase1_blur))

        phase2 = cv2.bitwise_and(phase1_blur, self.game_cv_image)
        phase2_blur = cv2.GaussianBlur(phase2, filter_size, 0)
        phase2_alpha_blended = alphaBlend(self.game_cv_image, phase2_blur, phase1_blur.__invert__())
        phase2_horizontal = np.hstack((phase2_alpha_blended, court_pattern_transformed))
        test_phase = np.hstack((self.game_cv_image, court_pattern_transformed))

        game_id = os.path.splitext(os.path.basename(self.game_filename))[0]
        game_id_dir = self.path + os.path.sep + 'tagged_data' + os.path.sep + game_id

        if not os.path.exists(game_id_dir):
            os.makedirs(game_id_dir)
        cv2.imwrite(game_id_dir + os.path.sep + "test.jpg", test_phase)
        cv2.imwrite(game_id_dir + os.path.sep + "court_binary_mask.jpg", phase1_blur)
        cv2.imwrite(game_id_dir + os.path.sep + "phase1_horizontal.jpg", phase1_horizontal)
        cv2.imwrite(game_id_dir + os.path.sep + "game_alpha_blended.jpg", phase2_alpha_blended)
        cv2.imwrite(game_id_dir + os.path.sep + "phase2_horizontal.jpg", phase2_horizontal)

        # save coords list and h matrix to a pickle file
        game_dict = {"game_pts": self.game_pts,
                     "court_pts": self.nba_court_pts,
                     "game_res": (self.game_cv_image.shape[0], self.game_cv_image.shape[1]),
                     "court_res": (nba_court_gray.shape[0], nba_court_gray.shape[1]),
                     "h_mat": h}
        with open(game_id_dir + os.path.sep + 'dict.pickle', 'wb') as handle:
            pickle.dump(game_dict, handle)

    def tagged_image_win(self, court_tk):
        self.newWindow = tk.Toplevel(self.master)

        game_image_label = tk.Label(self.newWindow, borderwidth=2, relief="groove")
        text = "Do these patterns match the ones in the original picture? if so, click save\n" \
               "if not, click cancel and try to tag more points before clicking on generate again"
        tag_question_label = tk.Label(self.newWindow, text=text)
        tag_question_label.config(font=("Arial", 14))
        tag_question_label.pack(pady=10)

        save_button = tk.Button(self.newWindow, text="Save image",
                                command=lambda: self.exit_popup_window(save_image=True))
        cancel_button = tk.Button(self.newWindow, text="Cancel",
                                  command=lambda: self.exit_popup_window(save_image=False))

        game_image_label.pack()
        save_button.pack(side='right', padx=5, pady=5)
        cancel_button.pack(side='right', padx=5)
        game_image_label.configure(image=court_tk)
        game_image_label.image = court_tk
        self.newWindow.protocol('WM_DELETE_WINDOW', lambda: self.exit_popup_window(save_image=False))

        self.master.withdraw()


def parse_args():
    parser = argparse.ArgumentParser(description="Manually tag images of potential players and their jersey number")
    parser.add_argument("--games_images_path", type=str, default=os.getcwd(),
                        help="Path to the folder that contains the frames from the NBA games")
    return parser.parse_args()


def run_gui_on_folder(frames_path):
    nba_court_file = frames_path + 'nba_court_white.jpg'
    show_msg = True
    for frame_filename in os.listdir(frames_folder):
        path = frames_folder + 'tagged_data' + os.path.sep
        if frame_filename.endswith(".jpeg") and not os.path.isdir(path + os.path.splitext(frame_filename)[0]):
            root = tk.Tk()
            app = GUI(nba_court_file, frames_path + frame_filename, master=root, show_welcome_message=show_msg)
            show_msg = False
            app.mainloop()
            if (root.getvar(name='result')) == "escape":
                break


if __name__ == "__main__":
    args = parse_args()

    frames_folder = args.games_images_path + os.path.sep
    assert os.path.isdir(frames_folder)
    assert (os.path.isfile(frames_folder + 'nba_court_gray_bitmap.bmp') and \
           os.path.isfile(frames_folder + 'nba_court_phase1_bitmap.bmp') and \
           os.path.isfile(frames_folder + 'nba_court_white.jpg'))
    run_gui_on_folder(frames_folder)