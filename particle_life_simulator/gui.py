import dearpygui.dearpygui as dpg
from tkinter import *


win= Tk()

win.geometry("650x250")

screen_width = win.winfo_screenwidth()
screen_height = win.winfo_screenheight()


dpg.create_context()

with dpg.window(label="Window a la Abdelali", tag="main_window"):

    with dpg.drawlist(width=16000, height=9000):

        # !!!
        dpg.draw_circle(center=(200, 200), radius=2, color=(255, 0, 0, 255), fill=(255, 255, 255, 255))
        dpg.draw_circle(center=(534, 587), radius=2, color=(0, 255, 0, 255), fill=(255, 255, 255, 255))
        dpg.draw_circle(center=(754, 200), radius=2, color=(0, 0, 255, 255), fill=(255, 255, 255, 255))
        # !!!
        
        dpg.draw_line((600, 600), (700, 700), color=(255, 0, 0, 255), thickness=1)
        dpg.draw_text((400, 400), "Cooler Text", color=(250, 250, 250, 255), size=15)
        dpg.draw_arrow((290, 640), (300, 250), color=(0, 200, 255), thickness=1, size=10)


    # BUTTONS
    start_button = dpg.add_button(label="Start", width=150, height=50)
    pause_button = dpg.add_button(label="Pause", width=150, height=50)
    reset_button = dpg.add_button(label="Reset", width=150, height=50)

    dpg.set_item_pos(start_button, (screen_width / 2 - 80, 20))
    dpg.set_item_pos(pause_button, (screen_width / 2 + 80, 20))
    dpg.set_item_pos(reset_button, (screen_width / 2, 80))


dpg.create_viewport(title="Particle Simulator - Group D", width=screen_width, height=screen_height)

dpg.set_primary_window("main_window", True)


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.maximize_viewport()
dpg.toggle_viewport_fullscreen
dpg.start_dearpygui()
dpg.destroy_context()