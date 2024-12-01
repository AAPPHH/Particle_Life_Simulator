import dearpygui.dearpygui as dpg

dpg.create_context()

with dpg.window(label="Window a la Abdelali"):

    with dpg.drawlist(width=800, height=800):

        # !!!
        dpg.draw_circle(center=(200, 200), radius=2, color=(255, 255, 255, 255), fill=(255, 255, 255, 255))
        # !!!
        
        dpg.draw_line((10, 10), (100, 100), color=(255, 0, 0, 255), thickness=1)
        dpg.draw_text((0, 0), "Cooler Text", color=(250, 250, 250, 255), size=15)
        dpg.draw_arrow((50, 70), (100, 65), color=(0, 200, 255), thickness=1, size=10)


dpg.create_viewport(title="Eeeeeeeewa", width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()