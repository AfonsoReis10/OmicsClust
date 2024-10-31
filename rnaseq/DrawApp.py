import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class DrawApp:
    def __init__(self, root, grid_size, mask_name, background_image_matrix):
        self.root = root
        self.root.title("Grid Drawing App")
        self.grid_size = grid_size

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calculate maximum cell size to fit within the screen dimensions
        self.cell_size = min((screen_width - 200) // grid_size, (screen_height - 200) // grid_size)
        self.axis_offset = 30
        self.label_padding = 10

        self.cell_size = 20  
        self.axis_offset = 30  
        self.label_padding = 10 

        self.canvas_size = self.cell_size * self.grid_size
        self.total_canvas_size = self.canvas_size + 2 * self.axis_offset

        self.canvas = tk.Canvas(root, width=self.total_canvas_size, height=self.total_canvas_size, bg='white')
        self.canvas.pack()

        self.background_image_matrix=background_image_matrix

        self.bg_image = self.convert_2d_to_rgb(background_image_matrix)
        self.bg_image = self.bg_image.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image, master=root)

        self.image_on_canvas = self.canvas.create_image(self.axis_offset, self.axis_offset, image=self.bg_photo, anchor=tk.NW)
        self.canvas.bind("<B1-Motion>", self.track_cursor)
        self.canvas.bind("<Button-1>", self.click)
        
        self.mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.rectangles = []
        self.norm_matrix = (background_image_matrix - background_image_matrix.min()) / (background_image_matrix.max() - background_image_matrix.min())
        colormap = plt.get_cmap('jet')

        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                color_tuple = colormap(self.norm_matrix[i, j])[:3]  
                color = mcolors.rgb2hex(color_tuple)  
                
                rect = self.canvas.create_rectangle(
                    j * self.cell_size + self.axis_offset, 
                    (self.grid_size - i - 1) * self.cell_size + self.axis_offset,
                    (j + 1) * self.cell_size + self.axis_offset, 
                    (self.grid_size - i) * self.cell_size + self.axis_offset,
                    fill=color, outline='#808080'
                )
                row.append(rect)
            self.rectangles.append(row)

        for i in range(self.grid_size):
            self.canvas.create_text(i * self.cell_size + self.axis_offset + self.cell_size // 2, 
                                    self.total_canvas_size - self.axis_offset // 2,
                                    text=str(i), fill="black", anchor=tk.N)

        for j in range(self.grid_size):
            self.canvas.create_text(self.axis_offset // 2, 
                                    self.total_canvas_size - (j * self.cell_size + self.axis_offset + self.cell_size // 2),
                                    text=str(j), fill="black", anchor=tk.E)

        self.save_button = tk.Button(root, text="Save Mask", command=lambda: self.save_mask(mask_name))
        self.save_button.pack(pady=(10, 0))
        self.root.bind('<Return>', lambda event: self.save_mask(mask_name))

        self.last_pixel = None

    def convert_2d_to_rgb(self, matrix):
        """Convert a 2D NumPy array to a colored RGB image using a colormap."""
        # Normalize the matrix to the range 0-1
        norm_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        colormap = plt.get_cmap('jet')
        colored_matrix = colormap(norm_matrix)
        colored_matrix = (colored_matrix[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(colored_matrix)

    def track_cursor(self, event):
        current_pixel = ((event.x - self.axis_offset) // self.cell_size, 
                         (event.y - self.axis_offset) // self.cell_size)
        if current_pixel != self.last_pixel:
            self.toggle_pixel(event.x, event.y)
            self.last_pixel = current_pixel

    def click(self, event):
        self.toggle_pixel(event.x, event.y)

    def toggle_pixel(self, x, y):
        i = (y - self.axis_offset) // self.cell_size
        j = (x - self.axis_offset) // self.cell_size
        if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
            current_color = self.canvas.itemcget(self.rectangles[self.grid_size - i - 1][j], "fill")
            if current_color == "#000000":
                color_tuple = plt.get_cmap('jet')(self.norm_matrix[self.grid_size - i - 1, j])[:3]
                color = mcolors.rgb2hex(color_tuple)
                self.canvas.itemconfig(self.rectangles[self.grid_size - i - 1][j], fill=color)
                self.mask[self.grid_size - i - 1, j] = 0
            else:
                self.canvas.itemconfig(self.rectangles[self.grid_size - i - 1][j], fill='#000000')
                self.mask[self.grid_size - i - 1, j] = 1

    def save_mask(self, mask_name):
        np.save('{}.npy'.format(mask_name), self.mask)
        print("Mask saved as '{}.npy'".format(mask_name))
        print(self.mask)


