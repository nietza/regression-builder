import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
from scipy import stats
#from regression_steps import RegressionSteps

class SaveDialog(ttk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        
        self.title("Save Graph")
        self.geometry("450x250")
        
        self.transient(parent)
        self.grab_set()
        
        content_frame = ttk.Frame(self, padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            content_frame, 
            text="Enter name for the graph",
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=(0, 10))
        
        self.name_entry = ttk.Entry(
            content_frame,
            width=40,
            style='primary'
        )
        self.name_entry.pack(pady=10, ipady=5)
        
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(
            button_frame,
            text="Save",
            style='primary.TButton',
            command=self.save,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            style='secondary.TButton',
            command=self.cancel,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        self.center_window()
        self.name_entry.focus_set()
        
        self.bind('<Return>', lambda e: self.save())
        self.bind('<Escape>', lambda e: self.cancel())

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def save(self):
        name = self.name_entry.get().strip()
        if not name:
            Messagebox.show_warning(
                "Please enter a name",
                "Warning",
                parent=self
            )
            return
        if any(c in r'\/:*?"<>|' for c in name):
            Messagebox.show_warning(
                "Name contains invalid characters",
                "Warning",
                parent=self
            )
            return
        self.result = name
        self.destroy()

    def cancel(self):
        self.destroy()

class RegressionManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression Analysis")
        self.root.geometry("1500x800")
        self.style = ttk.Style("darkly")

        self.dark_mode = True
        
        self.graphs_folder = "saved_graphs"
        if not os.path.exists(self.graphs_folder):
            os.makedirs(self.graphs_folder)

        self.setup_ui()
        self.last_x_values = ""
        self.last_y_values = ""
        self.last_x_name = ""
        self.last_y_name = ""
        
        self.current_graph_name = None
        self.refresh_graph_list()
        self.check_for_updates()

    def toggle_graph_mode(self):
        self.dark_mode = not self.dark_mode
        if hasattr(self, 'current_data'): 
            self.create_graph()  
        else:  
            bg_color = '#2b3e50' if self.dark_mode else 'white'
            self.ax.set_facecolor(bg_color)
            self.figure.set_facecolor(bg_color)
            self.canvas.draw()
        self.toggle_button.config(text="🌙" if self.dark_mode else "☀️")

    def setup_ui(self):
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        left_panel = self.create_input_panel(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 20))
        
        center_panel = self.create_graph_panel(main_container)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = self.create_saved_graphs_panel(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(20, 0))

    def create_input_panel(self, parent):
        frame = ttk.LabelFrame(
            parent,
            text="Input Data",
            padding=15,
            width=300
        )
        
        ttk.Label(
            frame,
            text="X Axis Name"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.x_axis_name = ttk.Entry(frame, width=30)
        self.x_axis_name.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            frame,
            text="Y Axis Name"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.y_axis_name = ttk.Entry(frame, width=30)
        self.y_axis_name.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            frame,
            text="X Values (space-separated)"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.x_values = ttk.Entry(frame, width=30)
        self.x_values.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            frame,
            text="Y Values (space-separated)"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.y_values = ttk.Entry(frame, width=30)
        self.y_values.pack(fill=tk.X, pady=(0, 15))
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(
            button_frame,
            text="Save Graph",
            style='primary.TButton',
            command=self.save_graph,
            width=15
        ).pack(pady=5)
        
        ttk.Button(
            button_frame,
            text="Load Graph",
            style='secondary.TButton',
            command=self.load_graph,
            width=15
        ).pack(pady=5)
        
        return frame

    def create_graph_panel(self, parent):
        frame = ttk.LabelFrame(
            parent,
            text="Regression Analysis",
            padding=15
        )
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(anchor=tk.NE, padx=10, pady=(0, 10))

        prediction_frame = ttk.Frame(button_frame)
        prediction_frame.pack(side=tk.LEFT, padx=(0, 20))

        x_pred_frame = ttk.Frame(prediction_frame)
        x_pred_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(
            x_pred_frame,
            text="if x =",
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=2)
        
        self.x_pred_entry = ttk.Entry(
            x_pred_frame,
            width=8,
            style='primary'
        )
        self.x_pred_entry.pack(side=tk.LEFT, padx=2)
        
        self.x_pred_result = ttk.Label(
            x_pred_frame,
            text="→ y = ",
            font=('Segoe UI', 10)
        )
        self.x_pred_result.pack(side=tk.LEFT, padx=2)
        y_pred_frame = ttk.Frame(prediction_frame)
        y_pred_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(
            y_pred_frame,
            text="if y =",
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=2)
        
        self.y_pred_entry = ttk.Entry(
            y_pred_frame,
            width=8,
            style='primary'
        )
        self.y_pred_entry.pack(side=tk.LEFT, padx=2)
        
        self.y_pred_result = ttk.Label(
            y_pred_frame,
            text="→ x = ",
            font=('Segoe UI', 10)
        )
        self.y_pred_result.pack(side=tk.LEFT, padx=2)

        # inv
        ttk.Label(
            button_frame,
            text="",
            width=70  # space
        ).pack(side=tk.LEFT)

        self.steps_button = ttk.Button(
            button_frame,
            text="📝 Deep Analyze",  
            style='primary.TButton', 
            width=16,
            command=self.show_steps
        )
        self.steps_button.pack(side=tk.LEFT, padx=5)

        self.toggle_button = ttk.Button(
            button_frame,
            text="🌙",
            style='primary.TButton', 
            width=3,
            command=self.toggle_graph_mode
        )
        self.toggle_button.pack(side=tk.LEFT)

        style = ttk.Style()
        style.configure('primary.TButton', font=('Segoe UI', 16), padding=5)

        self.figure = Figure(figsize=(8, 6), facecolor='#2b3e50')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2b3e50')
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.formula_label = ttk.Label(
            frame,
            text="",
            font=('Segoe UI', 12),
            justify=tk.CENTER
        )
        self.formula_label.pack(pady=10)

        self.x_pred_entry.bind('<KeyRelease>', self.predict_y)
        self.y_pred_entry.bind('<KeyRelease>', self.predict_x)

        return frame

    def predict_y(self, event=None):
        try:
            if not hasattr(self, 'current_data'):
                return
                
            x_val = float(self.x_pred_entry.get())
            x = np.array([float(x) for x in self.x_values.get().split()])
            y = np.array([float(y) for y in self.y_values.get().split()])
            
            slope, intercept, _, _, _ = stats.linregress(x, y)
            y_pred = slope * x_val + intercept
            
            self.x_pred_result.config(text=f"→ y = {y_pred:.2f}")
            
            try:
                if hasattr(self, 'pred_scatter'):
                    self.pred_scatter.remove()
            except:
                pass
                
            self.pred_scatter = self.ax.scatter([x_val], [y_pred], color='green', s=100, zorder=5)

            all_x = np.append(x, x_val)
            all_y = np.append(y, y_pred)
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            x_margin = x_range * 0.05
            y_margin = y_range * 0.05
            
            self.ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            self.ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
            
            self.canvas.draw()
            
        except ValueError:
            self.x_pred_result.config(text="→ y = ")
            try:
                if hasattr(self, 'pred_scatter'):
                    self.pred_scatter.remove()
                    x = np.array([float(x) for x in self.x_values.get().split()])
                    y = np.array([float(y) for y in self.y_values.get().split()])
                    x_range = max(x) - min(x)
                    y_range = max(y) - min(y)
                    x_margin = x_range * 0.05
                    y_margin = y_range * 0.05
                    self.ax.set_xlim(min(x) - x_margin, max(x) + x_margin)
                    self.ax.set_ylim(min(y) - y_margin, max(y) + y_margin)
                    self.canvas.draw()
            except:
                pass

    def predict_x(self, event=None):
        try:
            if not hasattr(self, 'current_data'):
                return
                
            y_val = float(self.y_pred_entry.get())
            x = np.array([float(x) for x in self.x_values.get().split()])
            y = np.array([float(y) for y in self.y_values.get().split()])
            
            slope, intercept, _, _, _ = stats.linregress(x, y)
            x_pred = (y_val - intercept) / slope
            
            self.y_pred_result.config(text=f"→ x = {x_pred:.2f}")

            try:
                if hasattr(self, 'pred_scatter'):
                    self.pred_scatter.remove()
            except:
                pass
                
            self.pred_scatter = self.ax.scatter([x_pred], [y_val], color='green', s=100, zorder=5)

            all_x = np.append(x, x_pred)
            all_y = np.append(y, y_val)

            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            x_margin = x_range * 0.05
            y_margin = y_range * 0.05
            
            self.ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            self.ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
            
            self.canvas.draw()
            
        except ValueError:
            self.y_pred_result.config(text="→ x = ")
            try:
                if hasattr(self, 'pred_scatter'):
                    self.pred_scatter.remove()
   
                    x = np.array([float(x) for x in self.x_values.get().split()])
                    y = np.array([float(y) for y in self.y_values.get().split()])
                    x_range = max(x) - min(x)
                    y_range = max(y) - min(y)
                    x_margin = x_range * 0.05
                    y_margin = y_range * 0.05
                    self.ax.set_xlim(min(x) - x_margin, max(x) + x_margin)
                    self.ax.set_ylim(min(y) - y_margin, max(y) + y_margin)
                    self.canvas.draw()
            except:
                pass
    def show_steps(self):
        if not hasattr(self, 'current_data'):
            Messagebox.show_warning(
                "No data to analyze",
                "Warning"
            )
            return
            
        try:
            x = np.array([float(x) for x in self.x_values.get().split()])
            y = np.array([float(y) for y in self.y_values.get().split()])
            import os
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
                
            from regression_steps import RegressionSteps
            steps_window = RegressionSteps(self.root, x, y)
            
        except Exception as e:
            Messagebox.show_error(
                f"Error: {str(e)}",
                "Error"
            )


    def create_saved_graphs_panel(self, parent):
        frame = ttk.LabelFrame(
            parent,
            text="Saved Graphs",
            padding=15,
            width=300
        )
        self.graph_treeview = ttk.Treeview(
            frame, 
            columns=('name', 'date'), 
            show='headings',
            height=10,
            style='Custom.Treeview'
        )
        self.graph_treeview.heading('name', text='Name')
        self.graph_treeview.heading('date', text='Date')
        self.graph_treeview.column('name', width=150)
        self.graph_treeview.column('date', width=150)

        style = ttk.Style()
        style.configure(
            'Custom.Treeview',
            background='#2b3e50',
            foreground='white',
            fieldbackground='#2b3e50',
            font=('Segoe UI', 10)
        )
        style.map('Custom.Treeview', background=[('selected', '#1e2b38')])

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.graph_treeview.yview)
        scrollbar.pack(side='right', fill='y')
        self.graph_treeview.configure(yscrollcommand=scrollbar.set)
        
        self.graph_treeview.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.preview_label = ttk.Label(frame)
        self.preview_label.pack(pady=(0, 15))

        ttk.Button(
            frame,
            text="Delete Graph",
            style='danger.TButton',
            command=self.delete_graph,
            width=15
        ).pack(pady=5)
        
        ttk.Button(
            frame,
            text="Refresh List",
            style='info.TButton',
            command=self.refresh_graph_list,
            width=15
        ).pack(pady=5)
        self.graph_treeview.bind('<<TreeviewSelect>>', self.on_graph_select)
        
        return frame
    def on_graph_select(self, event):
        selected_item = self.graph_treeview.selection()
        if selected_item:
            item_id = selected_item[0]
            graph_name = self.graph_treeview.item(item_id)['values'][0]
            thumb_path = os.path.join(self.graphs_folder, f"{graph_name}_thumb.png")
            if os.path.exists(thumb_path):
                img = Image.open(thumb_path)
                photo = ImageTk.PhotoImage(img)
                self.preview_label.config(image=photo)
                self.preview_label.image = photo
            else:
                self.preview_label.config(image='')

    def create_graph(self):
        try:
            x = np.array([float(x) for x in self.x_values.get().split()])
            y = np.array([float(y) for y in self.y_values.get().split()])
            
            if len(x) != len(y):
                return
            
            self.ax.clear()

            bg_color = '#2b3e50' if self.dark_mode else 'white'
            text_color = 'white' if self.dark_mode else 'black'
            scatter_color = '#5bc0de' if self.dark_mode else '#0066cc' 
            
            self.ax.set_facecolor(bg_color)
            self.figure.set_facecolor(bg_color)

            scatter = self.ax.scatter(x, y, color=scatter_color, s=100, alpha=0.6, label='Data points')
            
            point_annot = self.ax.annotate("", xy=(0,0), xytext=(10,10),
                                    textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc=bg_color, ec="none", alpha=0.8),
                                    color=text_color)
            point_annot.set_visible(False)

            cursor_text = self.ax.text(0.5, 1.02, '', transform=self.ax.transAxes, 
                                    ha='center', va='bottom',
                                    bbox=dict(boxstyle="round", fc=bg_color, ec="none", alpha=0.8),
                                    color=text_color)

            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            
            line_x = np.array([min(x), max(x)])
            line_y = slope * line_x + intercept
            regression_line, = self.ax.plot(line_x, line_y, color='#d9534f', linewidth=2, label='Regression line')

            def distance_to_line(x0, y0, x1, y1, x2, y2):
                return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

            def hover(event):
                if event.inaxes == self.ax:
                    x0, y0 = event.xdata, event.ydata
                    dist = distance_to_line(x0, y0, line_x[0], line_y[0], line_x[1], line_y[1])
                    if dist < 0.05 * (max(y) - min(y)):  
                        x_line = x0
                        y_line = slope * x_line + intercept
                        cursor_text.set_text(f'X: {x_line:.2f}, Y: {y_line:.2f}')
                        if hasattr(self, 'line_point'):
                            self.line_point.remove()
                        self.line_point = self.ax.scatter([x_line], [y_line], color='red', s=50, zorder=5)
                    else:
                        cursor_text.set_text(f'X: {x0:.2f}, Y: {y0:.2f}')
                        if hasattr(self, 'line_point'):
                            self.line_point.remove()
                            delattr(self, 'line_point')
                    
                    cont, ind = scatter.contains(event)
                    if cont:
                        pos = scatter.get_offsets()[ind["ind"][0]]
                        point_annot.xy = pos
                        text = f'x: {pos[0]:.2f}\ny: {pos[1]:.2f}'
                        point_annot.set_text(text)
                        point_annot.set_visible(True)
                    else:
                        point_annot.set_visible(False)
                    self.canvas.draw_idle()
                else:
                    cursor_text.set_text('')
                    point_annot.set_visible(False)
                    if hasattr(self, 'line_point'):
                        self.line_point.remove()
                        delattr(self, 'line_point')
                    self.canvas.draw_idle()

            self.canvas.mpl_connect("motion_notify_event", hover)

            formula = f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}"
            self.ax.text(0.05, 0.95, formula, transform=self.ax.transAxes, 
                        verticalalignment='top', fontsize=10, color=text_color,
                        bbox=dict(facecolor=bg_color, edgecolor='none', alpha=0.7))
            
            self.ax.set_xlabel(self.x_axis_name.get() or "X", color=text_color, fontsize=12)
            self.ax.set_ylabel(self.y_axis_name.get() or "Y", color=text_color, fontsize=12)
            self.ax.tick_params(colors=text_color)
            self.ax.legend(facecolor=bg_color, edgecolor='none', fontsize=10, labelcolor=text_color)
            self.ax.grid(True, color='gray', alpha=0.2)
            
            for spine in self.ax.spines.values():
                spine.set_color(text_color)
            
            self.canvas.draw()
            
            self.current_data = {
                'x_name': self.x_axis_name.get(),
                'y_name': self.y_axis_name.get(),
                'x_values': x.tolist(),
                'y_values': y.tolist(),
                'formula': formula
            }
            
        except ValueError:
            pass

    def check_for_updates(self):
        current_x_values = self.x_values.get()
        current_y_values = self.y_values.get()
        current_x_name = self.x_axis_name.get()
        current_y_name = self.y_axis_name.get()
        
        if (current_x_values != self.last_x_values or 
            current_y_values != self.last_y_values or
            current_x_name != self.last_x_name or
            current_y_name != self.last_y_name):
            
            self.last_x_values = current_x_values
            self.last_y_values = current_y_values
            self.last_x_name = current_x_name
            self.last_y_name = current_y_name
            
            try:
                x_list = [float(x) for x in current_x_values.split()]
                y_list = [float(y) for y in current_y_values.split()]
                
                if len(x_list) == len(y_list) and len(x_list) > 0:
                    self.create_graph()
            except ValueError:
                pass
        
        self.root.after(200, self.check_for_updates)

    def save_graph(self):
        if not hasattr(self, 'current_data'):
            Messagebox.show_error(
                "No graph to save",
                "Error"
            )
            return
        
        dialog = SaveDialog(self.root)
        self.root.wait_window(dialog)
        
        if dialog.result is None:
            return
            
        filename = dialog.result
        data_path = os.path.join(self.graphs_folder, f"{filename}.json")
        img_path = os.path.join(self.graphs_folder, f"{filename}.png")
        thumb_path = os.path.join(self.graphs_folder, f"{filename}_thumb.png")
        
        if os.path.exists(data_path) or os.path.exists(img_path):
            if not Messagebox.show_question(
                "A graph with this name already exists. Do you want to overwrite it?",
                "Warning"
            ):
                return
        
        try:
          
            self.current_data['save_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(data_path, 'w') as f:
                json.dump(self.current_data, f)
            
           
            self.figure.savefig(img_path, facecolor=self.ax.get_facecolor(), edgecolor='none')
            thumb_size = (100, 75)  
            img = Image.open(img_path)
            img.thumbnail(thumb_size)
            img.save(thumb_path)
            Messagebox.show_info("Graph saved successfully", "Success")
            self.refresh_graph_list()
        except Exception as e:
            Messagebox.show_error(f"Failed to save graph: {str(e)}", "Error")

    def load_graph(self):
        selected_item = self.graph_treeview.selection()
        if not selected_item:
            Messagebox.show_error("No graph selected", "Error")
            return
            
        filename = self.graph_treeview.item(selected_item[0])['values'][0]
        data_path = os.path.join(self.graphs_folder, f"{filename}.json")
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                
            self.x_axis_name.delete(0, tk.END)
            self.x_axis_name.insert(0, data['x_name'])
            
            self.y_axis_name.delete(0, tk.END)
            self.y_axis_name.insert(0, data['y_name'])
            
            self.x_values.delete(0, tk.END)
            self.x_values.insert(0, ' '.join(map(str, data['x_values'])))
            
            self.y_values.delete(0, tk.END)
            self.y_values.insert(0, ' '.join(map(str, data['y_values'])))
            
        except Exception as e:
            Messagebox.show_error(f"Failed to load graph: {str(e)}", "Error")

    def delete_graph(self):
        selected_items = self.graph_treeview.selection()
        if not selected_items:
            Messagebox.show_error(
                "No graph selected",
                "Error"
            )
            return
        
        for selected_item in selected_items:
            filename = self.graph_treeview.item(selected_item)['values'][0]
            data_path = os.path.join(self.graphs_folder, f"{filename}.json")
            img_path = os.path.join(self.graphs_folder, f"{filename}.png")
            thumb_path = os.path.join(self.graphs_folder, f"{filename}_thumb.png")
            
            try:
                if os.path.exists(data_path):
                    os.remove(data_path)
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(thumb_path):
                    os.remove(thumb_path)
                
                Messagebox.show_info(
                    "Graph deleted successfully",
                    "Success"
                )
                self.refresh_graph_list()
            except Exception as e:
                Messagebox.show_error(
                    f"Failed to delete graph: {str(e)}",
                    "Error"
                )

    def refresh_graph_list(self):
        for item in self.graph_treeview.get_children():
            self.graph_treeview.delete(item)
        
        try:
            files = sorted([f[:-5] for f in os.listdir(self.graphs_folder) if f.endswith('.json')])
            for file in files:
                with open(os.path.join(self.graphs_folder, f"{file}.json"), 'r') as f:
                    data = json.load(f)
                save_date = data.get('save_date', 'Unknown')
                self.graph_treeview.insert('', 'end', values=(file, save_date))
        except Exception as e:
            Messagebox.show_error(f"Failed to refresh list: {str(e)}", "Error")

    def export_graph(self):
        if not hasattr(self, 'current_data'):
            Messagebox.show_error(
                "No graph to export",
                "Error"
            )
            return
            
        try:
            export_path = os.path.join(
                self.graphs_folder,
                f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            self.figure.savefig(
                export_path,
                dpi=300,
                bbox_inches='tight',
                facecolor=self.ax.get_facecolor(),
                edgecolor='none'
            )
            Messagebox.show_info(
                f"Graph exported to {export_path}",
                "Success"
            )
        except Exception as e:
            Messagebox.show_error(
                f"Failed to export graph: {str(e)}",
                "Error"
            )

    def clear_inputs(self):
        self.x_axis_name.delete(0, tk.END)
        self.y_axis_name.delete(0, tk.END)
        self.x_values.delete(0, tk.END)
        self.y_values.delete(0, tk.END)
        self.formula_label.config(text="")
        self.ax.clear()
        self.ax.set_facecolor('#2b3e50')
        self.canvas.draw()

    def show_about(self):
        about_window = ttk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("400x350")
        
        content_frame = ttk.Frame(about_window, padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            content_frame,
            text="You won't find any help here, but anyway",
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=(0, 10))
        
        ttk.Label(
            content_frame,
            text="Yea, this shit can make linear regressions",
            wraplength=350,
            justify=tk.CENTER
        ).pack(pady=(0, 20))
        
        ttk.Label(
            content_frame,
            text="Features:",
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor=tk.W)
        
        features = [
            "• Real-time regression analysis",
            "• Dark Mode",
            "• Save and load regression models",
            "• Export high-quality graphs",
            "• Fucking hell"
        ]
        
        for feature in features:
            ttk.Label(
                content_frame,
                text=feature
            ).pack(anchor=tk.W, pady=2)
        
        ttk.Button(
            content_frame,
            text="Close",
            style='primary.TButton',
            command=about_window.destroy,
            width=15
        ).pack(pady=(20, 0))

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.clear_inputs)
        file_menu.add_command(label="Save Graph", command=self.save_graph)
        file_menu.add_command(label="Load Graph", command=self.load_graph)
        file_menu.add_separator()
        file_menu.add_command(label="Export Graph", command=self.export_graph)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear All", command=self.clear_inputs)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

def main():
    root = tk.Tk()
    app = RegressionManager(root)
    app.create_menu()
    root.mainloop()

if __name__ == "__main__":
    main()