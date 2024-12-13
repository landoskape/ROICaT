import tkinter as tk
from tkinter import ttk
import numpy as np
from pathlib import Path
import csv
from typing import Optional, List, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as mplPath
import scipy.interpolate
from typing import Optional, List, Tuple


class IntegratedLabeler:
    """
    A graphical interface for labeling image classes. The class displays a
    sequence of images in the left panel which can be labelled by pressing keys
    and the right panel is a scatterplot of an embedding of each image with the
    option to overlay images on the scatterplot. The user can use a lasso tool
    to select points on the scatterplot and these points will be shown to the
    user on the left panel for labelling. 
    The title of the window is the current image index. The overlays can be 
    toggled by pressing Control-Shift-T. The classification label and image
    index are stored as the ``self.labels_`` attribute and saved to a CSV file
    in self.path_csv. 

    Args:
        images (np.ndarray): 
            A numpy array of images. Either 3D: *(n_images, height, width)* or
            4D: *(n_images, height, width, n_channels)*. Images should be scaled
            between 0 and 255 and will be converted to uint8.
        embeddings (np.ndarray):
            A numpy array of embeddings for each image. Should be shape 
            *(n_images, 2)*.
        idx_images_overlay (np.ndarray):
            A numpy array of indices of images to overlay on the scatterplot.
        idx_selection (List[int]):
            A list of indices to select from the image array. If ``None``, all
            images will be selected. (Default is ``None``)
        figsize (float):
            The size of each panel in the figure (width and height). (Default is *5*)
        size_images_overlay (Optional[float]):
            The size of the images to overlay. If ``None``, the size is calculated
            based on nearest neighbors. (Default is ``None``)
        crop_images_overlay (Optional[float]):
            The fraction of the image to crop on each side. (Default is *0.35*)
        frac_overlap_allowed (float):
            The fraction of overlap allowed between images. (Default is *0.5*)
        image_overlay_raster_size (Tuple[int, int]):
            The size of the raster for the composite overlay. 
            (Default is*(1000, 1000)*)
        alpha_points (float):
            The transparency of the scatterplot points. (Default is *0.5*)
        size_points (float):
            The size of the scatterplot points. (Default is *20*)
        normalize_images (bool):
            Whether to normalize the images between min and max values. (Default
            is ``True``)
        verbose (bool):
            Whether to print status updates. (Default is ``True``)
        path_csv (Optional[str]):
            Path to the CSV file for saving results. If ``None``, results will
            not be saved.
        save_csv (bool):
            Whether to save the results to a CSV. (Default is ``True``)
        key_end (str):
            Key to press to end the session. (Default is ``'Escape'``)
        key_prev (str):
            Key to press to go back to the previous image. (Default is
            ``'Left'``)
        key_next (str):
            Key to press to go to the next image. (Default is ``'Right'``)


    Example:
        .. highlight:: python
        .. code-block:: python
            with IntegratedLabeler(
                images,
                embeddings=emb,
                idx_images_overlay=idx_images_overlay,
                size_images_overlay=0.25,
                frac_overlap_allowed=0.25,
                crop_images_overlay=0.25,
                alpha_points=1.0,
                size_points=3.0,
            ) as labeler:
                labeler.run()
            path_csv, labels = labeler.path_csv, labeler.labels_

    Attributes:
        path_csv (str): 
            Path to the CSV file for saving results. If ``None``, results will
            not be saved.
        save_csv (bool):
            Whether to save the results to a CSV. (Default is ``True``)
        labels_ (list):
            A list of tuples containing the image index and classification label
            for each image. The list is saved to a CSV file in self.path_csv.
    """
    def __init__(
        self, 
        images: np.ndarray,
        embeddings: np.ndarray = None,
        idx_images_overlay: Optional[np.ndarray] = None,
        idx_selection: Optional[List[int]] = None,
        figsize: float = 5,
        size_images_overlay: Optional[float] = None,
        crop_images_overlay: Optional[float] = 0.35,
        frac_overlap_allowed: float = 0.5,
        image_overlay_raster_size: Tuple[int, int] = (1000, 1000),
        alpha_points: float = 0.5,
        size_points: float = 20,
        normalize_images: bool = True,
        verbose: bool = True,
        path_csv: Optional[str] = None, 
        save_csv: bool = True,
        key_end: str = 'Escape', 
        key_prev: str = 'Left',
        key_next: str = 'Right',
        
    ):
        """Build the IntegratedLabeler Object."""
        # Data attributes
        self.images = images
        self.embeddings = embeddings
        self.idx_images_overlay = idx_images_overlay

        # Plotting properties
        self.size_images_overlay = size_images_overlay
        self.crop_images_overlay = crop_images_overlay
        self.frac_overlap_allowed = frac_overlap_allowed
        self.image_overlay_raster_size = image_overlay_raster_size
        self.alpha_points = alpha_points
        self.size_points = size_points
        self._normalize_images = normalize_images
        self.figsize = figsize
        self._verbose = verbose
        
        # Saving properties
        import tempfile
        import datetime
        self.path_csv = path_csv if path_csv is not None else str(Path(tempfile.gettempdir()) / ('roicat_labels_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'))
        self._save_csv = save_csv

        # Initialize the selection and current index
        self._index = -1 # Start at -1 so that the first image is 0
        self.update_selection(idx_selection)
    
        # Calculate data limits with padding
        self.data_limits = self._calculate_data_limits()
        
        # Create composite overlay if idx_images_overlay provided
        if self.idx_images_overlay is not None:
            self._create_composite_overlay()
            self.show_overlay = True

        # Results will be stored here
        self.labels_ = {}

        # Initialize GUI elements
        self._img_tk = None
        self._key_end = key_end
        self._key_prev = key_prev
        self._key_next = key_next
        self._root = None
        self.__call__ = self.run

        
    def run(self):
        """
        Runs the image labeler with both image display and matplotlib panels.
        """
        try:
            self._root = tk.Tk()
            self._root.title("Image Labeler")

            # Create main container
            main_container = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
            main_container.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

            # Left panel for image display
            left_panel = ttk.Frame(main_container)
            self._img_fig, self._img_ax = plt.subplots(figsize=(self.figsize, self.figsize))
            self._img_canvas = FigureCanvasTkAgg(self._img_fig, master=left_panel)
            self._img_canvas.draw()
            self._img_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            main_container.add(left_panel)

            # Right panel for scatterplot
            right_panel = ttk.Frame(main_container)
            self._scatter_fig, self._scatter_ax = plt.subplots(figsize=(self.figsize, self.figsize))
            self._scatter_canvas = FigureCanvasTkAgg(self._scatter_fig, master=right_panel)
            self._scatter_canvas.draw()
            self._scatter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            main_container.add(right_panel)

            # Make figures fill the space
            self._img_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
            self._scatter_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
            
            # Build scatter plot
            self._build_scatter_plot()

            # Bind keys
            self._root.bind("<Key>", self.classify)
            if self._key_end:
                self._root.bind(f'<Key-{self._key_end}>', self.end_session)
            if self._key_prev:
                self._root.bind(f'<Key-{self._key_prev}>', self.prev_img)
            if self._key_next:
                self._root.bind(f'<Key-{self._key_next}>', self.next_img)
            if self.idx_images_overlay is not None:
                self._root.bind(f'<Control-T>', self._toggle_overlay)
            self._root.protocol("WM_DELETE_WINDOW", self._on_closing)

            # Start the session
            self.next_img()
            self._root.mainloop()
            
        except Exception as e:
            warnings.warn('Error initializing image labeler: ' + str(e))
    
    def update_selection(self, idx_selection: List[int]):
        """
        Updates the selection of images to classify. The selection is a list of
        indices to select from the image array. Will show the first image in the
        new selection.

        Args:
            idx_selection (List[int]):
                A list of indices to select from the image array.
        """
        if idx_selection is not None:
            # Check if provided list is valid
            if min(idx_selection) < 0 or max(idx_selection) >= len(self.images):
                raise ValueError('idx_selection exceeds range of images (must be in [0, len(image_array)-1]).')
        self._idx_selection = idx_selection if idx_selection is not None else list(range(len(self.images)))

        self._index = -1
        if hasattr(self, '_root') and self._root is not None:
            # Only attempt to go to next image if the window is open
            self.next_img()

    def _on_closing(self):
        from tkinter import messagebox
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.end_session(None)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_session(None)
    
    def _get_current_idx(self):
        """Central method for getting the current image index."""
        return int(self._idx_selection[self._index])

    def next_img(self, event=None):
        """Displays the next image and updates the matplotlib plot."""
        self._index += 1
        if self._index < len(self._idx_selection):
            # Update image display
            im = self.images[self._get_current_idx()]
            im = (im / np.max(im)) * 255 if self._normalize_images else im
            # Update data of _img_ax
            self._img_ax.clear()
            self._img_ax.imshow(im, cmap='gray')
            self._img_ax.text(0, 0, "Press any key to label ROI\nPress left or right to switch ROIs", color="white", fontsize=12, ha="left", va="top")
            self._img_ax.set_xticks([])
            self._img_ax.set_yticks([])
            self._img_canvas.draw()
            
            self._root.title(str(self._get_current_idx()))

            if hasattr(self, 'scatter'):
                # Update colors to show selection
                colors = np.array(['gray'] * len(self.embeddings))
                colors[list(self._idx_selection)] = 'red'
                self.scatter.set_color(colors)

                # Update current scatter point
                self.current_scatter.set_offsets([self.embeddings[self._get_current_idx(), :]])
                self._scatter_canvas.draw()
        else:
            # Loop index back to start of current selection
            self._index = 0

    def prev_img(self, event=None):
        """
        Displays the previous image in the array.
        """
        self._index -= 2
        self.next_img()

    def end_session(self, event):
        """Ends the session and cleans up matplotlib resources."""
        try:
            if self._root is not None:
                self._root.quit()
                self._root.destroy()
                self._root = None
        except:
            pass
        
        try:
            if hasattr(self, '_img_fig') and self._img_fig is not None:
                plt.close(self._img_fig)
            if hasattr(self, '_scatter_fig') and self._scatter_fig is not None:
                plt.close(self._scatter_fig)
            
            self._img_fig = None
            self._scatter_fig = None
            self._img_canvas = None
            self._scatter_canvas = None
            
            import gc
            gc.collect()
        except:
            pass

        import gc
        gc.collect()
        gc.collect()

    def save_classification(self):
        """
        Saves the classification results to a CSV file.
        This function does not append, it overwrites the entire file.
        The file contains two columns: 'image_index' and 'label'.
        """
        ## make directory if it doesn't exist
        Path(self.path_csv).parent.mkdir(parents=True, exist_ok=True)
        ## Save the results
        with open(self.path_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('image_index', 'label'))
            writer.writerows(self.labels_.items())

    def get_labels(self, kind: str = 'dict') -> Union[dict, List[Tuple[int, str]], dict]:
        """
        Returns the labels. The format of the output is determined by the ``kind`` parameter. 
        If the labels dictionary is empty, returns ``None``. RH 2023

        Args:
            kind (str): 
                The type of object to return. (Default is ``'dict'``) \n
                * ``'dict'``: {idx: label, idx: label, ...}
                * ``'list'``: [label, label, ...] where the index is the image
                  index and unlabeled images are represented as ``'None'``.
                * ``'dataframe'``: {'index': [idx, idx, ...], 'label': [label, label, ...]}
                  This can be converted to a pandas dataframe with:
                  pd.DataFrame(self.get_labels('dataframe'))

        Returns:
            (Union[dict, List[Tuple[int, str]], dict]): 
                Depending on the ``kind`` parameter, it returns either: \n
                * dict: 
                    A dictionary where keys are the image indices and values are
                    the labels.
                * List[Tuple[int, str]]: 
                    A list of tuples, where each tuple contains an image index
                    and a label.
                * dict: 
                    A dictionary with keys 'index' and 'label' where values are
                    lists of indices and labels respectively.
        """
        ## if the dict is empty, return None
        if len(self.labels_) == 0:
            return None
        
        if kind == 'dict':
            return self.labels_
        elif kind == 'list':
            out = ['None',] * len(self.images)
            for idx, label in self.labels_.items():
                out[idx] = label
            return out
        elif kind == 'dataframe':
            import pandas as pd
            return pd.DataFrame(index=list(self.labels_.keys()), data={'label': list(self.labels_.values())})

    def classify(self, event):
        """
        Adds the current image index and pressed key as a label.
        Then saves the results and moves to the next image.

        Args:
            event (tkinter.Event):
                A tkinter event object.
        """
        # Prevent classify from running with any special keys
        if event.state != 8:
            return 
        label = event.char
        if label != '':
            print(f'Image {self._get_current_idx()}: {label}') if self._verbose else None
            self.labels_.update({self._get_current_idx(): str(label)})  ## Store the label
            self.save_classification() if self._save_csv else None ## Save the results
            self.next_img()  ## Move to the next image

    def _toggle_overlay(self, event):
        """Toggle the overlay on and off"""
        if self.idx_images_overlay is not None:
            self.show_overlay = not self.show_overlay
            self.im_composite.set_visible(self.show_overlay)
            self._scatter_canvas.draw()
            
    def _on_select(self, verts):
        """Handle lasso selection"""
        path = mplPath(verts)
        points = self.scatter.get_offsets()
        mask = path.contains_points(points)
        selected = np.sort(np.array(np.where(mask)[0]))
        self.update_selection(selected)

    def _build_scatter_plot(self):
        self.scatter = self._scatter_ax.scatter(
            self.embeddings[:, 0], self.embeddings[:, 1],
            alpha=self.alpha_points,
            s=self.size_points,
            c="gray",
            picker=True,
            zorder=1,
        )

        self.current_scatter = self._scatter_ax.scatter(
            self.embeddings[self._get_current_idx(), 0],
            self.embeddings[self._get_current_idx(), 1],
            alpha=1.0,
            s=self.size_points * 4,
            c="blue",
            zorder=2,
        )

        # Create composite overlay if images provided
        if self.idx_images_overlay is not None:
            # Show the overlay
            self.im_composite = self._scatter_ax.imshow(
                self.composite_overlay,
                extent=[
                    self.data_limits[0][0],
                    self.data_limits[1][0],
                    self.data_limits[0][1],
                    self.data_limits[1][1]
                ],
                aspect='auto',
                zorder=1000,
            )

        self._scatter_ax.set_xlim(self.data_limits[0][0], self.data_limits[1][0])
        self._scatter_ax.set_ylim(self.data_limits[0][1], self.data_limits[1][1])
        self._scatter_ax.set_xticks([])
        self._scatter_ax.set_yticks([])

        self._scatter_ax.text(
            self.data_limits[0][0], 
            self.data_limits[1][1], 
            "Draw lasso to select points for labeling", 
            color="black", 
            fontsize=12, 
            ha="left", 
            va="top"
        )
        
        self._scatter_ax.text(
            self.data_limits[0][0], 
            self.data_limits[0][1], 
            "Press Control-Shift-T to toggle overlay", 
            color="black", 
            fontsize=12, 
            ha="left", 
            va="bottom"
        )
        
        # Add selector for picking points
        self.lasso = LassoSelector(self._scatter_ax, onselect=self._on_select, button=1)

    def _create_composite_overlay(self):
        """Create a single composite image with all overlays"""
        if self.size_images_overlay is None:
            # Calculate optimal size based on nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2).fit(self.embeddings[self.idx_images_overlay])
            distances = nn.kneighbors(self.embeddings[self.idx_images_overlay])[0]
            min_dist = np.min(distances[:, 1])
            self.size_images_overlay = min_dist * (1 + self.frac_overlap_allowed)

        min_emb = np.nanmin(self.embeddings, axis=0)  ## shape (2,)
        max_emb = np.nanmax(self.embeddings, axis=0)  ## shape (2,)
        range_emb = max_emb - min_emb  ## shape (2,)
        aspect_ratio_ims = (range_emb[1] / range_emb[0])  ## shape (1,)

        assert isinstance(self.size_images_overlay, (int, float, np.ndarray)), 'size_images_overlay must be an int, float, or shape (2,) numpy array'
        if isinstance(self.size_images_overlay, (int, float)):
            self.size_images_overlay = np.array([self.size_images_overlay / aspect_ratio_ims, self.size_images_overlay])
        assert self.size_images_overlay.shape == (2,), 'size_images_overlay must be an int, float, or shape (2,) numpy array'

        # Create empty canvas
        iors = self.image_overlay_raster_size
        canvas = np.zeros((*iors, 4))  # RGBA
        
        # Create interpolators for mapping data coordinates to pixel coordinates
        interp_x = scipy.interpolate.interp1d(
            [self.data_limits[0][0], self.data_limits[1][0]],
            [0, iors[0]]
        )
        interp_y = scipy.interpolate.interp1d(
            [self.data_limits[0][1], self.data_limits[1][1]],
            [0, iors[1]]
        )
        
        # Calculate size of each image in pixels
        range_x = self.data_limits[1][0] - self.data_limits[0][0]
        range_y = self.data_limits[1][1] - self.data_limits[0][1]
        size_x = int((self.size_images_overlay[0] / range_x) * iors[0])
        size_y = int((self.size_images_overlay[1] / range_y) * iors[1])
        
        xwidth = self.images.shape[2]
        ywidth = self.images.shape[1]
        crop_value = min(1.0, self.crop_images_overlay)
        crop_value = max(0.1, crop_value)
        x_crop_points = int((xwidth - crop_value * xwidth)/2)
        y_crop_points = int((ywidth - crop_value * ywidth)/2)
        for idx in self.idx_images_overlay:
            # Normalize and convert to RGB if grayscale
            img = self.images[idx][x_crop_points:-x_crop_points, y_crop_points:-y_crop_points]
            if img.ndim == 2:
                img = (img - img.min()) / (img.max() - img.min())
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3:
                img = (img - img.min()) / (img.max() - img.min())
            
            # Resize image
            coords = np.stack(np.meshgrid(
                np.linspace(0, img.shape[0], size_x),
                np.linspace(0, img.shape[1], size_y)
            ), axis=-1)
            
            img_resized = scipy.interpolate.interpn(
                (np.arange(img.shape[0]), np.arange(img.shape[1])),
                img,
                coords,
                method='linear',
                bounds_error=False,
                fill_value=0
            )
            
            # Calculate position
            x = int(interp_x(self.embeddings[idx, 0]))
            y = int(interp_y(self.embeddings[idx, 1]))
            
            # Calculate bounds
            x1 = max(0, x - size_x // 2)
            x2 = min(iors[0], x + size_x // 2)
            y1 = max(0, y - size_y // 2)
            y2 = min(iors[1], y + size_y // 2)
            
            # Add to canvas
            canvas[y1:y2, x1:x2, :3] = img_resized[:y2-y1, :x2-x1]
            canvas[y1:y2, x1:x2, 3] = 1.0  # Alpha channel
        
        self.composite_overlay = np.flipud(canvas)  # Flip because imshow origin is bottom left

    def _calculate_data_limits(self):
        """Calculate data limits with padding"""
        pad = 0.07
        min_vals = np.min(self.embeddings, axis=0)
        max_vals = np.max(self.embeddings, axis=0)
        range_vals = max_vals - min_vals
        return (
            min_vals - range_vals * pad,
            max_vals + range_vals * pad
        )

    