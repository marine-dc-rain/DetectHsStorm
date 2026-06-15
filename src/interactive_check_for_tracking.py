import sys
import os
import numpy as np
import xarray as xr

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import cartopy.crs as ccrs

default_cycler = mpl.rcParams['axes.prop_cycle'].by_key()['color']
# ===============================
# UTILITIES
# ===============================


def duplicate_longitudes(lon, lat, values=None):
    """Duplicate data for wrap-around visualization"""
    lon_ext = np.concatenate([lon, lon + 360])
    lat_ext = np.concatenate([lat, lat])

    if values is not None:
        val_ext = np.concatenate([values, values])
        return lon_ext, lat_ext, val_ext

    return lon_ext, lat_ext


def find_nearest_point(click_lon, click_lat, lons, lats, tol=2):
    dist = np.sqrt((lons - click_lon) ** 2 + (lats - click_lat) ** 2)
    idx = np.argmin(dist)
    if dist[idx] < tol:
        return idx
    return None


# ===============================
# GUI APPLICATION
# ===============================


class StormEditor(QMainWindow):

    def __init__(self, swh_file, storm_file):
        super().__init__()
        self.storm_file = storm_file
        # Load datasets
        self.ds_swh = xr.open_dataset(swh_file)
        self.ds_storm = xr.open_dataset(storm_file)
        self.time_steps = self.ds_swh.time.values.copy()
        self.current_time_idx = 0

        # Copy data to mutable arrays
        self.lon = self.ds_storm["longitude"].values.copy()
        self.lat = self.ds_storm["latitude"].values.copy()
        self.time = self.ds_storm["time"].values.copy()
        self.label = self.ds_storm["numStorm"].values.copy()

        self.selected_points = []

        self.init_ui()
        self.plot()

    # ---------------------------
    def init_ui(self):
        self.setWindowTitle("Storm trajectory editor")

        self.figure = plt.figure(figsize=(18, 15))
        self.canvas = FigureCanvas(self.figure)

        self.canvas.mpl_connect("button_press_event", self.on_click)

        btn_next = QPushButton("Next time")
        btn_prev = QPushButton("Prev time")
        btn_merge = QPushButton("Merge")
        btn_split = QPushButton("Split")
        btn_save = QPushButton("Save")

        btn_next.clicked.connect(self.next_time)
        btn_prev.clicked.connect(self.prev_time)
        btn_merge.clicked.connect(self.merge)
        btn_split.clicked.connect(self.split)
        btn_save.clicked.connect(self.save)

        self.info = QLabel("Select points...")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.info)
        layout.addWidget(btn_merge)
        layout.addWidget(btn_split)
        layout.addWidget(btn_prev)
        layout.addWidget(btn_next)
        layout.addWidget(btn_save)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ---------------------------
    def plot(self):
        self.figure.clear()

        ax = plt.axes()  # projection=ccrs.PlateCarree())
        # ax.coastlines()

        t_val = self.time_steps[self.current_time_idx]

        # SWH field
        swh = self.ds_swh["hs"].isel(time=self.current_time_idx)

        ind_lon = np.nonzero(self.ds_swh["longitude"].values < 0)[0]
        swh_bis = np.concat([swh.values, swh.isel(longitude=ind_lon).values], axis=1)
        lon_bis = np.concat(
            [self.ds_swh["longitude"].values, self.ds_swh["longitude"].isel(longitude=ind_lon).values + 360]
        )
        ax.pcolormesh(lon_bis, self.ds_swh["latitude"], swh_bis)  # , transform=ccrs.PlateCarree())

        # Storm points at T and T-1
        mask_t = self.time == t_val
        mask_tm1 = self.time == self.get_prev_time(t_val)

        self.plot_points(ax, mask_tm1, 'v', "blue", 'T-1')
        self.plot_points(ax, mask_t, '^', "red", 'T')

        # ax.set_global()
        ax.set_xlim((-180, 360))
        ax.set_title(t_val)
        ax.legend(loc='upper left')
        self.canvas.draw()

    # ---------------------------
    def plot_points(self, ax, mask, marker, color, label):
        idxs = np.where(mask)[0]

        lons = self.lon[idxs]
        lats = self.lat[idxs]

        idxs_alone = []
        for idx in idxs:
            t_current = self.time_steps[self.current_time_idx]
            t_prev = self.get_prev_time(t_current)
            pair_idx = self.get_pair(idx, t_current, t_prev)
            if pair_idx is None:
                idxs_alone.append(idx)

        lons_alone = self.lon[idxs_alone]
        lats_alone = self.lat[idxs_alone]

        lons_ext, lats_ext = duplicate_longitudes(lons, lats)
        lons_ext_alone, lats_ext_alone = duplicate_longitudes(lons_alone, lats_alone)

        ax.scatter(lons_ext, lats_ext, marker=marker, color=color, s=30, label=label)  # , transform=ccrs.PlateCarree()
        ax.scatter(lons_ext_alone, lats_ext_alone, marker=marker, color=color, edgecolor="w", s=30)
        # HIGHLIGHT selected
        for i in idxs:
            if i in self.selected_points:
                ax.scatter(
                    self.lon[i],
                    self.lat[i],
                    color="yellow",
                    s=120,
                    edgecolor="black",
                    zorder=5,
                )  # transform=ccrs.PlateCarree(),

        # trajectories
        for lab in np.unique(self.label[idxs]):
            idx = self.label == lab
            lon_traj = self.lon[idx].copy()
            if np.std(lon_traj) > 100:
                lon_traj[lon_traj < 0] = lon_traj[lon_traj < 0] + 360
            ax.plot(
                lon_traj, self.lat[idx], color=default_cycler[lab % 10], linewidth=1
            )  # , transform=ccrs.PlateCarree()

    # ---------------------------
    def get_next_time(self, t):
        idx = np.where(self.ds_swh.time.values == t)[0][0]
        if idx < len(self.ds_swh.time) - 1:
            return self.ds_swh.time.values[idx + 1]
        return t

    # ---------------------------
    def get_pair(self, idx, t1, t2):
        """Return paired point idx if exists (T-1 <-> T with same label)"""
        t = self.time[idx]  # time of selected obs
        lab = self.label[idx]  # label of selected obs

        # paired points must have same label at adjacent time
        associated_t = t1 if t == t2 else t2

        pair_mask = (self.label == lab) & (self.time == associated_t)

        if np.any(pair_mask):
            return np.where(pair_mask)[0][0]

        # # also check forward pairing
        # t_next = self.get_next_time(t)
        # pair_mask = (self.label == lab) & (self.time == t_next)

        # if np.any(pair_mask):
        #     return np.where(pair_mask)[0][0]

        return None

    # ---------------------------
    def on_click(self, event):
        if event.inaxes is None:
            return

        click_lon = event.xdata
        click_lat = event.ydata

        # Current time values
        t_current = self.time_steps[self.current_time_idx]
        t_prev = self.get_prev_time(t_current)

        # mask only T and T-1 points
        valid_mask = (self.time == t_current) | (self.time == t_prev)

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return

        idx_local = find_nearest_point(click_lon, click_lat, self.lon[valid_indices], self.lat[valid_indices], tol=3)

        if idx_local is None:
            return

        # Convert back to global index
        idx = valid_indices[idx_local]

        if idx is None:
            return

        t_clicked = self.time[idx]

        if idx in self.selected_points:
            self.selected_points = []  # reset when clicking again on same pair
        else:
            # Case 1: already paired → auto select pair
            pair_idx = self.get_pair(idx, t_current, t_prev)

            if pair_idx is not None:
                self.selected_points = [idx, pair_idx]

            else:
                # Case 2: first selection
                if len(self.selected_points) == 0:
                    self.selected_points = [idx]

                elif len(self.selected_points) == 1:
                    first_idx = self.selected_points[0]

                    # must be different time
                    if self.time[first_idx] != t_clicked:
                        self.selected_points.append(idx)
                    else:
                        # invalid → replace selection
                        self.selected_points = [idx]

                else:
                    # already 2 selected → replace
                    self.selected_points = [idx]

        self.update_info()
        self.plot()

    # ---------------------------
    def merge(self):
        if len(self.selected_points) != 2:
            return

        i1, i2 = self.selected_points

        if self.time[i1] == self.time[i2]:
            return  # safety

        label1 = self.label[i1]
        label2 = self.label[i2]

        self.label[self.label == label2] = label1

        self.selected_points = []
        self.plot()

    # ---------------------------
    def split(self):
        if len(self.selected_points) != 2:
            return

        i1, i2 = self.selected_points

        # choose later time as split root
        if self.time[i1] > self.time[i2]:
            split_idx = i1
        else:
            split_idx = i2

        new_label = np.max(self.label) + 1

        mask = (self.label == self.label[split_idx]) & (self.time >= self.time[split_idx])
        self.label[mask] = new_label

        self.selected_points = []
        self.plot()

    # ---------------------------
    def next_time(self):
        self.current_time_idx += 1
        self.current_time_idx = min(self.current_time_idx, len(self.ds_swh.time) - 1)
        self.plot()

    def prev_time(self):
        self.current_time_idx -= 1
        self.current_time_idx = max(self.current_time_idx, 0)
        self.plot()

    # ---------------------------
    def get_prev_time(self, t):
        idx = np.where(self.ds_swh.time.values == t)[0][0]
        if idx > 0:
            return self.ds_swh.time.values[idx - 1]
        return 0

    # ---------------------------
    def update_info(self):
        if len(self.selected_points) == 0:
            self.info.setText("No selection")
        elif len(self.selected_points) == 1:
            self.info.setText(f"Selected: {self.selected_points[0]} (waiting pair)")
        else:
            self.info.setText(f"Selected pair: {self.selected_points}")

    # ---------------------------
    def save(self):
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save file",
            self.storm_file.replace('_checked.nc', '.nc').replace('.nc', '_checked.nc'),
            "*.nc",
        )
        if not fname:
            return

        ds_out = xr.Dataset(
            {
                "longitude": (["x"], self.lon),
                "latitude": (["x"], self.lat),
                "time": (["x"], self.time),
                "numStorm": (["x"], self.label),
            }
        )

        ds_out.attrs["paused_at_time_index"] = int(self.current_time_idx)

        ds_out.to_netcdf(fname)


# ===============================
# RUN APPLICATION
# ===============================

if __name__ == "__main__":
    path = (
        '/home/mdecarlo/Documents/perso/ifremer/Sauvegarde_MDC/Documents/Documents/PROJETS/TBH_Tempetes_bdd_historique/'
    )
    swh_file = path + 'LOPS_WW3-GLOB-30M_199301.nc'

    path2 = '/home/mdecarlo/Documents/Projets/CCI_SeaState/DATA_V4'
    # storm_file = path + 'Model/New_detect/WW3/tracking/transi/tracking_1993_01_xr.nc'
    storm_file = os.path.join(path2, 'WW3_tracking_storm_1993_01.nc')

    app = QApplication(sys.argv)
    window = StormEditor(swh_file, storm_file)
    window.show()
    sys.exit(app.exec_())
