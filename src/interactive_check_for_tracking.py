import sys
import os
import numpy as np
import xarray as xr

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QFileDialog,
    QShortcut,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.colors as mcolors

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


def relabel_sequential(arr, vmin=0):
    ind = np.nonzero(arr >= vmin)[0]
    labels = np.copy(arr)
    _, lab = np.unique(labels[ind], return_inverse=True)
    labels[ind] = lab + vmin
    return labels


# ===============================
# GUI APPLICATION
# ===============================


class StormEditor(QMainWindow):

    def __init__(self, path, path2, year, month, ancillary_variable='hs'):
        super().__init__()

        self.year = year
        self.month = month
        self.path2 = path2
        self.path = path
        self.var_aux = ancillary_variable
        self.read_new_month()
        self.init_ui()
        self.plot()

    def read_new_month(self):
        self.swh_file = self.path + f'CCI_WW3-GLOB-30M_{self.year}{self.month:02d}_hs.nc'
        self.storm_file = os.path.join(self.path2, f'WW3_tracking_storm_{self.year}_{self.month:02d}.nc')

        yearm1 = self.year + (self.month - 1 - 1) // 12
        monthm1 = (self.month - 1 - 1) % 12 + 1
        storm_file_before = os.path.join(self.path2, f'WW3_tracking_storm_{yearm1}_{monthm1:02d}_checked.nc')

        # Load datasets
        self.ds_swh = xr.open_dataset(self.swh_file)
        self.ds_storm = xr.open_dataset(self.storm_file)
        if os.path.isfile(storm_file_before):
            self.ds_storm_old = xr.open_dataset(storm_file_before)
            self.cnt_min = self.ds_storm_old["numStorm"].max().values + 1
        else:
            self.ds_storm_old = None
            self.cnt_min = 0
        self.time_steps = self.ds_swh.time.values.copy()
        self.current_time_idx = 0

        # Copy data to mutable arrays
        self.lon = self.ds_storm["longitude"].values.copy()
        self.lat = self.ds_storm["latitude"].values.copy()
        self.time = self.ds_storm["time"].values.copy()
        self.label = self.ds_storm["numStorm"].values.copy() + self.cnt_min
        self.hs = self.ds_storm["hs_max"].values.copy()

        if self.ds_storm_old is not None:
            self.add_last_step_fromprev()

        self.selected_points = []

    # --------------------------
    def add_last_step_fromprev(self):
        lon_old = self.ds_storm_old["longitude"].values.copy()
        lat_old = self.ds_storm_old["latitude"].values.copy()
        time_old = self.ds_storm_old["time"].values.copy()
        label_old = self.ds_storm_old["numStorm"].values.copy()
        hs_old = self.ds_storm_old["hs_max"].values.copy()
        t_val = self.time_steps[0] - np.timedelta64(1, 'h')
        mask = time_old == t_val
        idxs0 = np.nonzero(mask)[0]
        labels_old_selected = label_old[idxs0]

        idxs_old = []
        for l in np.unique(labels_old_selected):
            idxs_old.extend(np.nonzero(label_old == l)[0])

        self.lon = np.concat([lon_old[idxs_old], self.lon])
        self.lat = np.concat([lat_old[idxs_old], self.lat])
        self.time = np.concat([time_old[idxs_old], self.time])
        self.label = np.concat([label_old[idxs_old], self.label])
        self.hs = np.concat([hs_old[idxs_old], self.hs])

    # ---------------------------
    def init_ui(self):
        self.setWindowTitle("Storm trajectory editor")

        self.figure = plt.figure(figsize=(24, 12))

        self.ax_main = self.figure.add_subplot(211)
        self.ax_aux = self.figure.add_subplot(212)

        self.canvas = FigureCanvas(self.figure)

        self.canvas.mpl_connect("button_press_event", self.on_click)

        btn_last = QPushButton("Last time")
        btn_next = QPushButton("Next time")
        btn_prev = QPushButton("Prev time")
        btn_merge = QPushButton("Merge")
        btn_split = QPushButton("Split")
        btn_save = QPushButton("Save")

        btn_last.clicked.connect(self.last_time)
        btn_next.clicked.connect(self.next_time)
        btn_prev.clicked.connect(self.prev_time)
        btn_merge.clicked.connect(self.merge)
        btn_split.clicked.connect(self.split)
        btn_save.clicked.connect(self.save)

        self.info = QLabel("Select points...")

        layout_maps = QVBoxLayout()
        layout_maps.addWidget(self.canvas)
        layout_maps.addWidget(self.info)
        layout_btn = QVBoxLayout()
        layout_btn.addWidget(btn_merge)
        layout_btn.addWidget(btn_split)
        layout_btn.addWidget(btn_prev)
        layout_btn.addWidget(btn_next)
        layout_btn.addWidget(btn_last)
        layout_btn.addWidget(btn_save)

        container = QWidget()
        # container.setLayout(layout)
        self.setCentralWidget(container)
        main_layout = QHBoxLayout(container)
        main_layout.addLayout(layout_maps)
        main_layout.addLayout(layout_btn)

        btn_last.setFocusPolicy(Qt.NoFocus)
        btn_next.setFocusPolicy(Qt.NoFocus)
        btn_prev.setFocusPolicy(Qt.NoFocus)
        btn_merge.setFocusPolicy(Qt.NoFocus)
        btn_split.setFocusPolicy(Qt.NoFocus)
        btn_save.setFocusPolicy(Qt.NoFocus)

        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_time)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_time)
        QShortcut(QKeySequence(Qt.Key_M), self, self.merge)
        QShortcut(QKeySequence(Qt.Key_S), self, self.split)

    # ---------------------------
    def plot(self):
        self.figure.clear()
        self.ax_main = self.figure.add_subplot(211)
        self.ax_aux = self.figure.add_subplot(212)

        ax1 = self.ax_main
        ax2 = self.ax_aux

        t_val = self.time_steps[self.current_time_idx]

        # SWH field
        swh = self.ds_swh["hs"].isel(time=self.current_time_idx)
        aux = self.ds_swh[self.var_aux].isel(time=self.current_time_idx)

        ind_lon = np.nonzero(self.ds_swh["longitude"].values < 0)[0]
        swh_bis = np.concat([swh.values, swh.isel(longitude=ind_lon).values], axis=1)
        aux_bis = np.concat([aux.values, aux.isel(longitude=ind_lon).values], axis=1)

        lon_bis = np.concat(
            [self.ds_swh["longitude"].values, self.ds_swh["longitude"].isel(longitude=ind_lon).values + 360]
        )

        # --- Colormap for hs -----
        colors1 = plt.cm.viridis(np.linspace(0.0, 1, 128))
        colors2 = plt.cm.plasma_r(np.linspace(0, 1, 128))
        # combine them and build a new colormap
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=9, vmax=16)
        # # Create a ListedColormap from the merged colors
        im = ax1.pcolormesh(lon_bis, self.ds_swh["latitude"], swh_bis, cmap=mymap, norm=norm)
        plt.colorbar(im, ax=ax1)

        # combine them and build a new colormap
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=9, vmax=16)  # TO MODIFY VALUES

        im = ax2.pcolormesh(lon_bis, self.ds_swh["latitude"], aux_bis, cmap=mymap, norm=norm)
        plt.colorbar(im, ax=ax2)

        # Storm points at T and T-1
        mask_t = self.time == t_val
        mask_tm1 = self.time == self.get_prev_time(t_val)

        self.plot_points(ax1, mask_tm1, 'v', "blue", 'T-1')
        self.plot_points(ax1, mask_t, '^', "red", 'T')

        self.plot_points(ax2, mask_tm1, 'v', "blue", 'T-1')
        self.plot_points(ax2, mask_t, '^', "red", 'T')

        ax1.set_xlim((-180, 360))
        ax1.set_title(t_val)
        ax1.legend(loc='upper left')

        ax2.set_xlim((-180, 360))
        ax2.set_title(t_val)
        ax2.legend(loc='upper left')
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
            hs_traj = self.hs[idx].copy()
            if np.std(lon_traj) > 50:
                lon_traj[lon_traj < 0] = lon_traj[lon_traj < 0] + 360
            if np.max(hs_traj) >= 10:
                lw = 2
                c = 'r'
                ax.plot(lon_traj, self.lat[idx], color=c, linewidth=lw, zorder=9)

            lw = 1
            c = default_cycler[lab % 10]
            ax.plot(lon_traj, self.lat[idx], color=c, linewidth=lw, zorder=10)  # , transform=ccrs.PlateCarree()

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

        if len(self.selected_points) == 1:
            valid_indices = np.setdiff1d(valid_indices, self.selected_points)

        if len(valid_indices) == 0:
            return

        idx_local = find_nearest_point(click_lon, click_lat, self.lon[valid_indices], self.lat[valid_indices], tol=6)

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
    def last_time(self):
        self.current_time_idx = len(self.ds_swh.time) - 1
        self.plot()

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
        else:
            return t - np.timedelta64(1, 'h')
        # return 0

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
        print(fname)
        self.label = relabel_sequential(self.label, vmin=self.cnt_min)
        ind = np.nonzero(self.time >= self.time_steps[0])[0]
        ds_out = self.ds_storm
        ds_out.numStorm.values = self.label[ind]
        print(np.min(ds_out.numStorm.values))

        ds_out.attrs["paused_at_time_index"] = int(self.current_time_idx)

        ds_out.to_netcdf(fname)

        self.year = self.year + (self.month + 1 - 1) // 12
        self.month = (self.month + 1 - 1) % 12 + 1
        self.ds_swh.close()
        self.ds_storm.close()
        self.read_new_month()
        self.plot()


# ===============================
# RUN APPLICATION
# ===============================

if __name__ == "__main__":
    # path = (
    #     '/home/mdecarlo/Documents/perso/ifremer/Sauvegarde_MDC/Documents/Documents/PROJETS/TBH_Tempetes_bdd_historique/'
    # )
    year = 1996
    month = 1

    path = '/home/mdecarlo/Documents/Projets/CCI_SeaState/DATA_MODEL/WW3/'
    path2 = '/home/mdecarlo/Documents/Projets/CCI_SeaState/DATA_MODEL/tracking/'
    # storm_file = path + 'Model/New_detect/WW3/tracking/transi/tracking_1993_01_xr.nc'
    # while year < 2026:
    app = QApplication(sys.argv)
    window = StormEditor(path, path2, year, month)  # swh_file, storm_file, storm_file_before)
    window.show()
    sys.exit(app.exec_())
    # year = year + (month + 1 - 1) // 12
    # month = (month + 1 - 1) % 12 + 1
    # print(year)
