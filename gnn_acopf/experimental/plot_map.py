from pathlib import Path
from gnn_acopf.utils.power_net import PowerNetwork
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from owslib.wmts import WebMapTileService
import cartopy.io.img_tiles as cimgt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import patches
from matplotlib.gridspec import GridSpec

VOLTAGE_CMAP = "viridis"

class MapPlotter:
    def __init__(self, power_network, plot_CRS, geodetic_CRS):
        self.power_network = power_network
        self.plot_CRS = plot_CRS
        self.geodetic_CRS = geodetic_CRS
        # self.voltage_cmap = sns.cubehelix_palette(as_cmap=True)
        self.voltage_sm = ScalarMappable(norm=Normalize(
            vmin=0,
            vmax=500),
            cmap=VOLTAGE_CMAP)

    def plot_terrain(self, ax, detailed):
        stamen_terrain = cimgt.Stamen('terrain-background')
        # Add the Stamen data at zoom level 8.
        zoom_level = 10 if detailed else 8
        ax.add_image(stamen_terrain, zoom_level)

    def draw_borders(self, ax, detailed):
        ax.add_feature(cartopy.feature.STATES)

    def draw_coastline(self, ax, detailed):
        ax.add_feature(cartopy.feature.COASTLINE)

    def set_boundary(self, ax, x0, x1, y0, y1):
        x0, y0 = self.plot_CRS.transform_point(x0, y0, self.geodetic_CRS)
        x1, y1 = self.plot_CRS.transform_point(x1, y1, self.geodetic_CRS)
        ax.set_xlim((x0, x1))
        ax.set_ylim((y0, y1))
        return ax

    def plot_background(self, ax, terrain_background, x0, x1, y0, y1, detailed):
        background_types = {
            "terrain": self.plot_terrain,
            "coastline": self.draw_coastline,
            "satellite": self.plot_satellite,
            "borders": self.draw_borders
        }
        # ysize = 8
        # xsize = 2 * ysize * (x1 - x0) / (y1 - y0)
        # fig = plt.figure(figsize=(xsize, ysize), dpi=400)
        ax = self.set_boundary(ax, x0, x1, y0, y1)
        if not isinstance(terrain_background, list):
            terrain_background = [terrain_background]
        for single_background in terrain_background:
            draw_func = background_types[single_background]
            if draw_func is not None:
                draw_func(ax, detailed)
        return ax

    def plot_satellite(self, ax, detailed):
        # URL of NASA GIBS
        URL = 'http://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi'
        wmts = WebMapTileService(URL)

        # Layers for MODIS true color and snow RGB
        layer = 'MODIS_Terra_SurfaceReflectance_Bands143'

        date_str = '2018-06-02'
        # date_str = '2018-05-27'

        # 4.6, 11.0, 43.1, 47.4

        ax.add_wmts(wmts, layer, wmts_kwargs={'time': date_str})
        return ax

    def create_basemap(self, ax, terrain_background, boundaries=None, detailed=False):
        if boundaries is None:
            all_longs = [c[0] for c in self.power_network.coordinates.values()]
            all_lats = [c[1] for c in self.power_network.coordinates.values()]
            boundary = 0.25
            x0 = min(all_lats) - boundary
            x1 = max(all_lats) + boundary
            y0 = min(all_longs) - boundary
            y1 = max(all_longs) + boundary
        else:
            x0, x1, y0, y1 = boundaries
        ax = self.plot_background(ax, terrain_background, x0, x1, y0, y1, detailed=detailed)
        return ax

    @property
    def pn(self):
        return self.power_network

    def voltage_to_color(self, v):
        return self.voltage_sm.to_rgba(v)

    def plot_branches_and_nodes(self, ax):
        branches = sorted(self.pn.case_dict["branch"])
        lines = []
        colours = []
        for branch in branches:
            from_node = self.pn.case_dict["bus"][str(self.pn.case_dict["branch"][branch]["f_bus"])]
            to_node = self.pn.case_dict["bus"][str(self.pn.case_dict["branch"][branch]["t_bus"])]
            from_coord = self.pn.get_coordinates(from_node["name"])
            to_coord = self.pn.get_coordinates(to_node["name"])
            line = [self.plot_CRS.transform_point(c[1], c[0], self.geodetic_CRS) for c in [from_coord, to_coord]]
            lines.append(line)
            from_voltage, to_voltage = from_node["base_kv"], to_node["base_kv"]
            colours.append(self.voltage_to_color((from_voltage + to_voltage) / 2))
        busses = sorted(self.power_network.case_dict["bus"])

        line_segments = LineCollection(lines,
                                       colors=colours,
                                       # cmap="plasma"
                                       )
        ax.add_collection(line_segments)
        coords = [
            self.pn.get_coordinates(self.pn.case_dict["bus"][b]["name"]) for b in busses

        ]
        colours = [
            np.log(self.pn.case_dict["bus"][b]["base_kv"]) for b in busses
        ]
        coords = [self.plot_CRS.transform_point(c[1], c[0], self.geodetic_CRS) for c in coords]
        coords = np.stack(coords)
        longs, lats = coords.T

        pts = ax.scatter(longs, lats, c=colours, transform=self.plot_CRS, cmap=VOLTAGE_CMAP,
                         s=4, zorder=3)
        return ax

    def add_submap(self, original_ax, new_ax, terrain_background, x0, x1, y0, y1, title, title_loc="left"):
        new_ax = self.create_basemap(new_ax, terrain_background, [y0, y1, x0, x1], detailed=True)
        new_ax = self.plot_branches_and_nodes(new_ax)
        new_ax = self.set_boundary(new_ax, y0, y1, x0, x1)
        (x0, y0), (x1, y1) = [self.plot_CRS.transform_point(c[1], c[0], self.geodetic_CRS) for c in [(x0, y0), (x1, y1)]]
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=3, edgecolor='k', facecolor='none', transform=self.plot_CRS,
                                 zorder=4)
        original_ax.add_patch(rect)
        if title_loc == "left":
            new_ax.text(-0.2, 0.5, title, transform=new_ax.transAxes, verticalalignment='center', rotation=90,
                        fontsize=18)
        else:
            new_ax.text(0.5, 1.1, title, transform=new_ax.transAxes, verticalalignment='center', horizontalalignment="center",
                        fontsize=18)
        return new_ax

    def set_size(self, w, h, axes):
        """ w, h: width, height in inches """
        for ax in axes:
            l = ax.figure.subplotpars.left
            r = ax.figure.subplotpars.right
            t = ax.figure.subplotpars.top
            b = ax.figure.subplotpars.bottom
            figw = float(w) / (r - l)
            figh = float(h) / (t - b)
            ax.figure.set_size_inches(figw, figh)

    def plot_ACTIVSg2000_presentation(self, terrain_background):
        fig = plt.figure(figsize=(16, 10), dpi=400)
        gs = GridSpec(3, 3, width_ratios=[3, 1, 0.25], height_ratios=[1, 1, 1])
        gs.update(wspace=0., hspace=0.)
        texas_ax = fig.add_subplot(gs[:3, 0], projection=self.plot_CRS)
        texas_ax = self.create_basemap(texas_ax, terrain_background)
        texas_ax = self.plot_branches_and_nodes(texas_ax)
        # cax = fig.add_axes([0.25, 0, 0.5, 0.05])
        cax = fig.add_subplot(gs[:3, -1])
        fig.colorbar(self.voltage_sm, cax=cax, orientation='vertical')
        cax.text(0.5, 0.5, "Voltage (kV)", transform=cax.transAxes,
                 verticalalignment="center", horizontalalignment="center", fontsize=18,
                 rotation=90)
        cax.tick_params(labelsize=18)

        dallas_ax = fig.add_subplot(gs[0, 1], projection=self.plot_CRS)
        dallas_ax = self.add_submap(original_ax=texas_ax,
                                    new_ax=dallas_ax, terrain_background=terrain_background,
                                    x0=32.05, x1=33.55, y0=-97.75, y1=-96.25, title="Dallas")
        houston_ax = fig.add_subplot(gs[1, 1], projection=self.plot_CRS)
        # 29.762778, -95.383056
        houston_ax = self.add_submap(original_ax=texas_ax,
                                    new_ax=houston_ax, terrain_background=terrain_background,
                                    x0=29, x1=30.5, y0=-96, y1=-94.5, title="Houston")

        san_antonio_ax = fig.add_subplot(gs[2, 1], projection=self.plot_CRS)
        # 29.762778, -95.383056
        san_antonio_ax = self.add_submap(original_ax=texas_ax,
                                    new_ax=san_antonio_ax, terrain_background=terrain_background,
                                    x0=29.2, x1=30.7, y0=-98.9, y1=-97.4, title="San Antonio/Austin")

        return fig

    def plot_ACTIVSg2000_A4(self, terrain_background):
        fig = plt.figure(figsize=(10, 15), dpi=400)
        gs = GridSpec(4, 3, width_ratios=[1, 1, 1], height_ratios=[3, 0.1, 0.25, 1])
        gs.update(wspace=0., hspace=0.)
        texas_ax = fig.add_subplot(gs[0, :], projection=self.plot_CRS)
        texas_ax = self.create_basemap(texas_ax, terrain_background)
        texas_ax = self.plot_branches_and_nodes(texas_ax)
        # cax = fig.add_axes([0.25, 0, 0.5, 0.05])
        cax = fig.add_subplot(gs[1, :])
        fig.colorbar(self.voltage_sm, cax=cax, orientation='horizontal')
        cax.text(0.5, 0.5, "Voltage (kV)", transform=cax.transAxes, verticalalignment="center", horizontalalignment="center", fontsize=18)

        dallas_ax = fig.add_subplot(gs[3, 1], projection=self.plot_CRS)
        dallas_ax = self.add_submap(original_ax=texas_ax,
                                    new_ax=dallas_ax, terrain_background=terrain_background,
                                    x0=32.05, x1=33.55, y0=-97.75, y1=-96.25, title="Dallas", title_loc="top")
        houston_ax = fig.add_subplot(gs[3, 2], projection=self.plot_CRS)
        # 29.762778, -95.383056
        houston_ax = self.add_submap(original_ax=texas_ax,
                                    new_ax=houston_ax, terrain_background=terrain_background,
                                    x0=29, x1=30.5, y0=-96, y1=-94.5, title="Houston", title_loc="top")

        san_antonio_ax = fig.add_subplot(gs[3, 0], projection=self.plot_CRS)
        # 29.762778, -95.383056
        san_antonio_ax = self.add_submap(original_ax=texas_ax,
                                    new_ax=san_antonio_ax, terrain_background=terrain_background,
                                    x0=29.2, x1=30.7, y0=-98.9, y1=-97.4, title="San Antonio/Austin", title_loc="top")

        return fig


    def plot_map(self, terrain_background):
        fig = plt.figure(dpi=400)
        ax = fig.add_axes([0, 0, 1, 1], projection=self.plot_CRS)
        ax = self.create_basemap(ax, terrain_background)
        ax = self.plot_branches_and_nodes(ax)
        cax = fig.add_axes([0.25, 0, 0.5, 0.05])
        fig.colorbar(self.voltage_sm, cax=cax, orientation='horizontal')
        return fig

    def plot_ACTIVSg200(self, terrain_background):
        fig = plt.figure(dpi=400)
        ax = fig.add_axes([0, 0, 1, 1], projection=self.plot_CRS)
        ax = self.create_basemap(ax, terrain_background, detailed=True)
        ax = self.plot_branches_and_nodes(ax)
        cax = fig.add_axes([0.25, 0, 0.5, 0.05])
        fig.colorbar(self.voltage_sm, cax=cax, orientation='horizontal')
        return fig


def main_ACTIVSg2k_presentation():
    datapath = Path("../../data")
    casename = "ACTIVSg2000"
    area_name = "area"
    pgmin_to_zero = True
    pn = PowerNetwork.from_pickle(datapath / f"case_{casename}.pickle", area_name=area_name,
                                  pgmin_to_zero=pgmin_to_zero)
    pn.load_scenarios_file(datapath / f"scenarios_{casename}.m")
    pn.load_coordinates(datapath / f"{casename}_GIC_data.gic")
    mp = MapPlotter(pn, ccrs.Mercator(), ccrs.Geodetic())
    # mp.plot_map()
    #fig = mp.plot_ACTIVSg2000_A4(terrain_background=[
    #    "terrain",
    #    # "satellite",
    #    "borders",
    #    "coastline"
    #])

    fig = mp.plot_ACTIVSg2000_presentation(terrain_background=[
        "terrain",
        # "satellite",
        "borders",
        "coastline"
    ])

    plt.show()
    fig.savefig(f"./{casename}_plotted_presentation.png",
                bbox_inches="tight", pad_inches=0, transparent=True)

    fig = mp.plot_ACTIVSg2000_presentation(terrain_background=[
        # "terrain",
        "satellite",
        "borders",
        "coastline"
    ])
    fig.savefig(f"./{casename}_satellite_presentation.png",
                bbox_inches="tight", pad_inches=0, transparent=True)

def main_ACTIVSg2k():
    datapath = Path("../../data")
    casename = "ACTIVSg2000"
    area_name = "area"
    pgmin_to_zero = True
    pn = PowerNetwork.from_pickle(datapath / f"case_{casename}.pickle", area_name=area_name,
                                  pgmin_to_zero=pgmin_to_zero)
    pn.load_scenarios_file(datapath / f"scenarios_{casename}.m")
    pn.load_coordinates(datapath / f"{casename}_GIC_data.gic")
    mp = MapPlotter(pn, ccrs.Mercator(), ccrs.Geodetic())
    # mp.plot_map()
    fig = mp.plot_ACTIVSg2000_A4(terrain_background=[
        "terrain",
        # "satellite",
        "borders",
        "coastline"
    ])

    plt.show()
    fig.savefig(f"./{casename}_plotted_A4.pdf",
                bbox_inches="tight", pad_inches=0)


def main_ACTIVSg200():
    datapath = Path("../../data")
    casename = "ACTIVSg200"
    area_name = "zone"
    pgmin_to_zero = False
    pn = PowerNetwork.from_pickle(datapath / f"case_{casename}.pickle", area_name=area_name,
                                  pgmin_to_zero=pgmin_to_zero)
    pn.load_scenarios_file(datapath / f"scenarios_{casename}.m")
    pn.load_coordinates(datapath / f"{casename}_GIC_data.gic")
    mp = MapPlotter(pn, ccrs.Mercator(), ccrs.Geodetic())
    # mp.plot_map()
    fig = mp.plot_ACTIVSg200(terrain_background=[
        "terrain",
        # "satellite",
        "borders",
        "coastline"
    ])
    plt.show()
    fig.savefig(f"./{casename}_plotted.pdf",
                bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main_ACTIVSg2k_presentation()
