import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
# NOTE: need to comment this line if do not have GUI.
from mayavi import mlab

from collections import namedtuple
from itertools import accumulate
from typing import Optional, Tuple
from matplotlib.ticker import AutoMinorLocator

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

BLUE = (94/255, 129/255, 160/255)
GREEN = (163/255, 190/255, 128/255)
RED = (191/255, 97/255, 106/255)
PURPLE = (180/255, 142/255, 160/255)
OPACITY = 1.0


def show_flows(pc1, pc2, flow, inverse=False):
    if type(pc1) is not np.ndarray:
        pc1 = pc1.cpu().numpy()
        pc2 = pc2.cpu().numpy()
        flow = flow.detach().cpu().numpy()

    pc1_deform = pc1 + flow

    OPACITY = 1.0
    fig = mlab.figure(size=(800, 600), bgcolor=(1,1,1))
    mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], color=GREEN, figure=fig, opacity=OPACITY, scale_factor=0.07, resolution=25)
    mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], color=BLUE, figure=fig, opacity=1.0, scale_factor=0.07, resolution=25)

    mlab.points3d(pc1_deform[:,0], pc1_deform[:,1], pc1_deform[:,2], color=RED, figure=fig, opacity=OPACITY, scale_factor=0.07, resolution=25)
    obj = mlab.quiver3d(pc1[:,0], pc1[:,1], pc1[:,2], flow[:,0], flow[:,1], flow[:,2], mode='arrow', colormap='spring', scale_factor=1.0, line_width=0.001, resolution=25, opacity=0.3)

    obj.glyph.glyph_source.glyph_source.tip_length = 0.05
    obj.glyph.glyph_source.glyph_source.tip_radius = 0.02
    obj.glyph.glyph_source.glyph_source.shaft_radius = 0.005
    obj.glyph.glyph_source.glyph_source.shaft_resolution = 95
    
    mlab.show()
    
    pc1_o3d = o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    pc1_o3d.paint_uniform_color(BLUE)
    pc1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc1_o3d.orient_normals_to_align_with_direction()

    pc2_o3d = o3d.geometry.PointCloud()
    pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
    pc2_o3d.paint_uniform_color(GREEN)
    pc2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc2_o3d.orient_normals_to_align_with_direction()    

    pc1_def_o3d = o3d.geometry.PointCloud()
    pc1_def_o3d.points = o3d.utility.Vector3dVector(pc1_deform)
    pc1_def_o3d.paint_uniform_color(RED)
    pc1_def_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc1_def_o3d.orient_normals_to_align_with_direction()
    o3d.visualization.draw_geometries([pc1_o3d, pc2_o3d, pc1_def_o3d])


def make_colorwheel(transitions: tuple=DEFAULT_TRANSITIONS) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array, ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0])
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def flow_to_rgb(
    flow: np.ndarray,
    flow_max_radius: Optional[float]=None,
    background: Optional[str]="bright",
) -> np.ndarray:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(f"background should be one the following: {valid_backgrounds}, not {background}.")
    wheel = make_colorwheel()
    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))
    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))
    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int)] * (1 - angle_fractional) + wheel[angle_ceil.astype(np.int)] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        'ColorizationArgs', ['move_hue_valid_radius', 'move_hue_oversized_radius', 'invalid_color']
    )
    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)
    def move_hue_on_S_axis(hues, factors):
        return 255. - np.expand_dims(factors, -1) * (255. - hues)
    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float)
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float))
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    return colors.astype(np.uint8)


def calibration_pattern(
    pixel_size: int=151,
    flow_max_radius: float=1,
    **flow_to_rgb_args
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a calibration pattern to add as a legend to the scene flow plots.
    Args:
        pixel_size: Radius of the square test pattern.
        flow_max_radius: The maximum radius value represented by the calibration pattern.
        flow_to_rgb_args: kwargs passed to the flow_to_rgb function.
    Returns:
        calibration_img: The RGB image representation of the calibration pattern.
        calibration_flow: The flow represented in the calibration_pattern.
    """
    half_width = pixel_size // 2
    y_grid, x_grid = np.mgrid[:pixel_size, :pixel_size]
    u = flow_max_radius * (x_grid / half_width - 1)
    v = flow_max_radius * (y_grid / half_width - 1)
    flow = np.zeros((pixel_size, pixel_size, 2))
    flow[..., 0] = u
    flow[..., 1] = v
    flow_to_rgb_args["flow_max_radius"] = flow_max_radius
    img = flow_to_rgb(flow, **flow_to_rgb_args)
    return img, flow


def attach_calibration_pattern(ax, **calibration_pattern_kwargs):
    """Attach a calibration pattern to axes.
    This function uses calibration_pattern to generate a figure.
    Args:
        calibration_pattern_kwargs: kwargs, optional
            Parameters to be given to the calibration_pattern function.
    Returns:
        image_axes: matplotlib.AxesImage
            See matplotlib.imshow documentation
            Useful for changing the image dynamically
        circle_artist: matplotlib.artist
            See matplotlib.circle documentation
            Useful for removing the circle from the figure
    """
    
    pattern, flow = calibration_pattern(**calibration_pattern_kwargs)
    flow_max_radius = calibration_pattern_kwargs.get("flow_max_radius", 1)
    extent = (-flow_max_radius, flow_max_radius) * 2
    image = ax.imshow(pattern, extent=extent)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    for spine in ("bottom", "left"):
        ax.spines[spine].set_position("zero")
        ax.spines[spine].set_linewidth(1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    attach_coord(ax, flow, extent=extent)
    circle = plt.Circle((0, 0), flow_max_radius, fill=False, lw=1)
    ax.add_artist(circle)
    
    return image, circle


def attach_coord(ax, flow, extent=None):
    """Attach the flow value to the coordinate tooltip.
    It allows you to see on the same figure, the RGB value of the pixel and the underlying value of the flow.
    Shows cartesian and polar coordinates.
    Args:
        ax: matplotlib.axes
            The axes the arrows should be plotted on.
        flow: numpy.ndarray
            scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
        extent: sequence_like, optional
            Use this parameters in combination with matplotlib.imshow to resize the RGB plot.
            See matplotlib.imshow extent parameter.
            See attach_calibration_pattern
    """
    
    height, width, _ = flow.shape
    base_format = ax.format_coord
    if extent is not None:
        left, right, bottom, top = extent
        x_ratio = width / (right - left)
        y_ratio = height / (top - bottom)
        
    def new_format_coord(x, y):
        if extent is None:
            int_x = int(x + 0.5)
            int_y = int(y + 0.5)
        else:
            int_x = int((x - left) * x_ratio)
            int_y = int((y - bottom) * y_ratio)
        if 0 <= int_x < width and 0 <= int_y < height:
            format_string = "Coord: x={}, y={} / Flow: ".format(int_x, int_y)
            u, v = flow[int_y, int_x, :]
            if np.isnan(u) or np.isnan(v):
                format_string += "invalid"
            else:
                complex_flow = u - 1j * v
                r, h = np.abs(complex_flow), np.angle(complex_flow, deg=True)
                format_string += ("u={:.2f}, v={:.2f} (cartesian) ρ={:.2f}, θ={:.2f}° (polar)"
                                  .format(u, v, r, h))
            return format_string
        else:
            return base_format(x, y)
        
    ax.format_coord = new_format_coord


def custom_draw_geometry_with_key_callback(pcds):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([76/255, 86/255, 106/255])
        # opt.background_color = np.asarray([7/255, 54/255, 66/255])
        return False
    
    # def load_render_option(vis):
    #     vis.get_render_option().load_from_json(
    #         "../../TestData/renderoption.json")
    #     return False
    
    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False
    
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
    
    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    # key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback)
    