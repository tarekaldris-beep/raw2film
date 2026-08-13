"""
Image processing effects.
"""

import math
from collections.abc import Callable
from functools import lru_cache

import cv2 as cv
import lensfunpy
import numpy as np
from lensfunpy import util as lensfunpy_util
from numba import njit, prange
from scipy import ndimage
from spectral_film_lut.config import DEFAULT_DTYPE
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.grain_generation import generate_grain
from spectral_film_lut.xy_lut import apply_2d_lut

from raw2film.raw_conversion import CANVAS_MODES


def lens_correction(
    rgb: np.ndarray, metadata: dict, cam: None | str, lens: None | str
) -> np.ndarray:
    """Apply lens correction using lensfunpy."""
    # noinspection PyUnresolvedReferences
    rgb = rgb.astype(np.float64)
    if lens is None or cam is None:
        return rgb
    try:
        focal_length = metadata["EXIF:FocalLength"]
        aperture = float(metadata["EXIF:FNumber"])
    except (KeyError, ValueError):
        return rgb
    height, width = rgb.shape[0], rgb.shape[1]
    # noinspection PyUnresolvedReferences
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, pixel_format=np.float64)
    undistorted_cords = mod.apply_geometry_distortion()
    rgb = np.clip(lensfunpy_util.remap(rgb, undistorted_cords), a_min=0, a_max=None)
    mod.apply_color_modification(rgb)

    return rgb


def rotate(rgb: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate an image by a specified angle in degrees."""
    if degrees:
        input_height, input_width = rgb.shape[:2]
        image_center = tuple(np.array(rgb.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, -degrees, 1.0)
        rgb = cv.warpAffine(rgb, rot_mat, rgb.shape[1::-1], flags=cv.INTER_LINEAR)
        aspect_ratio = input_height / input_width
        angle = math.fabs(degrees) * math.pi / 180

        if aspect_ratio < 1:
            total_height = input_height
            aspect_ratio = 1 / aspect_ratio
            switch = True
        else:
            switch = False
            total_height = input_width

        w = total_height / (aspect_ratio * math.sin(angle) + math.cos(angle))
        h = w * aspect_ratio
        if switch:
            w, h = h, w
        crop_height = int((rgb.shape[0] - h) // 2)
        crop_width = int((rgb.shape[1] - w) // 2)
        rgb = rgb[
            crop_height : rgb.shape[0] - crop_height,
            crop_width : rgb.shape[1] - crop_width,
        ]
    return rgb


def crop_image(rgb: np.ndarray, zoom=1, aspect=1.5, flip=False) -> np.ndarray:
    """Crops rgb data to aspect ratio."""
    x, y, _ = rgb.shape
    if flip:
        aspect = 1 / aspect
    if x > y:
        if x > aspect * y:
            rgb = rgb[
                math.ceil(x / 2 - y * aspect / 2) : math.ceil(x / 2 + y * aspect / 2),
                :,
                :,
            ]
        else:
            rgb = rgb[
                :,
                math.ceil(y / 2 - x / aspect / 2) : math.ceil(y / 2 + x / aspect / 2),
                :,
            ]
    elif y > aspect * x:
        rgb = rgb[
            :, math.ceil(y / 2 - x * aspect / 2) : math.ceil(y / 2 + x * aspect / 2), :
        ]
    else:
        rgb = rgb[
            math.ceil(x / 2 - y / aspect / 2) : math.ceil(x / 2 + y / aspect / 2), :, :
        ]

    if zoom > 1:
        x, y, _ = rgb.shape
        zoom_factor = (zoom - 1) / (2 * zoom)
        x = math.ceil(zoom_factor * x)
        y = math.ceil(zoom_factor * y)
        rgb = rgb[x:-x, y:-y, :]

    return rgb


def mtf_curve(logf, vals):
    """Turn mtf values into an interpolated function on log space."""

    def func(x):
        return np.interp(np.log1p(x), logf, vals, left=1, right=0)

    return func


def compute_kernel_from_function(func, kernel_size_mm, pixel_size_mm):
    """Computes a convolution kernel from a mtf function."""
    kernel_size = round(kernel_size_mm / pixel_size_mm)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Frequency grid
    fx = np.fft.fftfreq(kernel_size, d=pixel_size_mm)
    fy = np.fft.fftfreq(kernel_size, d=pixel_size_mm)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX**2 + FY**2)  # radial frequency magnitude

    # Apply transfer function in frequency domain
    H = func(f)

    # Get spatial kernel by inverse FFT
    kernel = np.fft.ifft2(H)
    kernel = np.fft.fftshift(np.abs(kernel))  # center it
    kernel /= np.sum(kernel)

    return kernel


def convolve_2d(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if len(kernel.shape) == 2:
        rgb = cv.filter2D(rgb, -1, kernel)
    elif rgb.shape[-1] == 1:
        rgb = cv.filter2D(rgb, -1, kernel[..., 0])
    elif len(kernel.shape) == 3:
        for c in range(kernel.shape[-1]):
            rgb[..., c] = cv.filter2D(rgb[..., c], -1, kernel[..., c])
    if len(rgb.shape) == 2:
        rgb = rgb[..., np.newaxis]
    return rgb


def mtf_kernel_layer(logf, vals, scale):
    mtf_func = mtf_curve(np.asarray(logf), np.asarray(vals))
    kernel = compute_kernel_from_function(mtf_func, 0.1, 1 / scale)
    return kernel


@lru_cache(maxsize=50)
def mtf_kernel(
    stock: FilmSpectral,
    scale,
    sharpening_strength: float = 0.0,
    sharpening_sigma: float = 1.0,
):
    """Cache a mtf convolution kernel."""
    kernel = np.stack(
        [mtf_kernel_layer(logf, vals, scale) for logf, vals in stock.mtf],
        axis=-1,
        dtype=DEFAULT_DTYPE,
    )

    if sharpening_strength:
        sigma = sharpening_sigma * scale / 50
        unsharp_kernel = ndimage.gaussian_filter(kernel, sigma=sigma)

        kernel += sharpening_strength * (kernel - unsharp_kernel)

    return kernel


def film_sharpness(
    rgb: np.ndarray,
    stock: FilmSpectral,
    scale: float,
    sharpening_strength,
    sharpening_sigma,
):
    """Apply the sharpness and micro-contrast of a film stock ot an image."""
    kernel = mtf_kernel(stock, scale, sharpening_strength, sharpening_sigma)
    return convolve_2d(rgb, kernel)


@njit
def exponential_blur_kernel(size):
    """Compute an exponential blur kernel for halation."""
    radius = size / 2
    size = 2 * math.floor(math.ceil(size) / 2) + 1
    center = math.ceil(size / 2)
    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            dist = (i + 1 - center) ** 2 + (j + 1 - center) ** 2
            if not dist:
                kernel[i, j] = 1
            else:
                kernel[i, j] = (1 / dist) * max((radius - np.sqrt(dist)) / radius, 0)
    kernel /= np.sum(kernel)

    return kernel


def apply_grain(
    rgb: np.ndarray,
    stock: FilmSpectral,
    scale: float,
    grain_size_mm: float = 0.01,
    grain_sigma: float = 0.4,
    bw_grain: bool = False,
    adx: bool = True,
):
    """Applies a grain filter to an image."""
    grain = generate_grain(
        rgb.shape, scale, grain_size_mm, bw_grain, cached=True, grain_sigma=grain_sigma
    )
    grain_factors = stock.grain_transform(rgb, scale, adx=adx, bw_grain=bw_grain)
    grain = grain * grain_factors
    rgb += grain
    return rgb


def bloom_color_factors(color: float) -> tuple[float, float]:
    """Map a warm->white bloom color slider to green and blue glow factors."""
    green_factor = 0.4 + 0.6 * color
    blue_factor = color
    return green_factor, blue_factor


def compute_halation_kernel(
    scale: float,
    halation_size: float = 1.0,
):
    """Compute a normalized exponential blur kernel for halation (per channel)."""
    kernel = np.asarray(
        exponential_blur_kernel(scale / 4 * halation_size), dtype=DEFAULT_DTYPE
    )
    return np.dstack([kernel] * 3)


def halation_color_factors(
    halation_intensity: float,
    halation_red_factor: float = 1.0,
    halation_green_factor: float = 0.4,
    halation_blue_factor: float = 0.0,
    bw: bool = False,
) -> np.ndarray:
    """Per-channel halation glow weights, used to tint and scale the blurred halo."""
    if bw:
        halation_red_factor = halation_green_factor
        halation_blue_factor = halation_green_factor
    return halation_intensity * np.array(
        [halation_red_factor, halation_green_factor, halation_blue_factor],
        dtype=DEFAULT_DTYPE,
    )


def compute_white_luma(lut: np.ndarray) -> float:
    """Compute the developed-film luminance of a neutral white input for a 2D LUT."""
    white_xyz = np.array([[[1.0, 1.0, 1.0]]], dtype=DEFAULT_DTYPE)
    white_out = apply_2d_lut(white_xyz, lut)
    return float(
        0.2126 * white_out[0, 0, 0]
        + 0.7152 * white_out[0, 0, 1]
        + 0.0722 * white_out[0, 0, 2]
    )


def highlight_mask(
    luma: np.ndarray,
    white_luma: float = 1.0,
    threshold: float = 0.5,
    softness: float = 0.4,
) -> np.ndarray:
    """Compute a smooth highlight mask used to gate halation."""
    normalized = luma / max(white_luma, 1e-6)
    x = (normalized - threshold) / max(softness, 1e-6)
    mask = np.clip(x, 0.0, 1.0)
    return mask * mask * (3.0 - 2.0 * mask)


def halation(
    rgb: np.ndarray,
    scale: float,
    halation_size: float = 1.0,
    halation_red_factor: float = 1.0,
    halation_green_factor: float = 0.4,
    halation_blue_factor: float = 0.0,
    halation_intensity: float = 1.0,
    bw: bool = False,
    white_luma: float = 1.0,
    threshold: float = 0.5,
    softness: float = 0.4,
) -> np.ndarray:
    """A halation image processing effect.

    Only highlights emit a soft, warm glow that spills into the surrounding area.
    The glow is normalized by its own luminance, which keeps the tint warm (never
    blue/cyan) while staying soft and organic. Regions without glow are returned
    exactly unchanged.
    """
    kernel = compute_halation_kernel(scale, halation_size)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mask = highlight_mask(luma, white_luma, threshold, softness)

    # Only highlights emit light
    halo = convolve_2d(rgb * mask[..., np.newaxis], kernel)

    # Original-style normalized blend: (image + tinted glow) / (tint + 1)
    factors = halation_color_factors(
        halation_intensity,
        halation_red_factor,
        halation_green_factor,
        halation_blue_factor,
        bw,
    )
    glow = halo * factors
    # Normalize by the glow's luminance (a single scalar) instead of per-channel
    # factors, so highlights get warmer but never shift toward blue/cyan.
    glow_luma = 0.2126 * glow[..., 0] + 0.7152 * glow[..., 1] + 0.0722 * glow[..., 2]
    blended = (rgb + glow) / (1.0 + glow_luma[..., np.newaxis])
    # Where the glow is zero the blend returns the pixel unchanged, so the rest
    # of the image and untouched regions stay exact automatically.
    return blended


def bloom(
    rgb: np.ndarray,
    scale: float,
    bloom_size: float = 1.0,
    bloom_intensity: float = 1.0,
    bloom_threshold: float = 0.8,
    bloom_softness: float = 0.2,
    bloom_red_factor: float = 1.0,
    bloom_green_factor: float = 0.8,
    bloom_blue_factor: float = 0.5,
) -> np.ndarray:
    """A bloom image processing effect.

    Pixels above a luminance threshold emit a soft glow that is added back on top
    of the image, brightening highlights and spilling light around them.
    """
    kernel = compute_halation_kernel(scale, bloom_size)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    x = (luma - bloom_threshold) / max(bloom_softness, 1e-6)
    emit = np.clip(x, 0.0, 1.0)
    emit = emit * emit * (3.0 - 2.0 * emit)

    glow = convolve_2d(rgb * emit[..., np.newaxis], kernel)
    factors = bloom_intensity * np.array(
        [bloom_red_factor, bloom_green_factor, bloom_blue_factor],
        dtype=DEFAULT_DTYPE,
    )
    rgb = rgb + glow * factors
    return rgb


def get_canvas_data(
    shape,
    canvas_mode: CANVAS_MODES,
    canvas_scale: float = 1.0,
    canvas_ratio: float = 1.0,
):
    if "white" in canvas_mode:
        canvas_color = (255, 255, 255)
    elif "black" in canvas_mode:
        canvas_color = (0, 0, 0)
    else:
        canvas_color = (128, 128, 128)

    if "Proportional" in canvas_mode:
        canvas_ratio = shape[1] / shape[0]
        img_ratio = shape[1] / shape[0]
        if img_ratio > canvas_ratio:
            output_resolution = (
                int(shape[1] / canvas_ratio * canvas_scale),
                int(shape[1] * canvas_scale),
            )
        else:
            output_resolution = (
                int(shape[0] * canvas_scale),
                int(shape[0] * canvas_ratio * canvas_scale),
            )
    elif "Fixed" in canvas_mode:
        img_ratio = shape[1] / shape[0]
        if img_ratio > canvas_ratio:
            output_resolution = (
                int(shape[1] / canvas_ratio * canvas_scale),
                int(shape[1] * canvas_scale),
            )
        else:
            output_resolution = (
                int(shape[0] * canvas_scale),
                int(shape[0] * canvas_ratio * canvas_scale),
            )
    elif "Uniform" in canvas_mode:
        side_length = max(shape[:2])
        border_size = int(side_length * (canvas_scale - 1))
        output_resolution = (shape[0] + border_size, shape[1] + border_size)

    offset = np.subtract(output_resolution, shape[:2]) // 2

    return output_resolution, canvas_color, offset


def add_canvas(
    image: np.ndarray,
    canvas_mode: CANVAS_MODES,
    canvas_scale: float = 1.0,
    canvas_ratio: float = 1.0,
):
    """Adds a background canvas to an image."""
    if canvas_mode == "No":
        return image

    output_resolution, canvas_color, offset = get_canvas_data(
        image.shape, canvas_mode, canvas_scale, canvas_ratio
    )

    canvas = np.tensordot(np.ones(output_resolution), canvas_color, axes=0)
    canvas[
        offset[0] : offset[0] + image.shape[0], offset[1] : offset[1] + image.shape[1]
    ] = image

    return canvas.astype(dtype="uint8")


def down_up_blur(image: np.ndarray, scale: int = 50, func: Callable | None = None):
    """
    Blur by downsampling and then upsampling. Very fast on CPU compared to accurate
    blur filters.
    """
    scale = math.ceil(min(image.shape[:2]) / scale)
    # Downsample
    blurred_channels = []
    for c in range(image.shape[-1]):
        # Downsample
        down = cv.resize(
            image[:, :, c],
            (image.shape[1] // scale, image.shape[0] // scale),
            interpolation=cv.INTER_AREA,
        )
        if func is not None:
            down = func(down)
        # Downsample channel
        blurred = ndimage.gaussian_filter(down, sigma=3, truncate=2)

        # Upsample back
        up = ndimage.zoom(blurred, scale, order=1)
        # Crop or pad to match original shape
        up_resized = np.pad(
            up, [(0, max(x - y, 0)) for x, y in zip(image.shape, up.shape)], mode="edge"
        )[: image.shape[0], : image.shape[1]]
        blurred_channels.append(up_resized)

    # Stack back into (H, W, 3)
    return np.stack(blurred_channels, axis=-1)


def burn(
    image: np.ndarray,
    negative_film: FilmSpectral,
    highlight_burn: float,
    burn_scale: float,
):
    """
    Simulates highlight burning, which is a darkroom printing technique to reduce
    the contrast and brightness of highlights. Similar to modern local tone-mapping
    techniques.
    """

    def func(x):
        return np.clip(
            x - negative_film.d_ref[1 if len(negative_film.d_ref) > 1 else 0],
            0,
            None,
        )

    if image.shape[-1] == 3:
        image = image - highlight_burn * down_up_blur(image[..., 1:2], burn_scale, func)
    else:
        image = image - highlight_burn * down_up_blur(image, burn_scale, func)

    image = np.clip(image, 0, None)

    return image


@njit
def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """A fast 1D gaussian kernel."""
    assert size % 2 == 1
    k = size // 2

    kernel = np.zeros(size, dtype=DEFAULT_DTYPE)
    s2 = 2.0 * sigma * sigma

    for i in range(size):
        x = i - k
        kernel[i] = math.exp(-(x * x) / s2)

    kernel /= kernel.sum()
    return kernel


@njit(parallel=True)
def blur_horizontal_masked(
    image: np.ndarray, kernel: np.ndarray, blur_mask: np.ndarray
):
    """Horizontal blur with a mask."""
    h, w, c = image.shape
    k = kernel.shape[0] // 2
    temp = np.empty_like(image)

    for y in prange(h):
        for x in range(w):
            for ch in range(c):
                if blur_mask[ch]:
                    acc = 0.0
                    for i in range(-k, k + 1):
                        ix = min(max(x + i, 0), w - 1)
                        acc += image[y, ix, ch] * kernel[i + k]
                    temp[y, x, ch] = acc
                else:
                    # pass-through for unblurred channels
                    temp[y, x, ch] = image[y, x, ch]

    return temp


@njit(parallel=True)
def blur_vertical_masked(image: np.ndarray, kernel: np.ndarray, blur_mask: np.ndarray):
    """Vertical blur with a mask."""
    h, w, c = image.shape
    k = kernel.shape[0] // 2
    output = np.empty_like(image)

    for y in prange(h):
        for x in range(w):
            for ch in range(c):
                if blur_mask[ch]:
                    acc = 0.0
                    for i in range(-k, k + 1):
                        iy = min(max(y + i, 0), h - 1)
                        acc += image[iy, x, ch] * kernel[i + k]
                    output[y, x, ch] = acc
                else:
                    output[y, x, ch] = image[y, x, ch]

    return output


@njit
def gaussian_blur_separable_masked(image, kernel, blur_channels):
    """
    image: (H, W, C) float32
    kernel: 1D Gaussian kernel
    blur_channels: iterable of booleans, length C
    """
    blur_mask = np.asarray(blur_channels, dtype=np.bool_)
    temp = blur_horizontal_masked(image, kernel, blur_mask)
    return blur_vertical_masked(temp, kernel, blur_mask)


@njit(parallel=True)
def XYZ_to_xyY(image: np.ndarray, eps: float = 1e-8):
    """Converts from CIE XYZ to xyY."""
    h, w, _ = image.shape
    out = np.empty_like(image)

    for y in prange(h):
        for x in range(w):
            X = image[y, x, 0]
            Y = image[y, x, 1]
            Z = image[y, x, 2]

            denom = X + Y + Z
            if denom > eps:
                out[y, x, 0] = X / denom  # x
                out[y, x, 1] = Y / denom  # y
            else:
                out[y, x, 0] = 0.0
                out[y, x, 1] = 0.0

            out[y, x, 2] = Y

    return out


@njit(parallel=True)
def xyY_to_XYZ(image: np.ndarray, eps: float = 1e-8):
    """Converts from CIE xyY to XYZ."""
    h, w, _ = image.shape
    out = np.empty_like(image)

    for y in prange(h):
        for x in range(w):
            cx = image[y, x, 0]
            cy = image[y, x, 1]
            Y = image[y, x, 2]

            if cy > eps:
                inv = Y / cy
                out[y, x, 0] = cx * inv  # X
                out[y, x, 1] = Y  # Y
                out[y, x, 2] = (1.0 - cx - cy) * inv  # Z
            else:
                out[y, x, 0] = 0.0
                out[y, x, 1] = 0.0
                out[y, x, 2] = 0.0

    return out


@njit
def chroma_nr_filter(image: np.ndarray, size: int = 0):
    """A simple chroma noise reduction filter by blurring only color channels."""

    image = XYZ_to_xyY(image)
    size = int(size) * 2 + 1
    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel = gaussian_kernel_1d(size, sigma)

    # Correct blur
    image = gaussian_blur_separable_masked(image, kernel, [True, True, False])

    image = xyY_to_XYZ(image)

    return image
