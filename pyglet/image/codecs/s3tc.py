"""Software decoder for S3TC compressed texture (i.e., DDS).

http://oss.sgi.com/projects/ogl-sample/registry/EXT/texture_compression_s3tc.txt

The decoders accept any bytes-like object (``bytes``, ``bytearray``,
``memoryview`` or ctypes arrays) and decode whole 4x4 blocks into tightly
packed output rows, supporting image dimensions that are not a multiple of
four.  Truncated input, invalid dimensions and size overflows raise
:class:`~pyglet.image.codecs.ImageDecodeException` instead of silently
producing misaligned pixels.

Decoded images are returned as regular
:class:`~pyglet.image.ImageData` objects (RGB or RGBA, unsigned bytes), which
upload through the standard uncompressed texture path; the hardware
``glCompressedTexImage2D`` path in :class:`~pyglet.image.CompressedImageData`
is untouched.
"""

import ctypes

from pyglet.gl import GL_RGB, GL_RGBA
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecodeException

# Compressed block sizes, in bytes.
_DXT1_BLOCK_SIZE = 8
_DXT3_DXT5_BLOCK_SIZE = 16


def _check_dimensions(width, height):
    """Validate the decoded image dimensions."""
    if not isinstance(width, int) or not isinstance(height, int) \
            or isinstance(width, bool) or isinstance(height, bool):
        raise ImageDecodeException(
            'Invalid S3TC image dimensions: width=%r, height=%r'
            % (width, height))
    if width <= 0 or height <= 0:
        raise ImageDecodeException(
            'Invalid S3TC image dimensions: width=%d, height=%d'
            % (width, height))
    if width > 0xffffffff or height > 0xffffffff:
        raise ImageDecodeException(
            'S3TC image dimensions overflow: width=%d, height=%d'
            % (width, height))
    # Guard the ctypes output allocation against pathological sizes.
    if width * height > 0x7fffffff:
        raise ImageDecodeException(
            'S3TC image too large to decode: %dx%d' % (width, height))


def _validate_data(data, width, height, block_size, format_name):
    """Return a (memoryview, blocks_x, blocks_y) tuple after validation.

    S3TC stores complete 4x4 blocks, including for image dimensions that are
    not multiples of four, so the expected compressed size is known exactly.
    """
    _check_dimensions(width, height)
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    expected_size = blocks_x * blocks_y * block_size
    try:
        view = memoryview(data)
    except TypeError:
        raise ImageDecodeException(
            '%s input must be a bytes-like object, got %s'
            % (format_name, type(data).__name__))
    if view.itemsize != 1 or view.format not in ('B', 'b', '<B', '>B', '@B'):
        raise ImageDecodeException(
            '%s input must be a byte buffer, got format %r'
            % (format_name, view.format))
    # Cast to an unsigned-byte view so element access works for every
    # bytes-like input (ctypes arrays expose a platform-format view such as
    # '<B', whose elements cannot be indexed directly on Python 3.14+).
    view = view.cast('B')
    if view.nbytes != expected_size:
        raise ImageDecodeException(
            '%s data length %d does not match %dx%d image: expected %d bytes '
            'for %dx%d blocks of %d bytes (truncated or corrupt stream?)'
            % (format_name, view.nbytes, width, height, expected_size,
               blocks_x, blocks_y, block_size))
    return view, blocks_x, blocks_y


def _expand5(value):
    """Replicate a 5-bit channel to 8 bits (bit replication)."""
    return (value << 3) | (value >> 2)


def _expand6(value):
    """Replicate a 6-bit channel to 8 bits (bit replication)."""
    return (value << 2) | (value >> 4)


def _read_color_endpoints(view, offset):
    """Read two RGB565 endpoints as both raw 5/6-bit and expanded 8-bit channels."""
    color0 = view[offset] | (view[offset + 1] << 8)
    color1 = view[offset + 2] | (view[offset + 3] << 8)

    r0 = color0 & 0x1f
    g0 = (color0 >> 5) & 0x3f
    b0 = (color0 >> 11) & 0x1f
    r1 = color1 & 0x1f
    g1 = (color1 >> 5) & 0x3f
    b1 = (color1 >> 11) & 0x1f
    raw0 = (r0, g0, b0)
    raw1 = (r1, g1, b1)
    channels0 = (_expand5(r0), _expand6(g0), _expand5(b0))
    channels1 = (_expand5(r1), _expand6(g1), _expand5(b1))
    return color0, color1, channels0, channels1, raw0, raw1


def _palette_565(color0, color1, c0, c1):
    """Four-entry RGB565 palette; interpolation stays in 5/6-bit precision."""
    r0, g0, b0 = c0
    r1, g1, b1 = c1
    if color0 > color1:
        color2 = ((2 * r0 + r1) // 3,
                  (2 * g0 + g1) // 3,
                  (2 * b0 + b1) // 3)
        color3 = ((r0 + 2 * r1) // 3,
                  (g0 + 2 * g1) // 3,
                  (b0 + 2 * b1) // 3)
    else:
        # 3-color mode: index 3 is black in RGB (no-alpha) DXT1.
        color2 = ((r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2)
        color3 = (0, 0, 0)
    return (
        color0,
        color1,
        color2[0] | (color2[1] << 5) | (color2[2] << 11),
        color3[0] | (color3[1] << 5) | (color3[2] << 11),
    )


def _palette_rgba(color0, color1, c0, c1, dxt1):
    """Four-entry RGBA palette with 8-bit bit-replicated channels."""
    r0, g0, b0 = c0
    r1, g1, b1 = c1

    if color0 > color1:
        color2 = ((2 * r0 + r1) // 3,
                  (2 * g0 + g1) // 3,
                  (2 * b0 + b1) // 3)
        color3 = ((r0 + 2 * r1) // 3,
                  (g0 + 2 * g1) // 3,
                  (b0 + 2 * b1) // 3)
    else:
        color2 = ((r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2)
        if dxt1:
            # DXT1 3-color mode: index 3 is transparent black.
            color3 = None
        else:
            # DXT3/DXT5 only ever use four colors; index 3 equals color1.
            color3 = (r1, g1, b1)

    palette = [(r0, g0, b0, 255), (r1, g1, b1, 255),
               (color2[0], color2[1], color2[2], 255), None]
    if color3 is None:
        palette[3] = (0, 0, 0, 0)
    else:
        palette[3] = (color3[0], color3[1], color3[2], 255)
    return palette


def _read_color_codes(view, offset):
    """Read the 16 2-bit color indices as a 32-bit word."""
    return (view[offset] | (view[offset + 1] << 8)
            | (view[offset + 2] << 16) | (view[offset + 3] << 24))


def decode_dxt1_rgb(data, width, height):
    """Decode a DXT1 texture without alpha to an 8-bit RGB ImageData."""
    view, blocks_x, blocks_y = _validate_data(
        data, width, height, _DXT1_BLOCK_SIZE, 'DXT1')

    # Decode into native 5:6:5 precision first, then expand to RGB bytes.
    packed = (ctypes.c_uint16 * (width * height))()
    offset = 0
    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            color0, color1, _, _, raw0, raw1 = _read_color_endpoints(
                view, offset)
            palette = _palette_565(color0, color1, raw0, raw1)
            bits = _read_color_codes(view, offset + 4)

            x0 = block_x * 4
            y0 = block_y * 4
            for sub_y in range(4):
                y = y0 + sub_y
                if y >= height:
                    bits >>= 8
                    continue
                row = y * width
                for sub_x in range(4):
                    x = x0 + sub_x
                    code = bits & 0x3
                    bits >>= 2
                    if x >= width:
                        continue
                    packed[row + x] = palette[code]
            offset += _DXT1_BLOCK_SIZE

    out = (ctypes.c_ubyte * (width * height * 3))()
    i = 0
    for pixel in packed:
        out[i] = _expand5((pixel >> 11) & 0x1f)       # red
        out[i + 1] = _expand6((pixel >> 5) & 0x3f)    # green
        out[i + 2] = _expand5(pixel & 0x1f)           # blue
        i += 3
    return ImageData(width, height, 'RGB', out)


def _dxt3_alpha_reader(view, offset):
    """Closure yielding the 16 explicit 4-bit alpha values as bytes."""
    alpha_word = 0
    for i in range(8):
        alpha_word |= view[offset + i] << (8 * i)
    state = {'word': alpha_word}

    def read(skip=0):
        value = 0
        for _ in range(skip + 1):
            value = state['word'] & 0xf
            state['word'] >>= 4
        # Bit-replicate the nibble into a full byte.
        return (value << 4) | value

    return read


def _dxt5_alpha_reader(view, offset):
    """Closure yielding the 16 interpolated alpha values."""
    alpha0 = view[offset]
    alpha1 = view[offset + 1]

    # The 48 3-bit indices are stored little-endian across six bytes.
    alpha_bits = 0
    for i in range(6):
        alpha_bits |= view[offset + 2 + i] << (8 * i)

    if alpha0 > alpha1:
        table = (
            alpha0,
            alpha1,
            (6 * alpha0 + alpha1) // 7,
            (5 * alpha0 + 2 * alpha1) // 7,
            (4 * alpha0 + 3 * alpha1) // 7,
            (3 * alpha0 + 4 * alpha1) // 7,
            (2 * alpha0 + 5 * alpha1) // 7,
            (alpha0 + 6 * alpha1) // 7,
        )
    else:
        table = (
            alpha0,
            alpha1,
            (4 * alpha0 + alpha1) // 5,
            (3 * alpha0 + 2 * alpha1) // 5,
            (2 * alpha0 + 3 * alpha1) // 5,
            (alpha0 + 4 * alpha1) // 5,
            0,
            255,
        )

    state = {'bits': alpha_bits}

    def read(skip=0):
        value = 0
        for _ in range(skip + 1):
            value = table[state['bits'] & 0x7]
            state['bits'] >>= 3
        return value

    return read


def _decode_rgba(data, width, height, block_size, format_name, alpha_factory):
    """Shared RGBA decoder for DXT1-RGBA, DXT3 and DXT5.

    ``alpha_factory`` is ``None`` for DXT1 (alpha comes from the color
    palette); otherwise it builds a per-block alpha reader closure.
    """
    view, blocks_x, blocks_y = _validate_data(
        data, width, height, block_size, format_name)
    dxt1 = alpha_factory is None

    out = (ctypes.c_ubyte * (width * height * 4))()
    offset = 0
    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            if dxt1:
                color_offset = offset
                alpha_read = None
            else:
                alpha_read = alpha_factory(view, offset)
                color_offset = offset + 8

            color0, color1, ch0, ch1, _, _ = _read_color_endpoints(
                view, color_offset)
            palette = _palette_rgba(color0, color1, ch0, ch1, dxt1)
            bits = _read_color_codes(view, color_offset + 4)

            x0 = block_x * 4
            y0 = block_y * 4
            for sub_y in range(4):
                y = y0 + sub_y
                if y >= height:
                    bits >>= 8
                    if alpha_read is not None:
                        alpha_read(skip=3)
                    continue
                row = y * width * 4
                for sub_x in range(4):
                    x = x0 + sub_x
                    code = bits & 0x3
                    bits >>= 2
                    if alpha_read is not None:
                        alpha = alpha_read()
                    else:
                        alpha = None
                    if x >= width:
                        continue
                    r, g, b, palette_alpha = palette[code]
                    if alpha is None:
                        alpha = palette_alpha
                    out[row + x * 4] = r
                    out[row + x * 4 + 1] = g
                    out[row + x * 4 + 2] = b
                    out[row + x * 4 + 3] = alpha
            offset += block_size

    return ImageData(width, height, 'RGBA', out)


def decode_dxt1_rgba(data, width, height):
    """Decode a DXT1 texture with 1-bit alpha to an 8-bit RGBA ImageData."""
    return _decode_rgba(data, width, height, _DXT1_BLOCK_SIZE, 'DXT1', None)


def decode_dxt3(data, width, height):
    """Decode a DXT3 (explicit 4-bit alpha) texture to an 8-bit RGBA ImageData."""
    return _decode_rgba(data, width, height, _DXT3_DXT5_BLOCK_SIZE,
                        'DXT3', _dxt3_alpha_reader)


def decode_dxt5(data, width, height):
    """Decode a DXT5 (interpolated alpha) texture to an 8-bit RGBA ImageData."""
    return _decode_rgba(data, width, height, _DXT3_DXT5_BLOCK_SIZE,
                        'DXT5', _dxt5_alpha_reader)
