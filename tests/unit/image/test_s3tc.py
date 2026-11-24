"""Unit tests for the S3TC (DXT1/DXT3/DXT5) software decoder.

The tests use hand-built, fixed compressed blocks and an independent
reference decoder (written straight from the S3TC spec, with a different
code structure than the production decoder) to verify exact RGBA output,
multi-block placement, non-multiple-of-four dimensions and failure
boundaries.
"""

import os
import unittest

from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecodeException, s3tc
from pyglet.image.codecs.dds import DDSImageDecoder

__noninteractive = True

TEST_IMAGES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'images'))


# ---------------------------------------------------------------------------
# Independent reference helpers (spec text re-implementation, deliberately
# shaped differently from the production code).
# ---------------------------------------------------------------------------

def _ref_expand5(v):
    return (v << 3) | (v >> 2)


def _ref_expand6(v):
    return (v << 2) | (v >> 4)


def ref_dxt_colors(c0, c1, dxt1):
    """Return the four (r, g, b, a) palette entries per the S3TC spec."""
    p0 = (_ref_expand5(c0 & 0x1f),
          _ref_expand6((c0 >> 5) & 0x3f),
          _ref_expand5((c0 >> 11) & 0x1f))
    p1 = (_ref_expand5(c1 & 0x1f),
          _ref_expand6((c1 >> 5) & 0x3f),
          _ref_expand5((c1 >> 11) & 0x1f))

    def blend(na, nb, n):
        return tuple((na * a + nb * b) // n
                     for a, b in zip(p0, p1))

    if c0 > c1:
        p2, p3 = blend(2, 1, 3), blend(1, 2, 3)
        a2 = a3 = 255
    else:
        p2 = blend(1, 1, 2)
        a2 = 255
        if dxt1:
            p3, a3 = (0, 0, 0), 0          # transparent black
        else:
            p3, a3 = p1, 255               # DXT3/5: color1, opaque
    return [p0 + (255,), p1 + (255,), p2 + (a2,), p3 + (a3,)]


def ref_dxt3_alpha(block):
    word = int.from_bytes(block[0:8], 'little')
    return [((word >> (4 * i)) & 0xf) for i in range(16)]


def ref_dxt5_alpha(block):
    a0, a1 = block[0], block[1]
    bits = int.from_bytes(block[2:8], 'little')
    if a0 > a1:
        table = [a0, a1]
        for num, den in ((6, 7), (5, 7), (4, 7), (3, 7), (2, 7), (1, 7)):
            table.append((num * a0 + (den - num) * a1) // den)
    else:
        table = [a0, a1]
        for num, den in ((4, 5), (3, 5), (2, 5), (1, 5)):
            table.append((num * a0 + (den - num) * a1) // den)
        table += [0, 255]
    return [table[(bits >> (3 * i)) & 0x7] for i in range(16)]


def ref_decode(block, width, height, mode):
    """Decode a single 4x4 block; only texels inside width/height are kept."""
    if mode == 'dxt1':
        c0 = int.from_bytes(block[0:2], 'little')
        c1 = int.from_bytes(block[2:4], 'little')
        codes = int.from_bytes(block[4:8], 'little')
        palette = ref_dxt_colors(c0, c1, dxt1=True)
        alphas = [255] * 16
    else:
        if mode == 'dxt3':
            color_base = 8
            alphas_nib = ref_dxt3_alpha(block)
            alphas = [(a << 4) | a for a in alphas_nib]
        else:
            color_base = 8
            alphas = ref_dxt5_alpha(block)
        c0 = int.from_bytes(block[color_base:color_base + 2], 'little')
        c1 = int.from_bytes(block[color_base + 2:color_base + 4], 'little')
        codes = int.from_bytes(block[color_base + 4:color_base + 8], 'little')
        palette = ref_dxt_colors(c0, c1, dxt1=False)

    out = {}
    for i in range(16):
        x, y = i % 4, i // 4
        if x >= width or y >= height:
            continue
        r, g, b, a = palette[(codes >> (2 * i)) & 0x3]
        if mode != 'dxt1':
            a = alphas[i]
        out[(x, y)] = (r, g, b, a)
    return out


def make_dxt1_block(c0, c1, codes, alpha_mode=False):
    """Build an 8-byte DXT1 block. *codes* is 16 2-bit indices in texel order."""
    word = 0
    for i, code in enumerate(codes):
        word |= (code & 0x3) << (2 * i)
    return c0.to_bytes(2, 'little') + c1.to_bytes(2, 'little') \
        + word.to_bytes(4, 'little')


def make_dxt3_block(alpha_nibs, c0, c1, codes):
    word = 0
    for i, code in enumerate(codes):
        word |= (code & 0x3) << (2 * i)
    aword = 0
    for i, nib in enumerate(alpha_nibs):
        aword |= (nib & 0xf) << (4 * i)
    return aword.to_bytes(8, 'little') \
        + c0.to_bytes(2, 'little') + c1.to_bytes(2, 'little') \
        + word.to_bytes(4, 'little')


def make_dxt5_block(a0, a1, alpha_codes, c0, c1, codes):
    cword = 0
    for i, code in enumerate(codes):
        cword |= (code & 0x3) << (2 * i)
    abit = 0
    for i, code in enumerate(alpha_codes):
        abit |= (code & 0x7) << (3 * i)
    return bytes((a0, a1)) + abit.to_bytes(6, 'little') \
        + c0.to_bytes(2, 'little') + c1.to_bytes(2, 'little') \
        + cword.to_bytes(4, 'little')


def rgba_at(image, x, y):
    raw = image.get_bytes()
    pitch = image.width * 4
    i = y * pitch + x * 4
    return tuple(raw[i:i + 4])


def rgb_at(image, x, y):
    raw = image.get_bytes()
    pitch = image.width * 3
    i = y * pitch + x * 3
    return tuple(raw[i:i + 3])


# Two distinctive RGB565 endpoints: red-ish and blue-ish.
C0 = 0xf800   # r31 g0  b0
C1 = 0x001f   # r0  g0  b31


class DXT1DecodeTests(unittest.TestCase):

    def test_single_block_all_codes_opaque_mode(self):
        codes = [0, 1, 2, 3] * 4
        block = make_dxt1_block(C0, C1, codes)  # c0 > c1: 4-color mode
        image = s3tc.decode_dxt1_rgba(block, 4, 4)
        self.assertIsInstance(image, ImageData)
        expected = ref_decode(block, 4, 4, 'dxt1')
        for (x, y), want in expected.items():
            self.assertEqual(rgba_at(image, x, y), want)

    def test_three_color_mode_transparent_black(self):
        # c0 <= c1 selects 3-color mode; code 3 must be (0,0,0,0).
        codes = [0, 1, 2, 3] * 4
        block = make_dxt1_block(C1, C0, codes)
        image = s3tc.decode_dxt1_rgba(block, 4, 4)
        expected = ref_decode(block, 4, 4, 'dxt1')
        for (x, y), want in expected.items():
            self.assertEqual(rgba_at(image, x, y), want)
        self.assertEqual(rgba_at(image, 3, 0), (0, 0, 0, 0))
        # Opaque texels keep full alpha.
        self.assertEqual(rgba_at(image, 0, 0)[3], 255)

    def test_rgb_path_has_no_alpha_and_three_color_black_opaque(self):
        codes = [0, 1, 2, 3] * 4
        block = make_dxt1_block(C1, C0, codes)  # 3-color mode
        image = s3tc.decode_dxt1_rgb(block, 4, 4)
        self.assertEqual(image._current_format, 'RGB')
        raw = image.get_bytes()
        self.assertEqual(len(raw), 4 * 4 * 3)
        # Index 3 texel is black RGB (no alpha channel exists).
        i = (0 * 4 + 3) * 3
        self.assertEqual(tuple(raw[i:i + 3]), (0, 0, 0))
        # Endpoint expansion uses bit replication, not plain shifts.
        i0 = 0
        self.assertEqual(tuple(raw[i0:i0 + 3]),
                         (_ref_expand5(0), _ref_expand6(0), _ref_expand5(31)))

    def test_multiple_blocks_placement_and_stride(self):
        # 8x8 image: four blocks, each filled with a single distinct code.
        blocks = []
        block_pixels = {}
        for by in range(2):
            for bx in range(2):
                code = (by * 2 + bx) & 0x3
                block = make_dxt1_block(C0, C1, [code] * 16)
                blocks.append(block)
                block_pixels[(bx, by)] = ref_decode(block, 4, 4, 'dxt1')
        data = b''.join(blocks)  # order: row 0 left-to-right, then row 1
        image = s3tc.decode_dxt1_rgba(data, 8, 8)
        for (bx, by), ref in block_pixels.items():
            for (x, y), want in ref.items():
                self.assertEqual(rgba_at(image, bx * 4 + x, by * 4 + y), want)

    def test_non_multiple_of_four_dimensions(self):
        for w, h in ((1, 1), (3, 5), (5, 3), (7, 6), (4, 9), (9, 4)):
            blocks_x, blocks_y = (w + 3) // 4, (h + 3) // 4
            data = b''.join(
                make_dxt1_block(C0, C1, [(i + j) % 4 for i in range(16)])
                for j in range(blocks_x * blocks_y))
            image = s3tc.decode_dxt1_rgba(data, w, h)
            self.assertEqual((image.width, image.height), (w, h))
            self.assertEqual(len(image.get_bytes()), w * h * 4)
            # Verify against the reference, block by block.
            for by in range(blocks_y):
                for bx in range(blocks_x):
                    block = data[(by * blocks_x + bx) * 8:
                                 (by * blocks_x + bx + 1) * 8]
                    ref = ref_decode(block, min(4, w - bx * 4),
                                     min(4, h - by * 4), 'dxt1')
                    for (x, y), want in ref.items():
                        self.assertEqual(
                            rgba_at(image, bx * 4 + x, by * 4 + y), want,
                            msg=(w, h, bx, by, x, y))


class DXT3DecodeTests(unittest.TestCase):

    def test_explicit_alpha_and_color_codes(self):
        nibs = [i % 16 for i in range(16)]
        codes = [(i * 3) % 4 for i in range(16)]
        block = make_dxt3_block(nibs, C0, C1, codes)
        image = s3tc.decode_dxt3(block, 4, 4)
        expected = ref_decode(block, 4, 4, 'dxt3')
        for (x, y), want in expected.items():
            self.assertEqual(rgba_at(image, x, y), want)

    def test_alpha_nibble_bit_replication(self):
        block = make_dxt3_block([0xa] * 16, C0, C1, [0] * 16)
        image = s3tc.decode_dxt3(block, 4, 4)
        self.assertEqual(rgba_at(image, 0, 0)[3], 0xaa)
        block = make_dxt3_block([0x0] * 16, C0, C1, [0] * 16)
        image = s3tc.decode_dxt3(block, 4, 4)
        self.assertEqual(rgba_at(image, 0, 0)[3], 0x00)
        block = make_dxt3_block([0xf] * 16, C0, C1, [0] * 16)
        image = s3tc.decode_dxt3(block, 4, 4)
        self.assertEqual(rgba_at(image, 0, 0)[3], 0xff)

    def test_color0_le_color1_still_four_colors(self):
        # DXT3 never has a transparent palette entry: code 3 == color1.
        block = make_dxt3_block([0xf] * 16, C1, C0, [3] * 16)
        image = s3tc.decode_dxt3(block, 4, 4)
        px = rgba_at(image, 0, 0)
        self.assertEqual(px, (_ref_expand5(0), _ref_expand6(0),
                              _ref_expand5(31), 255))

    def test_non_multiple_of_four_dimensions(self):
        w, h = 5, 7
        blocks_x, blocks_y = 2, 2
        data = b''.join(
            make_dxt3_block([(i + k) % 16 for i in range(16)],
                            C0, C1, [(i * 7 + k) % 4 for i in range(16)])
            for k in range(blocks_x * blocks_y))
        image = s3tc.decode_dxt3(data, w, h)
        self.assertEqual(len(image.get_bytes()), w * h * 4)
        for by in range(blocks_y):
            for bx in range(blocks_x):
                block = data[(by * blocks_x + bx) * 16:
                             (by * blocks_x + bx + 1) * 16]
                ref = ref_decode(block, min(4, w - bx * 4),
                                 min(4, h - by * 4), 'dxt3')
                for (x, y), want in ref.items():
                    self.assertEqual(rgba_at(image, bx * 4 + x, by * 4 + y),
                                     want)


class DXT5DecodeTests(unittest.TestCase):

    def test_interpolated_eight_alpha_gradient(self):
        a0, a1 = 255, 0
        alpha_codes = list(range(8)) * 2
        block = make_dxt5_block(a0, a1, alpha_codes, C0, C1,
                                [i % 4 for i in range(16)])
        image = s3tc.decode_dxt5(block, 4, 4)
        expected = ref_decode(block, 4, 4, 'dxt5')
        for (x, y), want in expected.items():
            self.assertEqual(rgba_at(image, x, y), want)

    def test_six_alpha_gradient_with_explicit_endpoints(self):
        # a0 <= a1: codes 6 and 7 are hard 0 and 255, no interpolation.
        block = make_dxt5_block(100, 200, list(range(8)) * 2,
                                C0, C1, [i % 4 for i in range(16)])
        image = s3tc.decode_dxt5(block, 4, 4)
        expected = ref_decode(block, 4, 4, 'dxt5')
        for (x, y), want in expected.items():
            self.assertEqual(rgba_at(image, x, y), want)
        self.assertEqual(rgba_at(image, 6 % 4, 6 // 4)[3], 0)
        self.assertEqual(rgba_at(image, 7 % 4, 7 // 4)[3], 255)

    def test_non_multiple_of_four_dimensions(self):
        w, h = 6, 3
        blocks_x, blocks_y = 2, 1
        data = b''.join(
            make_dxt5_block(200, 40, [(i + k) % 8 for i in range(16)],
                            C0, C1, [(i + k) % 4 for i in range(16)])
            for k in range(blocks_x * blocks_y))
        image = s3tc.decode_dxt5(data, w, h)
        self.assertEqual(len(image.get_bytes()), w * h * 4)
        for bx in range(blocks_x):
            block = data[bx * 16:(bx + 1) * 16]
            ref = ref_decode(block, min(4, w - bx * 4), h, 'dxt5')
            for (x, y), want in ref.items():
                self.assertEqual(rgba_at(image, bx * 4 + x, y), want)


class InputValidationTests(unittest.TestCase):

    def test_accepts_all_bytes_like_inputs(self):
        # The original bug: a string regex raised TypeError on bytes.
        block = make_dxt1_block(C0, C1, [0] * 16)
        for factory in (bytes, bytearray, memoryview):
            image = s3tc.decode_dxt1_rgba(factory(block), 4, 4)
            self.assertEqual(len(image.get_bytes()), 64)
        import ctypes
        image = s3tc.decode_dxt1_rgba(
            (ctypes.c_ubyte * 8)(*block), 4, 4)
        self.assertEqual(len(image.get_bytes()), 64)

    def test_truncated_data_raises_with_location(self):
        block = make_dxt1_block(C0, C1, [0] * 16)
        with self.assertRaises(ImageDecodeException) as ctx:
            s3tc.decode_dxt1_rgba(block[:7], 4, 4)
        self.assertIn('DXT1', str(ctx.exception))
        self.assertIn('expected 8 bytes', str(ctx.exception))

        dxt5_block = make_dxt5_block(255, 0, [0] * 16, C0, C1, [0] * 16)
        with self.assertRaises(ImageDecodeException) as ctx:
            s3tc.decode_dxt5(dxt5_block + b'\x00', 4, 4)
        self.assertIn('DXT5', str(ctx.exception))

    def test_truncated_multi_block_stream(self):
        # 8x4 needs two DXT1 blocks; supply only one.
        block = make_dxt1_block(C0, C1, [0] * 16)
        with self.assertRaises(ImageDecodeException) as ctx:
            s3tc.decode_dxt1_rgba(block, 8, 4)
        self.assertIn('2x1 blocks', str(ctx.exception))

    def test_invalid_dimensions(self):
        block8 = make_dxt1_block(C0, C1, [0] * 16)
        for w, h in ((0, 4), (4, 0), (-1, 4), (4, -1)):
            with self.assertRaises(ImageDecodeException):
                s3tc.decode_dxt1_rgba(block8, w, h)
        with self.assertRaises(ImageDecodeException):
            s3tc.decode_dxt1_rgb(block8, 4.0, 4)
        with self.assertRaises(ImageDecodeException):
            s3tc.decode_dxt1_rgba(block8, True, 4)

    def test_dimension_overflow_raises(self):
        with self.assertRaises(ImageDecodeException):
            s3tc.decode_dxt5(b'', 1 << 31, 1 << 31)

    def test_non_buffer_input_raises(self):
        with self.assertRaises(ImageDecodeException):
            s3tc.decode_dxt3('not-a-buffer', 4, 4)

    def test_non_byte_buffer_input_raises(self):
        import array
        with self.assertRaises(ImageDecodeException):
            s3tc.decode_dxt1_rgba(array.array('H', [0] * 4), 4, 4)

    def test_dxt3_dxt5_require_sixteen_byte_blocks(self):
        good = make_dxt3_block([0] * 16, C0, C1, [0] * 16)
        s3tc.decode_dxt3(good, 4, 4)
        with self.assertRaises(ImageDecodeException):
            s3tc.decode_dxt3(good[:15], 4, 4)
        good5 = make_dxt5_block(0, 0, [0] * 16, C0, C1, [0] * 16)
        s3tc.decode_dxt5(good5, 4, 4)
        with self.assertRaises(ImageDecodeException):
            s3tc.decode_dxt5(good5 + b'xx', 4, 4)


class DDSIntegrationTests(unittest.TestCase):
    """End-to-end checks through the DDS decoder and fallback decoder hook."""

    def _load(self, name):
        path = os.path.join(TEST_IMAGES, name)
        with open(path, 'rb') as f:
            return DDSImageDecoder().decode(path, f)

    def test_fixed_dds_files_decode_through_software_path(self):
        cases = {
            'rgb_dxt1.dds': ('RGB', 3),
            'rgba_dxt1.dds': ('RGBA', 4),
            'rgba_dxt3.dds': ('RGBA', 4),
            'rgba_dxt5.dds': ('RGBA', 4),
        }
        for name, (fmt, channels) in cases.items():
            compressed = self._load(name)
            image = compressed.decoder(compressed.data,
                                       compressed.width, compressed.height)
            self.assertIsInstance(image, ImageData)
            self.assertEqual((image.width, image.height), (256, 256))
            self.assertEqual(image._current_format, fmt)
            self.assertEqual(len(image.get_bytes()),
                             256 * 256 * channels)

    def test_dds_hardware_path_metadata_unchanged(self):
        compressed = self._load('rgba_dxt5.dds')
        # The CompressedImageData hardware path (gl_format, extension, raw
        # bytes) must remain exactly as the DDS reader produced it.
        self.assertEqual(compressed.extension,
                         'GL_EXT_texture_compression_s3tc')
        self.assertEqual(len(compressed.data), 65536)

    def test_truncated_dds_file_is_rejected(self):
        path = os.path.join(TEST_IMAGES, 'rgba_dxt5.dds')
        with open(path, 'rb') as f:
            raw = f.read()
        import io
        with self.assertRaises(ImageDecodeException):
            DDSImageDecoder().decode('truncated.dds', io.BytesIO(raw[:-100]))

    def test_bad_header_is_rejected(self):
        import io
        with self.assertRaises(ImageDecodeException):
            DDSImageDecoder().decode('bad.dds', io.BytesIO(b'NOTADDS' + b'\0' * 200))

    def test_zero_dimension_header_is_rejected(self):
        import io
        import struct
        path = os.path.join(TEST_IMAGES, 'rgba_dxt3.dds')
        with open(path, 'rb') as f:
            header = bytearray(f.read(128))
        # dwHeight lives at header offset 12, dwWidth at 16.
        struct.pack_into('<I', header, 12, 0)
        with self.assertRaises(ImageDecodeException) as ctx:
            DDSImageDecoder().decode('zero.dds', io.BytesIO(bytes(header)))
        self.assertIn('dimensions', str(ctx.exception))

    def test_zero_mipmap_count_header_is_rejected(self):
        import io
        import struct
        from pyglet.image.codecs.dds import DDSD_MIPMAPCOUNT
        path = os.path.join(TEST_IMAGES, 'rgba_dxt3.dds')
        with open(path, 'rb') as f:
            header = bytearray(f.read(128))
        flags = struct.unpack_from('<I', header, 8)[0] | DDSD_MIPMAPCOUNT
        struct.pack_into('<I', header, 8, flags)
        struct.pack_into('<I', header, 28, 0)  # dwMipMapCount
        with self.assertRaises(ImageDecodeException) as ctx:
            DDSImageDecoder().decode('nomips.dds', io.BytesIO(bytes(header)))
        self.assertIn('mipmap', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
