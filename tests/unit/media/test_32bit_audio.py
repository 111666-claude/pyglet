"""Tests for 32 bit (integer and floating point) sample support in the
media pipeline: format description, decoding, buffering and driver
backends.

All sample data is generated synthetically with boundary values so that
sign, amplitude, channel interleaving, duration and the decode-to-output
roundtrip can be verified exactly.
"""
import struct
import time
import wave
from unittest import mock

import pytest

from pyglet.media.codecs.base import AudioFormat, Source, StaticSource
from pyglet.media.exceptions import MediaException


# ---------------------------------------------------------------------------
# Helpers to generate synthetic audio files
# ---------------------------------------------------------------------------

def write_int_wav(path, channels, sample_size, sample_rate, frames):
    """Write an integer PCM wave file. ``frames`` is a flat list of
    interleaved integer samples (unsigned for 8 bit, signed otherwise)."""
    if sample_size == 8:
        data = struct.pack(f'<{len(frames)}B', *frames)
    elif sample_size == 16:
        data = struct.pack(f'<{len(frames)}h', *frames)
    elif sample_size == 32:
        data = struct.pack(f'<{len(frames)}i', *frames)
    else:
        raise ValueError(sample_size)

    with wave.open(str(path), 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_size // 8)
        w.setframerate(sample_rate)
        w.writeframes(data)
    return data


def write_float_wav(path, channels, sample_rate, frames):
    """Write a 32 bit IEEE float wave file by hand (the standard library
    wave module cannot write these). ``frames`` is a flat list of
    interleaved floats."""
    data = struct.pack(f'<{len(frames)}f', *frames)
    block_align = channels * 4
    byte_rate = sample_rate * block_align
    fmt = struct.pack('<HHIIHH', 3, channels, sample_rate, byte_rate,
                      block_align, 32)
    riff_size = 4 + (8 + len(fmt)) + (8 + len(data))
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', riff_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', len(fmt)))
        f.write(fmt)
        f.write(b'data')
        f.write(struct.pack('<I', len(data)))
        f.write(data)
    return data


def decode_all(source, chunk=4096):
    """Decode a source completely. Returns (data, timestamps, durations)."""
    chunks = []
    timestamps = []
    durations = []
    while True:
        packet = source.get_audio_data(chunk)
        if packet is None:
            break
        chunks.append(packet.data)
        timestamps.append(packet.timestamp)
        durations.append(packet.duration)
    return b''.join(chunks), timestamps, durations


# Boundary samples, chosen to catch sign, amplitude and interleaving bugs.
S32_LEFT = [0, 1, -1, 2 ** 31 - 1, -2 ** 31, 2 ** 30, -(2 ** 30), 123456789]
S32_RIGHT = [-1, 1, 0, -2 ** 31, 2 ** 31 - 1, -(2 ** 30), 2 ** 30, -987654321]

F32_LEFT = [0.0, 1.0, -1.0, 0.5, -0.5, 1e-7, -1e-7, 3.0e38]
F32_RIGHT = [-0.0, -1.0, 1.0, -0.25, 0.25, -1e-7, 1e-7, -3.0e38]

S16_LEFT = [0, 1, -1, 32767, -32768, 16384, -16384, 12345]
S16_RIGHT = [-1, 1, 0, -32768, 32767, -16384, 16384, -12345]

U8_LEFT = [0, 1, 127, 128, 129, 255, 64, 192]
U8_RIGHT = [255, 254, 128, 127, 126, 0, 192, 64]

SAMPLE_RATE = 44100


def interleave(left, right):
    return [s for pair in zip(left, right) for s in pair]


# ---------------------------------------------------------------------------
# AudioFormat
# ---------------------------------------------------------------------------

class TestAudioFormatSampleType:
    def test_default_sample_type_backward_compatible(self):
        assert AudioFormat(1, 8, 44100).sample_type == 'u'
        assert AudioFormat(1, 16, 44100).sample_type == 's'
        assert AudioFormat(2, 24, 44100).sample_type == 's'
        assert AudioFormat(2, 32, 44100).sample_type == 's'

    def test_explicit_sample_type(self):
        assert AudioFormat(2, 32, 44100, 'f').sample_type == 'f'
        assert AudioFormat(2, 32, 44100, 's').sample_type == 's'

    def test_int32_and_float32_are_different_formats(self):
        assert AudioFormat(2, 32, 44100, 's') != AudioFormat(2, 32, 44100, 'f')

    def test_legacy_equality_unaffected(self):
        assert AudioFormat(2, 16, 44100) == AudioFormat(2, 16, 44100, 's')
        assert AudioFormat(1, 8, 22050) == AudioFormat(1, 8, 22050, 'u')

    def test_invalid_sample_type_rejected(self):
        with pytest.raises(ValueError):
            AudioFormat(2, 16, 44100, 'x')

    def test_float_requires_32_or_64_bit(self):
        with pytest.raises(ValueError):
            AudioFormat(2, 16, 44100, 'f')

    def test_bytes_per_frame_32bit(self):
        assert AudioFormat(1, 32, 44100, 'f').bytes_per_frame == 4
        assert AudioFormat(2, 32, 44100, 's').bytes_per_frame == 8
        assert AudioFormat(2, 32, 44100, 'f').bytes_per_second == 352800


# ---------------------------------------------------------------------------
# Driver backend format mapping
# ---------------------------------------------------------------------------

class TestOpenALFormatMapping:
    @pytest.fixture(autouse=True)
    def openal_interface(self):
        pytest.importorskip('pyglet.media.drivers.openal.interface')
        from pyglet.media.drivers.openal import interface
        self.interface = interface
        self.al = interface.al

    def test_legacy_formats_unchanged(self):
        get = self.interface.get_openal_format
        assert get(AudioFormat(1, 8, 44100)) == self.al.AL_FORMAT_MONO8
        assert get(AudioFormat(1, 16, 44100)) == self.al.AL_FORMAT_MONO16
        assert get(AudioFormat(2, 8, 44100)) == self.al.AL_FORMAT_STEREO8
        assert get(AudioFormat(2, 16, 44100)) == self.al.AL_FORMAT_STEREO16

    def test_float32_with_extension(self):
        get = self.interface.get_openal_format
        assert get(AudioFormat(1, 32, 44100, 'f'), True) == self.al.AL_FORMAT_MONO_FLOAT32
        assert get(AudioFormat(2, 32, 44100, 'f'), True) == self.al.AL_FORMAT_STEREO_FLOAT32

    def test_float32_without_extension_raises(self):
        with pytest.raises(MediaException):
            self.interface.get_openal_format(AudioFormat(2, 32, 44100, 'f'), False)

    def test_int32_raises_instead_of_wrong_bit_depth(self):
        # OpenAL has no 32 bit integer format; it must refuse rather than
        # silently play the data at the wrong bit depth.
        with pytest.raises(MediaException):
            self.interface.get_openal_format(AudioFormat(2, 32, 44100, 's'), True)


class TestPulseFormatMapping:
    @pytest.fixture(autouse=True)
    def pulse_interface(self):
        pytest.importorskip('pyglet.media.drivers.pulse.interface')
        from pyglet.media.drivers.pulse import interface
        from pyglet.media.drivers.pulse import lib_pulseaudio as pa
        self.interface = interface
        self.pa = pa

    def create_spec(self, audio_format):
        # create_sample_spec does not use instance state.
        return self.interface.PulseAudioStream.create_sample_spec(None, audio_format)

    def test_legacy_formats_unchanged(self):
        assert self.create_spec(AudioFormat(1, 8, 44100)).format == self.pa.PA_SAMPLE_U8
        assert self.create_spec(AudioFormat(2, 16, 44100)).format == self.pa.PA_SAMPLE_S16LE
        assert self.create_spec(AudioFormat(2, 24, 44100)).format == self.pa.PA_SAMPLE_S24LE

    def test_int32(self):
        spec = self.create_spec(AudioFormat(2, 32, 44100, 's'))
        assert spec.format == self.pa.PA_SAMPLE_S32LE
        assert spec.rate == 44100
        assert spec.channels == 2

    def test_float32(self):
        spec = self.create_spec(AudioFormat(2, 32, 48000, 'f'))
        assert spec.format == self.pa.PA_SAMPLE_FLOAT32LE
        assert spec.rate == 48000

    def test_unsupported_raises(self):
        with pytest.raises(MediaException):
            self.create_spec(AudioFormat(2, 64, 44100, 'f'))


# ---------------------------------------------------------------------------
# FFmpeg decoding of 32 bit samples
# ---------------------------------------------------------------------------

ffmpeg = pytest.importorskip('pyglet.media.codecs.ffmpeg')


class TestFFmpegDecode32Bit:
    def decode_file(self, path):
        source = ffmpeg.FFmpegSource(str(path))
        data, timestamps, durations = decode_all(source)
        return source, data, timestamps, durations

    def test_int32_stereo(self, tmp_path):
        path = tmp_path / 'test_s32.wav'
        frames = interleave(S32_LEFT, S32_RIGHT)
        write_int_wav(path, 2, 32, SAMPLE_RATE, frames)

        source, data, timestamps, durations = self.decode_file(path)

        # Format description keeps integer semantics, channels and rate.
        assert source.audio_format.sample_size == 32
        assert source.audio_format.sample_type == 's'
        assert source.audio_format.channels == 2
        assert source.audio_format.sample_rate == SAMPLE_RATE

        # Sign, amplitude and interleaving are preserved exactly.
        decoded = struct.unpack(f'<{len(frames)}i', data)
        assert list(decoded) == frames
        assert list(decoded[0::2]) == S32_LEFT
        assert list(decoded[1::2]) == S32_RIGHT

        # Duration and timestamps.
        expected_duration = len(S32_LEFT) / SAMPLE_RATE
        assert abs(source.duration - expected_duration) < 0.01
        assert abs(sum(durations) - expected_duration) < 0.01
        assert timestamps[0] == pytest.approx(0.0, abs=0.01)
        assert list(timestamps) == sorted(timestamps)

    def test_float32_stereo(self, tmp_path):
        path = tmp_path / 'test_f32.wav'
        frames = interleave(F32_LEFT, F32_RIGHT)
        write_float_wav(path, 2, SAMPLE_RATE, frames)

        source, data, timestamps, durations = self.decode_file(path)

        # Format description keeps float semantics.
        assert source.audio_format.sample_size == 32
        assert source.audio_format.sample_type == 'f'
        assert source.audio_format.channels == 2
        assert source.audio_format.sample_rate == SAMPLE_RATE

        # Bit-exact float roundtrip, including sign of zero and tiny/huge
        # magnitudes.
        expected = struct.pack(f'<{len(frames)}f', *frames)
        assert data == expected

        expected_duration = len(F32_LEFT) / SAMPLE_RATE
        assert abs(source.duration - expected_duration) < 0.01
        assert abs(sum(durations) - expected_duration) < 0.01

    def test_float32_not_downgraded_to_16bit(self, tmp_path):
        # Regression: float sources used to be described as 16 bit and
        # converted to S16. Full-scale and tiny values must survive.
        path = tmp_path / 'test_f32_range.wav'
        frames = interleave([1.0, -1.0, 1e-7, -1e-7], [0.5, -0.5, 1e-7, -1e-7])
        write_float_wav(path, 2, SAMPLE_RATE, frames)

        source, data, _, _ = self.decode_file(path)
        assert source.audio_format.sample_type == 'f'
        # Bit-exact against the float32 representation of the input.
        assert data == struct.pack(f'<{len(frames)}f', *frames)

    def test_legacy_formats_still_decode(self, tmp_path):
        for sample_size, left, right in ((8, U8_LEFT, U8_RIGHT),
                                         (16, S16_LEFT, S16_RIGHT)):
            path = tmp_path / f'test_{sample_size}.wav'
            frames = interleave(left, right)
            write_int_wav(path, 2, sample_size, SAMPLE_RATE, frames)

            source, data, _, _ = self.decode_file(path)
            assert source.audio_format.sample_size == sample_size
            assert source.audio_format.sample_type == ('u' if sample_size == 8 else 's')

            fmt = 'B' if sample_size == 8 else 'h'
            decoded = struct.unpack(f'<{len(frames)}{fmt}', data)
            assert list(decoded) == frames

    def test_static_source_roundtrip(self, tmp_path):
        # Decode -> buffer in memory -> read back: data must be identical
        # to the streaming decode.
        path = tmp_path / 'test_static.wav'
        frames = interleave(S32_LEFT, S32_RIGHT)
        write_int_wav(path, 2, 32, SAMPLE_RATE, frames)

        streaming = ffmpeg.FFmpegSource(str(path))
        streamed, _, _ = decode_all(streaming)

        static = StaticSource(ffmpeg.FFmpegSource(str(path)))
        assert static.audio_format == streaming.audio_format
        memory_source = static.get_queue_source()
        buffered, _, _ = decode_all(memory_source)
        assert buffered == streamed

    def test_wave_encoder_roundtrip_int32(self, tmp_path):
        # Decode -> StaticSource -> save as wave -> reload: boundary
        # samples must survive the whole trip.
        path = tmp_path / 'test_enc_src.wav'
        frames = interleave(S32_LEFT, S32_RIGHT)
        write_int_wav(path, 2, 32, SAMPLE_RATE, frames)

        static = StaticSource(ffmpeg.FFmpegSource(str(path)))
        out = tmp_path / 'roundtrip.wav'
        static.get_queue_source().save(str(out))

        with wave.open(str(out), 'rb') as w:
            assert w.getnchannels() == 2
            assert w.getsampwidth() == 4
            assert w.getframerate() == SAMPLE_RATE
            data = w.readframes(w.getnframes())

        decoded = struct.unpack(f'<{len(frames)}i', data)
        assert list(decoded) == frames


# ---------------------------------------------------------------------------
# Decode-to-output roundtrip through the silent driver
# ---------------------------------------------------------------------------

class CountingSource(Source):
    """Wrap a source and count the bytes consumed by the audio driver."""

    def __init__(self, source):
        self._source = source.get_queue_source()
        self.audio_format = self._source.audio_format
        self.video_format = None
        self.bytes_consumed = 0

    def get_queue_source(self):
        return self

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        packet = self._source.get_audio_data(num_bytes, compensation_time)
        if packet is not None:
            self.bytes_consumed += packet.length
        return packet


@pytest.mark.parametrize('sample_type', ['s', 'f'])
def test_decode_to_output_roundtrip(tmp_path, sample_type):
    """Play a 32 bit source through the silent driver and verify the
    driver consumes exactly the decoded stream (correct frame size and
    duration accounting, no truncation or wrong bit depth)."""
    frames = interleave(S32_LEFT, S32_RIGHT)
    if sample_type == 'f':
        frames = interleave(F32_LEFT, F32_RIGHT)
        # Repeat the boundary pattern to get a playable duration.
        frames = frames * 4000
        write_float_wav(tmp_path / 'src.wav', 2, SAMPLE_RATE, frames)
    else:
        frames = frames * 4000
        write_int_wav(tmp_path / 'src.wav', 2, 32, SAMPLE_RATE, frames)

    static = StaticSource(ffmpeg.FFmpegSource(str(tmp_path / 'src.wav')))
    expected_bytes = static.audio_format.bytes_per_frame * (len(frames) // 2)

    from pyglet.media.drivers.silent import create_audio_driver
    from pyglet.media.player import Player

    driver = create_audio_driver()
    try:
        with mock.patch('pyglet.media.player.get_audio_driver', return_value=driver):
            player = Player()
            counting = CountingSource(static)
            player.queue(counting)
            player.play()

            deadline = time.perf_counter() + 10.0
            while player.time < static.duration and time.perf_counter() < deadline:
                time.sleep(0.01)

            player.pause()
            player.delete()

        assert counting.bytes_consumed == expected_bytes
    finally:
        driver.delete()
