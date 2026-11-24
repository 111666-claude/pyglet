"""Uniform contract tests for ``pyglet.graphics.shader.ShaderProgram``.

These tests pin down the contract between pyglet's uniform type mapping
(``_uniform_setters`` / ``_uniform_getters``) and the OpenGL calls that a
``ShaderProgram`` emits, *without requiring a GPU or a GL context*.

How it works
------------
``FakeGLDriver`` is a scriptable, recording stand-in for the OpenGL driver.
It is installed via ``monkeypatch`` over the replaceable GL entry points used
by ``pyglet.graphics.shader``:

* introspection queries (``glGetProgramiv``, ``glGetActiveUniform``,
  ``glGetUniformLocation``, ``glGetActiveUniformBlock*``, ...) are answered
  from a scripted list of uniforms / uniform blocks;
* every uniform setter/getter referenced by ``_uniform_setters`` /
  ``_uniform_getters`` is wrapped in a recorder that decodes the ctypes
  upload buffer, so tests can verify *what* data would reach the GPU, in
  *which* call, in *which* order.

The mapping itself (which GL enum -> which setter function, ctypes type and
element count) always comes from the real ``pyglet.graphics.shader`` code —
only the GL entry points are replaced.

All tests exercise the public ``ShaderProgram`` API
(``program['name'] = value``, ``program['name']``, ``program.uniforms``,
``program.uniform_blocks['Block'].create_ubo()`` ...). Each behavioural test
runs twice: once for the legacy ``glUniform*`` path and once for the DSA
``glProgramUniform*`` path.

Running
-------
Local:   ``python -m pytest tests/unit/shader/test_uniform_contract.py -v``
CI:      picked up automatically by the ``PyTest`` workflow
         (``.github/workflows/unittests.yml`` runs ``pytest tests``).

See ``tests/unit/shader/README.md`` for a representative failure report
showing what a broken mapping looks like when caught by this suite.
"""
from __future__ import annotations

import ctypes
import itertools
import re
from _ctypes import _Pointer
from types import SimpleNamespace
from typing import NamedTuple

import pytest

import pyglet
from pyglet.gl import gl
from pyglet.graphics import shader as shader_module
from pyglet.graphics.shader import ShaderException, ShaderProgram

# ---------------------------------------------------------------------------
# Independent contract specification.
#
# This table is written *independently* of pyglet.graphics.shader._uniform_setters
# (from the OpenGL specification: matN = N*N floats, vecN = N components,
# samplers/images are integer texture-unit/image-unit handles, ...).
# If the two disagree, the contract test fails.
# ---------------------------------------------------------------------------

_SAMPLER_AND_IMAGE_TYPES = [
    # 1D samplers
    'GL_SAMPLER_1D', 'GL_SAMPLER_1D_ARRAY',
    'GL_INT_SAMPLER_1D', 'GL_INT_SAMPLER_1D_ARRAY',
    'GL_UNSIGNED_INT_SAMPLER_1D', 'GL_UNSIGNED_INT_SAMPLER_1D_ARRAY',
    # 2D samplers
    'GL_SAMPLER_2D', 'GL_SAMPLER_2D_ARRAY',
    'GL_INT_SAMPLER_2D', 'GL_INT_SAMPLER_2D_ARRAY',
    'GL_UNSIGNED_INT_SAMPLER_2D', 'GL_UNSIGNED_INT_SAMPLER_2D_ARRAY',
    # Multisample samplers
    'GL_SAMPLER_2D_MULTISAMPLE',
    'GL_INT_SAMPLER_2D_MULTISAMPLE',
    'GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE',
    # Cube samplers
    'GL_SAMPLER_CUBE', 'GL_INT_SAMPLER_CUBE', 'GL_UNSIGNED_INT_SAMPLER_CUBE',
    'GL_SAMPLER_CUBE_MAP_ARRAY', 'GL_INT_SAMPLER_CUBE_MAP_ARRAY',
    'GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY',
    # 3D samplers
    'GL_SAMPLER_3D', 'GL_INT_SAMPLER_3D', 'GL_UNSIGNED_INT_SAMPLER_3D',
    # Images
    'GL_IMAGE_1D', 'GL_IMAGE_2D', 'GL_IMAGE_2D_RECT', 'GL_IMAGE_3D',
    'GL_IMAGE_1D_ARRAY', 'GL_IMAGE_2D_ARRAY',
    'GL_IMAGE_2D_MULTISAMPLE', 'GL_IMAGE_2D_MULTISAMPLE_ARRAY',
    'GL_IMAGE_BUFFER', 'GL_IMAGE_CUBE', 'GL_IMAGE_CUBE_MAP_ARRAY',
]

# enum name -> (ctypes type name, legacy setter, DSA setter, element count)
EXPECTED_UNIFORM_SETTERS = {
    'GL_BOOL':           ('GLint', 'glUniform1iv', 'glProgramUniform1iv', 1),
    'GL_BOOL_VEC2':      ('GLint', 'glUniform1iv', 'glProgramUniform1iv', 2),
    'GL_BOOL_VEC3':      ('GLint', 'glUniform1iv', 'glProgramUniform1iv', 3),
    'GL_BOOL_VEC4':      ('GLint', 'glUniform1iv', 'glProgramUniform1iv', 4),

    'GL_INT':            ('GLint', 'glUniform1iv', 'glProgramUniform1iv', 1),
    'GL_INT_VEC2':       ('GLint', 'glUniform2iv', 'glProgramUniform2iv', 2),
    'GL_INT_VEC3':       ('GLint', 'glUniform3iv', 'glProgramUniform3iv', 3),
    'GL_INT_VEC4':       ('GLint', 'glUniform4iv', 'glProgramUniform4iv', 4),

    'GL_FLOAT':          ('GLfloat', 'glUniform1fv', 'glProgramUniform1fv', 1),
    'GL_FLOAT_VEC2':     ('GLfloat', 'glUniform2fv', 'glProgramUniform2fv', 2),
    'GL_FLOAT_VEC3':     ('GLfloat', 'glUniform3fv', 'glProgramUniform3fv', 3),
    'GL_FLOAT_VEC4':     ('GLfloat', 'glUniform4fv', 'glProgramUniform4fv', 4),

    'GL_FLOAT_MAT2':     ('GLfloat', 'glUniformMatrix2fv', 'glProgramUniformMatrix2fv', 4),
    'GL_FLOAT_MAT3':     ('GLfloat', 'glUniformMatrix3fv', 'glProgramUniformMatrix3fv', 9),
    'GL_FLOAT_MAT4':     ('GLfloat', 'glUniformMatrix4fv', 'glProgramUniformMatrix4fv', 16),
}
for _name in _SAMPLER_AND_IMAGE_TYPES:
    # Samplers and images are bound to integer texture/image units.
    EXPECTED_UNIFORM_SETTERS[_name] = ('GLint', 'glUniform1iv', 'glProgramUniform1iv', 1)

EXPECTED_BY_ENUM = {getattr(gl, name): spec for name, spec in EXPECTED_UNIFORM_SETTERS.items()}


def find_mapping_violations(mapping, getters=None):
    """Compare a ``_uniform_setters``-style mapping against the independent spec.

    Returns a list of human-readable violations; an empty list means the
    mapping honours the contract.
    """
    violations = []
    for enum_name, (ctype_name, legacy_name, dsa_name, length) in EXPECTED_UNIFORM_SETTERS.items():
        enum = getattr(gl, enum_name)
        entry = mapping.get(enum)
        if entry is None:
            violations.append(f"{enum_name}: missing from _uniform_setters")
            continue
        gl_type, legacy_fn, dsa_fn, actual_length = entry
        if gl_type is not getattr(gl, ctype_name):
            violations.append(
                f"{enum_name}: ctypes type is {gl_type.__name__}, expected {ctype_name}")
        if legacy_fn.__name__ != legacy_name:
            violations.append(
                f"{enum_name}: legacy setter is {legacy_fn.__name__}, expected {legacy_name}")
        if dsa_fn.__name__ != dsa_name:
            violations.append(
                f"{enum_name}: DSA setter is {dsa_fn.__name__}, expected {dsa_name}")
        if actual_length != length:
            violations.append(
                f"{enum_name}: length is {actual_length}, expected {length}")
        if getters is not None and gl_type not in getters:
            violations.append(
                f"{enum_name}: no getter registered in _uniform_getters for {gl_type.__name__}")

    for enum in mapping:
        if enum not in EXPECTED_BY_ENUM:
            violations.append(
                f"GL enum 0x{enum:04x}: present in _uniform_setters but not covered by the contract spec")
    return violations


def setter_name_component_count(fn_name):
    """Element count implied by a glUniform* function name, or None if not applicable.

    A '1'-suffixed setter (glUniform1iv / glUniform1fv) says nothing about the
    component count: pyglet deliberately uses it for bool vectors, samplers,
    images and array uploads of any length. Only suffixes > 1 and MatrixN
    names pin down the element count.
    """
    match = re.fullmatch(r'gl(?:Program)?UniformMatrix(\d)fv', fn_name)
    if match:  # e.g. glUniformMatrix3fv -> 9
        return int(match.group(1)) ** 2
    match = re.fullmatch(r'gl(?:Program)?Uniform([2-4])[fi]v', fn_name)
    if match:  # e.g. glUniform3fv -> 3
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Recording fake GL driver
# ---------------------------------------------------------------------------

class Upload(NamedTuple):
    """One decoded uniform upload call."""
    fn: str                 # e.g. 'glUniformMatrix4fv' / 'glProgramUniform1iv'
    program: int | None     # program id for DSA setters, None for legacy
    location: int
    count: int
    transpose: int | None   # matrix setters only
    values: tuple           # decoded contents of the ctypes upload buffer


class FakeGLDriver:
    """A scriptable, recording stand-in for the OpenGL driver.

    uniforms: iterable of dicts with keys
        ``name`` (as glGetActiveUniform reports it, e.g. 'weights[0]'),
        ``type`` (GL enum), ``size`` (array size, 1 for scalars),
        ``location`` (int, or None for uniform-block members).
    uniform_blocks: iterable of dicts with keys
        ``name``, ``members`` (global uniform indices), ``data_size``,
        ``binding`` and ``offsets`` ({global uniform index: byte offset}).
    """

    program_id = 77

    def __init__(self, uniforms=(), uniform_blocks=(), dsa=False, max_bindings=36):
        self.dsa = dsa
        self.max_bindings = max_bindings
        self.calls = []                 # mixed log: Upload tuples and (name, *args) tuples
        self._uniforms = list(uniforms)
        self._blocks = list(uniform_blocks)
        self._locations = {}            # uniform query name -> location
        self._components = {}           # location -> elements per single 'count'
        self._element_to_base = {}      # array element location -> (base location, index)
        self.state = {}                 # location -> current uniform values (getter source)
        self._offsets = {}
        self._buffer_ids = itertools.count(1)

        for spec in self._uniforms:
            location = spec.get('location')
            if location is None:  # uniform block member: no location
                continue
            length = EXPECTED_BY_ENUM[spec['type']][3]
            name, size = spec['name'], spec['size']
            self._locations[name] = location
            base = name[:-3] if name.endswith('[0]') else name
            for i in range(size):
                element_location = location + i
                if i:
                    self._locations[f'{base}[{i}]'] = element_location
                    self._element_to_base[element_location] = (location, i)
                self._components[element_location] = length
                self.state[element_location] = [0] * length
            # The base location holds the whole array for array getters.
            self.state[location] = [0] * (length * size)

        for block in self._blocks:
            self._offsets.update(block['offsets'])

        self.context = SimpleNamespace(
            delete_buffer=lambda *args: None,
            delete_shader=lambda args: None,
            delete_shader_program=lambda *args: None,
        )

    # -- installation -------------------------------------------------------

    def install(self, monkeypatch):
        """Replace the GL entry points used by pyglet.graphics.shader with this driver."""
        sm = 'pyglet.graphics.shader'
        monkeypatch.setattr(f'{sm}._link_program', lambda *shaders: self.program_id)
        monkeypatch.setattr(f'{sm}._introspect_attributes', lambda program_id: {})
        monkeypatch.setattr(f'{sm}.glGetProgramiv', self._glGetProgramiv)
        monkeypatch.setattr(f'{sm}.glUseProgram', self._glUseProgram)
        monkeypatch.setattr(f'{sm}.glBindBufferBase', self._glBindBufferBase)
        monkeypatch.setattr(f'{sm}.gl_info', SimpleNamespace(
            have_version=lambda *args: self.dsa,
            have_extension=lambda *args: False))
        monkeypatch.setattr(f'{sm}._uniform_setters', self._recording_setters())
        monkeypatch.setattr(f'{sm}._uniform_getters', self._recording_getters())

        monkeypatch.setattr('pyglet.gl.current_context', self.context)
        glm = 'pyglet.gl.gl'
        monkeypatch.setattr(f'{glm}.glGetActiveUniform', self._glGetActiveUniform)
        monkeypatch.setattr(f'{glm}.glGetUniformLocation', self._glGetUniformLocation)
        monkeypatch.setattr(f'{glm}.glGetActiveUniformBlockName', self._glGetActiveUniformBlockName)
        monkeypatch.setattr(f'{glm}.glGetActiveUniformBlockiv', self._glGetActiveUniformBlockiv)
        monkeypatch.setattr(f'{glm}.glGetActiveUniformsiv', self._glGetActiveUniformsiv)
        monkeypatch.setattr(f'{glm}.glUniformBlockBinding', self._glUniformBlockBinding)
        monkeypatch.setattr(f'{glm}.glGetIntegerv', self._glGetIntegerv)

        vb = 'pyglet.graphics.vertexbuffer'
        monkeypatch.setattr(f'{vb}.glGenBuffers', self._glGenBuffers)
        monkeypatch.setattr(f'{vb}.glBindBuffer', self._glBindBuffer)
        monkeypatch.setattr(f'{vb}.glBufferData', self._glBufferData)

    # -- setter/getter recorders ---------------------------------------------

    def _recording_setters(self):
        """Wrap every setter in the real mapping; the mapping decisions stay under test."""
        wrapped = {}
        for enum, (gl_type, legacy_fn, dsa_fn, length) in shader_module._uniform_setters.items():
            wrapped[enum] = (gl_type,
                             self._wrap_setter(legacy_fn.__name__),
                             self._wrap_setter(dsa_fn.__name__),
                             length)
        return wrapped

    def _wrap_setter(self, fn_name):
        def recording_setter(*args):
            self._record_upload(fn_name, args)
        recording_setter.__name__ = fn_name
        return recording_setter

    def _recording_getters(self):
        return {gl_type: self._wrap_getter(fn.__name__)
                for gl_type, fn in shader_module._uniform_getters.items()}

    def _wrap_getter(self, fn_name):
        def recording_getter(program, location, ptr):
            self.calls.append((fn_name, program, location))
            for i, value in enumerate(self.state[location]):
                ptr[i] = value
        recording_getter.__name__ = fn_name
        return recording_getter

    def _record_upload(self, fn_name, args):
        data = args[-1]
        if 'Matrix' in fn_name:
            head, count, transpose = args[:-3], args[-3], args[-2]
        else:
            head, count, transpose = args[:-2], args[-2], None
        location = head[-1]
        program = head[-2] if len(head) > 1 else None

        n = count * self._components[location]
        if isinstance(data, (ctypes.Array, _Pointer)):
            values = tuple(data[:n])
        else:  # a bare ctypes scalar (single array-element uploads)
            values = (data.value,)

        self.state[location][:len(values)] = values
        if location in self._element_to_base:  # keep the full-array view in sync
            base, index = self._element_to_base[location]
            width = self._components[location]
            self.state[base][index * width:(index + 1) * width] = values

        self.calls.append(Upload(fn_name, program, location, count, transpose, values))

    # -- introspection fakes ---------------------------------------------------

    def _glGetProgramiv(self, program, pname, out):
        target = getattr(out, '_obj', out)
        if pname == gl.GL_ACTIVE_UNIFORMS:
            target.value = len(self._uniforms)
        elif pname == gl.GL_ACTIVE_UNIFORM_BLOCKS:
            target.value = len(self._blocks)
        else:
            raise AssertionError(f"FakeGLDriver: unexpected glGetProgramiv pname {pname}")

    def _glGetActiveUniform(self, program, index, buf_size, length, size, type_, name_buf):
        spec = self._uniforms[index]
        size.value = spec['size']
        type_.value = spec['type']
        name_buf.value = spec['name'].encode()

    def _glGetUniformLocation(self, program, name_buf):
        name = name_buf.value.decode()
        self.calls.append(('glGetUniformLocation', name))
        return self._locations.get(name, -1)

    def _glGetActiveUniformBlockName(self, program, index, buf_size, size, name_buf):
        name = self._blocks[index]['name']
        name_buf.value = name.encode()
        size.value = len(name)

    def _glGetActiveUniformBlockiv(self, program, index, pname, out):
        block = self._blocks[index]
        if pname == gl.GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS:
            out.value = len(block['members'])
        elif pname == gl.GL_UNIFORM_BLOCK_DATA_SIZE:
            out.value = block['data_size']
        elif pname == gl.GL_UNIFORM_BLOCK_BINDING:
            out.value = block['binding']
        elif pname == gl.GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES:
            for i, member in enumerate(block['members']):
                out[i] = member
        else:
            raise AssertionError(f"FakeGLDriver: unexpected glGetActiveUniformBlockiv pname {pname}")

    def _glGetActiveUniformsiv(self, program, count, indices, pname, out):
        if pname != gl.GL_UNIFORM_OFFSET:
            raise AssertionError(f"FakeGLDriver: unexpected glGetActiveUniformsiv pname {pname}")
        for i in range(count):
            out[i] = self._offsets[indices[i]]

    def _glGetIntegerv(self, pname, out):
        if pname != gl.GL_MAX_UNIFORM_BUFFER_BINDINGS:
            raise AssertionError(f"FakeGLDriver: unexpected glGetIntegerv pname {pname}")
        target = getattr(out, '_obj', out)
        target.value = self.max_bindings

    # -- misc recorded calls ---------------------------------------------------

    def _glUseProgram(self, program):
        self.calls.append(('glUseProgram', program))

    def _glUniformBlockBinding(self, program, index, binding):
        self.calls.append(('glUniformBlockBinding', program, index, binding))

    def _glBindBufferBase(self, target, index, buffer_id):
        self.calls.append(('glBindBufferBase', target, index, buffer_id))

    def _glGenBuffers(self, n, ids):
        ids.value = next(self._buffer_ids)

    def _glBindBuffer(self, target, buffer_id):
        self.calls.append(('glBindBuffer', target, buffer_id))

    def _glBufferData(self, target, size, data, usage):
        self.calls.append(('glBufferData', target, size, usage))

    # -- assertions helpers ------------------------------------------------------

    def uploads(self):
        """All recorded uniform uploads, in call order."""
        return [call for call in self.calls if isinstance(call, Upload)]


@pytest.fixture(params=[False, True], ids=['legacy-glUniform', 'dsa-glProgramUniform'])
def gl_driver(request, monkeypatch):
    """Factory fixture: build and install a FakeGLDriver for the scripted GL state.

    Every test using this fixture runs twice: once with the legacy glUniform*
    entry points and once with the DSA glProgramUniform* entry points.
    """
    def factory(*, uniforms=(), uniform_blocks=()):
        driver = FakeGLDriver(uniforms=uniforms, uniform_blocks=uniform_blocks, dsa=request.param)
        driver.install(monkeypatch)
        return driver
    return factory


def make_program(driver):
    """Create a ShaderProgram through its public constructor, against the fake driver."""
    return ShaderProgram(SimpleNamespace(id=1), SimpleNamespace(id=2))


def expected_setter_name(driver, legacy_name):
    """The setter name expected for the current driver mode."""
    return legacy_name.replace('glUniform', 'glProgramUniform') if driver.dsa else legacy_name


# ---------------------------------------------------------------------------
# 1. Static mapping contract (no GL needed at all)
# ---------------------------------------------------------------------------

class TestUniformMappingContract:
    """The _uniform_setters table must match the independent contract spec."""

    def test_mapping_matches_contract_spec(self):
        violations = find_mapping_violations(shader_module._uniform_setters,
                                             getters=shader_module._uniform_getters)
        assert violations == [], "\n".join(violations)

    def test_setter_name_suffix_consistent_with_length(self):
        """The numeric suffix of each setter name must agree with the mapped length."""
        violations = []
        for enum, (gl_type, legacy_fn, dsa_fn, length) in shader_module._uniform_setters.items():
            for fn in (legacy_fn, dsa_fn):
                implied = setter_name_component_count(fn.__name__)
                if implied is not None and implied != length:
                    violations.append(
                        f"GL enum 0x{enum:04x}: {fn.__name__} implies {implied} elements, "
                        f"but mapped length is {length}")
        assert violations == [], "\n".join(violations)

    def test_sabotaged_length_is_detected(self):
        """Proof of protection: a deliberately wrong element count must fail the contract."""
        corrupted = dict(shader_module._uniform_setters)
        gl_type, legacy_fn, dsa_fn, _ = corrupted[gl.GL_FLOAT_MAT4]
        corrupted[gl.GL_FLOAT_MAT4] = (gl_type, legacy_fn, dsa_fn, 15)  # mat4 is 16 floats

        violations = find_mapping_violations(corrupted)
        assert any('GL_FLOAT_MAT4' in v and 'length' in v for v in violations), violations

    def test_sabotaged_setter_signature_is_detected(self):
        """Proof of protection: a deliberately swapped setter must fail the contract."""
        corrupted = dict(shader_module._uniform_setters)
        gl_type, _, dsa_fn, length = corrupted[gl.GL_FLOAT_VEC3]
        corrupted[gl.GL_FLOAT_VEC3] = (gl_type, gl.glUniform2fv, dsa_fn, length)

        violations = find_mapping_violations(corrupted)
        assert any('GL_FLOAT_VEC3' in v and 'legacy setter' in v for v in violations), violations


# ---------------------------------------------------------------------------
# 2. Scalars
# ---------------------------------------------------------------------------

class TestScalarUniforms:

    def test_float_scalar_upload_and_readback(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'opacity', 'type': gl.GL_FLOAT, 'size': 1, 'location': 3}])
        program = make_program(driver)

        assert program.uniforms['opacity'] == {'location': 3, 'length': 1, 'size': 1}

        program['opacity'] = 0.5
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform1fv')
        assert upload.location == 3
        assert upload.count == 1
        assert upload.values == (0.5,)

        # Getter reads back what was uploaded.
        assert program['opacity'] == 0.5

    def test_int_scalar_uploads_as_int(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'count', 'type': gl.GL_INT, 'size': 1, 'location': 0}])
        program = make_program(driver)

        program['count'] = 7
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform1iv')
        assert upload.values == (7,)
        assert isinstance(upload.values[0], int)
        assert program['count'] == 7

    def test_bool_scalar_uploads_as_int(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'enabled', 'type': gl.GL_BOOL, 'size': 1, 'location': 1}])
        program = make_program(driver)

        program['enabled'] = True
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform1iv')
        assert upload.values == (1,)
        assert program['enabled'] == 1

    def test_call_order_and_program_binding(self, gl_driver):
        """Legacy path must glUseProgram first; DSA path must pass the program id."""
        driver = gl_driver(uniforms=[
            {'name': 'opacity', 'type': gl.GL_FLOAT, 'size': 1, 'location': 0}])
        program = make_program(driver)

        program['opacity'] = 0.25
        (upload,) = driver.uploads()
        if driver.dsa:
            assert upload.program == driver.program_id
            assert ('glUseProgram', driver.program_id) not in driver.calls
        else:
            assert upload.program is None
            index = driver.calls.index(upload)
            assert driver.calls[index - 1] == ('glUseProgram', driver.program_id)


# ---------------------------------------------------------------------------
# 3. Vectors
# ---------------------------------------------------------------------------

class TestVectorUniforms:

    def test_float_vec3(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'color', 'type': gl.GL_FLOAT_VEC3, 'size': 1, 'location': 2}])
        program = make_program(driver)

        assert program.uniforms['color'] == {'location': 2, 'length': 3, 'size': 1}

        program['color'] = (1.0, 0.5, 0.25)
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform3fv')
        assert upload.count == 1
        assert upload.values == (1.0, 0.5, 0.25)
        assert program['color'] == [1.0, 0.5, 0.25]

    def test_int_vec4(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'rect', 'type': gl.GL_INT_VEC4, 'size': 1, 'location': 0}])
        program = make_program(driver)

        program['rect'] = (1, 2, 3, 4)
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform4iv')
        assert upload.values == (1, 2, 3, 4)
        assert all(isinstance(v, int) for v in upload.values)

    def test_bool_vec2(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'flags', 'type': gl.GL_BOOL_VEC2, 'size': 1, 'location': 0}])
        program = make_program(driver)

        program['flags'] = (True, False)
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform1iv')
        assert upload.values == (1, 0)

    def test_wrong_component_count_rejected(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'color', 'type': gl.GL_FLOAT_VEC3, 'size': 1, 'location': 0}])
        program = make_program(driver)

        with pytest.raises(ValueError):
            program['color'] = (1.0, 2.0)  # vec3 requires exactly 3 components
        assert driver.uploads() == []


# ---------------------------------------------------------------------------
# 4. Matrices
# ---------------------------------------------------------------------------

class TestMatrixUniforms:

    @pytest.mark.parametrize('enum_name, matrix_fn, count', [
        ('GL_FLOAT_MAT2', 'Matrix2', 4),
        ('GL_FLOAT_MAT3', 'Matrix3', 9),
        ('GL_FLOAT_MAT4', 'Matrix4', 16),
    ])
    def test_matrix_upload(self, gl_driver, enum_name, matrix_fn, count):
        values = tuple(float(i) for i in range(1, count + 1))
        driver = gl_driver(uniforms=[
            {'name': 'mat', 'type': getattr(gl, enum_name), 'size': 1, 'location': 2}])
        program = make_program(driver)

        assert program.uniforms['mat'] == {'location': 2, 'length': count, 'size': 1}

        program['mat'] = values
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, f'glUniform{matrix_fn}fv')
        assert upload.count == 1
        assert upload.transpose == 0  # GL_FALSE: pyglet never transposes
        # The full element count is uploaded, in the caller's order.
        assert upload.values == values

        assert program['mat'] == list(values)


# ---------------------------------------------------------------------------
# 5. Uniform arrays
# ---------------------------------------------------------------------------

class TestUniformArrays:

    def test_scalar_array_full_upload(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'weights[0]', 'type': gl.GL_FLOAT, 'size': 3, 'location': 5}])
        program = make_program(driver)

        assert 'weights' in program.uniforms
        assert program.uniforms['weights'] == {'location': 5, 'length': 1, 'size': 3}

        program['weights'] = [1.0, 2.0, 3.0]
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform1fv')
        assert upload.location == 5
        assert upload.count == 3  # the whole array in one call
        assert upload.values == (1.0, 2.0, 3.0)

    def test_array_element_upload_uses_element_location(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'weights[0]', 'type': gl.GL_FLOAT, 'size': 3, 'location': 5}])
        program = make_program(driver)

        program['weights'][2] = 9.0
        (upload,) = driver.uploads()
        assert upload.count == 1
        assert upload.location == 7  # location of 'weights[2]'
        assert upload.values == (9.0,)
        # The driver was asked for the *element* location, not the base one.
        assert ('glGetUniformLocation', 'weights[2]') in driver.calls

    def test_array_len_getitem_and_slice(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'weights[0]', 'type': gl.GL_FLOAT, 'size': 3, 'location': 5}])
        program = make_program(driver)
        program['weights'] = [1.0, 2.0, 3.0]

        array = program['weights']
        assert len(array) == 3
        assert array[1] == 2.0
        assert array[0:2] == (1.0, 2.0)

    def test_vector_array(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'palette[0]', 'type': gl.GL_FLOAT_VEC3, 'size': 2, 'location': 10}])
        program = make_program(driver)

        # Multi-component arrays are set as one sequence per array element.
        program['palette'] = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform3fv')
        assert upload.count == 2
        assert upload.values == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        program['palette'][1] = (7.0, 8.0, 9.0)
        element_upload = driver.uploads()[-1]
        assert element_upload.location == 11
        assert element_upload.count == 1
        assert element_upload.values == (7.0, 8.0, 9.0)

        assert program['palette'][0] == (1.0, 2.0, 3.0)

    def test_array_set_with_wrong_size_rejected(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'weights[0]', 'type': gl.GL_FLOAT, 'size': 3, 'location': 5}])
        program = make_program(driver)

        with pytest.raises(AssertionError):
            program['weights'] = [1.0, 2.0]  # array has 3 elements
        assert driver.uploads() == []

    def test_array_delete_rejected(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'weights[0]', 'type': gl.GL_FLOAT, 'size': 3, 'location': 5}])
        program = make_program(driver)

        with pytest.raises(ShaderException):
            del program['weights'][0]

    def test_array_out_of_range_rejected(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'weights[0]', 'type': gl.GL_FLOAT, 'size': 3, 'location': 5}])
        program = make_program(driver)

        with pytest.raises(ShaderException):
            program['weights'][9]


# ---------------------------------------------------------------------------
# 6. Samplers and images (integer unit handles)
# ---------------------------------------------------------------------------

class TestSamplersAndImages:

    @pytest.mark.parametrize('enum_name', [
        'GL_SAMPLER_2D', 'GL_INT_SAMPLER_2D', 'GL_SAMPLER_3D',
        'GL_SAMPLER_CUBE_MAP_ARRAY', 'GL_IMAGE_2D', 'GL_IMAGE_2D_MULTISAMPLE_ARRAY',
    ])
    def test_unit_uploads_use_integer_setter(self, gl_driver, enum_name):
        driver = gl_driver(uniforms=[
            {'name': 'tex', 'type': getattr(gl, enum_name), 'size': 1, 'location': 0}])
        program = make_program(driver)

        program['tex'] = 4  # texture/image unit
        (upload,) = driver.uploads()
        assert upload.fn == expected_setter_name(driver, 'glUniform1iv')
        assert upload.count == 1
        assert upload.values == (4,)
        assert isinstance(upload.values[0], int)


# ---------------------------------------------------------------------------
# 7. Uniform blocks
# ---------------------------------------------------------------------------

def block_test_uniforms():
    return [
        {'name': 'TestBlock.density', 'type': gl.GL_FLOAT, 'size': 1, 'location': None},
        {'name': 'TestBlock.tint', 'type': gl.GL_FLOAT_VEC3, 'size': 1, 'location': None},
        {'name': 'TestBlock.weights[0]', 'type': gl.GL_FLOAT, 'size': 2, 'location': None},
    ]


def block_test_blocks(binding=0):
    # std140-style layout: float @0, vec3 @16, float[2] @32, block size 64.
    return [{'name': 'TestBlock', 'members': [0, 1, 2], 'data_size': 64,
             'binding': binding, 'offsets': {0: 0, 1: 16, 2: 32}}]


class TestUniformBlocks:

    def test_block_introspection_metadata(self, gl_driver):
        driver = gl_driver(uniforms=block_test_uniforms(), uniform_blocks=block_test_blocks())
        program = make_program(driver)

        block = program.uniform_blocks['TestBlock']
        assert block.size == 64
        assert block.uniform_count == 3

        # (name, ctypes type, element length, array size) per member.
        # A scalar float must stay a scalar: length 1, size 1, plain c_float.
        assert block.uniforms[0] == ('density', ctypes.c_float, 1, 1)
        assert block.uniforms[1] == ('tint', ctypes.c_float, 3, 1)
        assert block.uniforms[2] == ('weights[0]', ctypes.c_float, 1, 2)

        # Block members must not leak into the regular uniform namespace.
        assert 'density' not in program.uniforms
        assert 'TestBlock.density' not in program.uniforms

    def test_block_binding_auto_assigned(self, gl_driver):
        driver = gl_driver(uniforms=block_test_uniforms(), uniform_blocks=block_test_blocks(binding=0))
        program = make_program(driver)

        block = program.uniform_blocks['TestBlock']
        assert block.binding == 1  # 0 is reserved for pyglet's WindowBlock
        assert ('glUniformBlockBinding', driver.program_id, 0, 1) in driver.calls

    def test_block_explicit_binding_respected(self, gl_driver):
        driver = gl_driver(uniforms=block_test_uniforms(), uniform_blocks=block_test_blocks(binding=3))
        program = make_program(driver)

        block = program.uniform_blocks['TestBlock']
        assert block.binding == 3
        # No reassignment may happen for explicitly bound blocks.
        assert not [c for c in driver.calls
                    if isinstance(c, tuple) and c[0] == 'glUniformBlockBinding']

    def test_create_ubo_view_layout(self, gl_driver):
        driver = gl_driver(uniforms=block_test_uniforms(), uniform_blocks=block_test_blocks())
        program = make_program(driver)
        block = program.uniform_blocks['TestBlock']

        ubo = block.create_ubo()
        view_cls = type(ubo.view)
        fields = dict(view_cls._fields_)

        # Scalar members map to plain ctypes types, vectors/arrays to ctypes arrays.
        assert fields['density'] is ctypes.c_float
        assert fields['tint'] == ctypes.c_float * 3
        assert fields['weights'] == ctypes.c_float * 2

        # Offsets follow the std140 layout reported by the driver.
        assert view_cls.density.offset == 0
        assert view_cls.tint.offset == 16
        assert view_cls.weights.offset == 32
        assert ctypes.sizeof(ubo.view) == block.size == 64

    def test_ubo_write_flushes_buffer_to_binding_point(self, gl_driver):
        driver = gl_driver(uniforms=block_test_uniforms(), uniform_blocks=block_test_blocks())
        program = make_program(driver)
        block = program.uniform_blocks['TestBlock']
        ubo = block.create_ubo()

        with ubo as view:
            view.density = 0.5

        # On exit the view is uploaded to a buffer of exactly the block size,
        # bound to the block's binding point.
        assert ('glBindBufferBase', gl.GL_UNIFORM_BUFFER, block.binding, ubo.buffer.id) in driver.calls
        buffer_uploads = [c for c in driver.calls
                          if isinstance(c, tuple) and c[0] == 'glBufferData']
        assert buffer_uploads, "UBO writes must reach glBufferData"
        assert buffer_uploads[-1][2] == 64  # exact block byte size

    def test_set_binding_validation(self, gl_driver):
        driver = gl_driver(uniforms=block_test_uniforms(), uniform_blocks=block_test_blocks())
        program = make_program(driver)
        block = program.uniform_blocks['TestBlock']

        with pytest.raises(AssertionError):
            block.set_binding(0)  # reserved for pyglet's WindowBlock
        with pytest.raises(ShaderException):
            block.set_binding(driver.max_bindings)  # beyond hardware limit

        block.set_binding(5)
        assert block.binding == 5
        assert ('glUniformBlockBinding', driver.program_id, 0, 5) in driver.calls


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------

class TestUniformErrors:

    def test_unknown_uniform_set_raises(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'opacity', 'type': gl.GL_FLOAT, 'size': 1, 'location': 0}])
        program = make_program(driver)

        with pytest.raises(ShaderException):
            program['does_not_exist'] = 1.0
        assert driver.uploads() == []

    def test_unknown_uniform_get_raises(self, gl_driver):
        driver = gl_driver(uniforms=[
            {'name': 'opacity', 'type': gl.GL_FLOAT, 'size': 1, 'location': 0}])
        program = make_program(driver)

        with pytest.raises(ShaderException):
            program['does_not_exist']
