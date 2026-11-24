# Uniform contract tests (no GPU required)

`test_uniform_contract.py` pins down the contract between pyglet's uniform
type mapping (`pyglet.graphics.shader._uniform_setters` / `_uniform_getters`)
and the OpenGL calls a `ShaderProgram` actually emits. It runs in any
environment — no GPU, GL context, or window is needed.

## What is covered

* **Static mapping contract** — every mapped GL enum (scalars, vectors,
  matrices, samplers, images) is checked against an independently written
  spec: ctypes type, legacy `glUniform*` setter, DSA `glProgramUniform*`
  setter, and element count (`matN` = N², `vecN` = N, sampler/image = 1
  integer unit handle). Setter-name suffixes are cross-checked against the
  mapped lengths.
* **Behavioural contract** — a scriptable, recording `FakeGLDriver` replaces
  the GL entry points via `monkeypatch`. Tests drive the **public**
  `ShaderProgram` API (`program['name'] = value`, `program['name']`,
  `program.uniforms`, `program.uniform_blocks['B'].create_ubo()`, …) and
  assert on the decoded GL call stream: which function, which location,
  which count, and the exact bytes of the ctypes upload buffer, in order.
  Every behavioural test runs twice: legacy `glUniform*` and DSA
  `glProgramUniform*`.
* **Scalars / vectors / matrices / arrays / samplers / uniform blocks** —
  including per-element array uploads (location of `name[i]`, count 1),
  full-array uploads (count = array size), `glUseProgram` ordering on the
  legacy path, program id on the DSA path, std140 block view layout
  (field types, offsets, total size), binding-point assignment, and UBO
  flushes of exactly the block byte size.
* **Error handling** — unknown uniform names (`ShaderException`), wrong
  component/array sizes (`ValueError` / `AssertionError`), array deletion
  and out-of-range indexing (`ShaderException`), invalid binding points.
* **Sabotage self-tests** — `test_sabotaged_length_is_detected` and
  `test_sabotaged_setter_signature_is_detected` deliberately corrupt a copy
  of the mapping (wrong element count / swapped setter) and prove the
  contract check reports it.

## Running locally

```bash
# just the contract suite (verbose shows legacy/DSA parametrization)
python -m pytest tests/unit/shader/test_uniform_contract.py -v

# all shader unit tests
python -m pytest tests/unit/shader
```

## Running in CI

Nothing to configure: the existing `PyTest` workflow
(`.github/workflows/unittests.yml`) runs `pytest tests` on Linux, macOS and
Windows, which discovers this file automatically. The suite needs no
display, GL context, or `xvfb`.

## Representative failure report

This suite was developed against a real defect in
`pyglet/graphics/shader.py`: `GL_FLOAT_MAT3` was mapped with element count
**6** instead of **9**, so a `mat3` uniform would silently upload only 6 of
its 9 floats (and reject a correct 9-element assignment at the ctypes
level). With the defect present, the suite fails like this:

```
________ TestUniformMappingContract.test_mapping_matches_contract_spec _________
    def test_mapping_matches_contract_spec(self):
        violations = find_mapping_violations(shader_module._uniform_setters,
                                             getters=shader_module._uniform_getters)
>       assert violations == [], "\n".join(violations)
E       AssertionError: GL_FLOAT_MAT3: length is 6, expected 9

____ TestUniformMappingContract.test_setter_name_suffix_consistent_with_length ___
E       AssertionError: GL enum 0x8b5b: glUniformMatrix3fv implies 9 elements, but mapped length is 6
E         GL enum 0x8b5b: glProgramUniformMatrix3fv implies 9 elements, but mapped length is 6

_ TestMatrixUniforms.test_matrix_upload[legacy-glUniform-GL_FLOAT_MAT3-Matrix3-9] _
>       assert program.uniforms['mat'] == {'location': 2, 'length': count, 'size': 1}
E       AssertionError: assert {'length': 6, ...} == {'length': 9, ...}

_ TestMatrixUniforms.test_matrix_upload[dsa-glProgramUniform-GL_FLOAT_MAT3-Matrix3-9] _
E       AssertionError: assert {'length': 6, ...} == {'length': 9, ...}

4 failed, 64 passed
```

The fix was a one-line change in `pyglet/graphics/shader.py`
(`GL_FLOAT_MAT3` length `6` → `9`); the suite then passes in full.
