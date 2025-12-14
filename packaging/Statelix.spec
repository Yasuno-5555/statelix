# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\kouno\\Desktop\\Projects\\statelix\\statelix_py\\app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['statelix_py.models', 'sklearn', 'sklearn.linear_model', 'sklearn.neighbors', 'sklearn.utils._typedefs', 'scipy.sparse.csgraph', 'scipy.special.cython_special', 'pandas'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Statelix',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Statelix',
)
