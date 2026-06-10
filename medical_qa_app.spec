# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import site

from PyInstaller.utils.hooks import collect_submodules


project_root = Path.cwd()
site.getusersitepackages = lambda: ''
site.ENABLE_USER_SITE = False

hiddenimports = (
    collect_submodules('jieba')
    + [
        'importlib.resources',
        'joblib',
        'joblib.numpy_pickle',
        'sklearn.feature_extraction.text',
        'sklearn.metrics.pairwise',
        'scipy.sparse',
        'scipy.sparse._coo',
        'scipy.sparse._csc',
        'scipy.sparse._csr',
    ]
)

datas = [
    ('ui_preview.html', '.'),
    ('Data_数据', 'Data_数据'),
]

a = Analysis(
    ['medical_qa_app.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'PIL',
        'botocore',
        'cupy',
        'dask',
        'lxml',
        'matplotlib',
        'openpyxl',
        'pandas',
        'pygame',
        'tensorflow',
        'torch',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MedicalQA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='MedicalQA',
)
