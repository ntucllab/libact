project(
    'libact',
    'c',
    'cython',
    'cpp',
    meson_version: '>=1.5.0',
    default_options: ['buildtype=release', 'c_std=c11', 'cpp_std=c++17'],
)
py = import('python').find_installation(pure: false)

result = run_command(
    py,
    '-c', 'import os; print("True" if os.environ.get("READTHEDOCS") else "False")',
    check: true,
)
is_readthedocs = result.stdout().strip() == 'True'

if not is_readthedocs
    subdir('libact')
endif