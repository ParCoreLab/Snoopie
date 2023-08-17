import os

is_electron = False

try:
    import zstandard as zstd
except:
    is_electron = True