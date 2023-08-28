import os
import io
from .electron_checker import is_electron
from typing import List

if not is_electron:
    import zstandard as zstd


def file_from_filepath(filepath):
    f = None
    if os.path.exists(filepath):
        f = open(filepath, "r")
    return f


def file_from_filepath_compressed(filepath):
    if is_electron:
        print("electron build currently does not support decompressing zstd files")
        exit(1)
    f = None
    if os.path.exists(filepath):
        f = open(filepath, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        f = io.TextIOWrapper(reader, encoding="utf-8")
    return f


def file_from_upload(file):
    tmp_stream = io.BytesIO(file.read())
    f = io.TextIOWrapper(tmp_stream, encoding="utf-8")
    return f, file


def file_from_upload_compressed(file):
    if is_electron:
        print("electron build currently does not support decompressing zstd files")
        exit(1)
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(file)
    f = io.TextIOWrapper(reader, encoding="utf-8")
    return f, file


def file_from_filepath_check(filepath):
    if filepath.endswith(".zst"):
        return file_from_filepath_compressed(filepath)
    else:
        return file_from_filepath(filepath)


def file_from_upload_check(file):
    if file.name.endswith(".zst"):
        return file_from_upload_compressed(file)
    else:
        return file_from_upload(file)
    
def multi_file_frop_upload_check(file: List[str]):
    tmp = [file_from_upload_check(i) for i in file]
    ret = [(),()]
    ret[0] = [i[0] for i in tmp]
    ret[1] = [i[1] for i in tmp]
    return ret