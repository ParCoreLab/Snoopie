import os
import io

def file_from_filepath(filepath):
    f = None
    if os.path.exists(filepath):
        f = open(filepath,"r")
    return f

def file_from_filepath_compressed(filepath):
    f = None
    if os.path.exists(filepath):
        f = open(filepath, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        f = io.TextIOWrapper(reader, encoding='utf-8')
    return f

def file_from_upload(file):
    tmp_stream = io.BytesIO(file.read())
    f = io.TextIOWrapper(tmp_stream, encoding="utf-8")
    return f

def file_from_upload_compressed(file):
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(file)
    f = io.TextIOWrapper(reader, encoding='utf-8')
    return f

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
