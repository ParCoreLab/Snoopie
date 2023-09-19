import os

is_electron = False

try:
    import zstandard as zstd
except:
    is_electron = True

is_debug = "SNOOPIE_DEBUG" in os.environ and os.environ["SNOOPIE_DEBUG"] is not None

def electron_load_highlight():
    if not os.path.exists("highlight.min.js"):
        return "console.error('Please follow the instructions to generate highlight.min.js file on electron.')"
    with open("highlight.min.js","r") as f:
        text = f.read()
        return text
