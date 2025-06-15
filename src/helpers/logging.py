import sys
from datetime import datetime
DEBUG_LEVEL = 0

def debug(msg, debug_level=1, debug_override=False):
    doPrint = False
    if debug_level <= DEBUG_LEVEL:
        doPrint = True
    if debug_override is not False:
        doPrint = debug_override
    if doPrint:
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}", file=sys.stderr)