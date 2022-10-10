import os
import shutil


def clear_cache(tmpdir):
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)


