import os
import shutil
import tarfile

import time

class LevelCacheTar(object):
    def __init__(self, cache_tar):
        self._intar_parent_dir_name = os.path.splitext(os.path.basename(cache_tar))[0]
        self._cache_tar = tarfile.open(cache_tar)
        #self._cache_tar_filenames = self._cache_tar.getnames()

    def fetch(self, key, pk3_path):
        t = time.time()
        intar_path = os.path.join(self._intar_parent_dir_name, key)
        #assert intar_path in self._cache_tar_filenames
        member = self._cache_tar.getmember(intar_path)
        with open(pk3_path, 'wb') as f:
            f.write(self._cache_tar.extractfile(member).read())
        print('Cache hit: {}, {}'.format(key, time.time() - t))
        return True

    def write(self, key, pk3_path):
        assert False, 'There must be no cache miss.'

