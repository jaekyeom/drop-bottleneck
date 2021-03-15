import os
import shutil

class LevelCache(object):
    def __init__(self, cache_dir):
        self._cache_dir = cache_dir
        try:
            os.makedirs(self._cache_dir)
        except:
            pass

    def fetch(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        if os.path.isfile(path):
            print('Cache hit: {}'.format(key))
            shutil.copyfile(path, pk3_path)
            return True
        print('Cache miss: {}'.format(key))
        return False

    def write(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        shutil.copyfile(pk3_path, path)

