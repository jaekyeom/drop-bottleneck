# coding=utf-8
import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
from collections import defaultdict
import numpy as np

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError

class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s'%filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, np.ndarray) and val.dtype == np.uint8:
                continue
            if isinstance(val, np.ndarray) and val.dtype in [np.float32, np.float64, np.float]:
                continue
            #if isinstance(val, float):
            if hasattr(val, '__float__'):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1: # add space unless this is the last one
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'wt')

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()

class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = ','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = [x for x in kvs.keys() if x not in self.keys]
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, dir):
        if not os.path.exists(dir):
          os.makedirs(dir)
        self.dir = dir
        self.step = 1
        prefix = 'events'
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            if isinstance(v, np.ndarray) and v.dtype == np.uint8:
                from PIL import Image
                import io
                num_channels = (1 if len(v.shape) == 2 else v.shape[2])
                img = Image.fromarray(v, mode={1: 'L', 3: 'RGB'}[num_channels])
                try:
                    io_output = io.BytesIO()
                except:
                    io_output = io.StringIO()
                img.save(io_output, format='PNG')
                img_string = io_output.getvalue()
                io_output.close()
                return self.tf.Summary.Value(**{
                    'tag': k,
                    'image': self.tf.Summary.Image(
                            encoded_image_string=img_string,
                            height=v.shape[0],
                            width=v.shape[1],
                            colorspace=num_channels,
                    ),
                })
            if isinstance(v, np.ndarray) and v.dtype in [np.float32, np.float64, np.float]:
                try:
                    return self.tf.Summary.Value(**{
                        'tag': k,
                        'histo': make_histogram(np.reshape(v, [-1]).astype(float), debug_name=k),
                    })
                except ValueError as e:
                    print(e.message.replace('\n', '\t'))
                    return None
                    
            if hasattr(v, '__float__'):
                simple_value = float(v)
            else:
                simple_value = -1.0
            kwargs = {'tag': k, 'simple_value': simple_value}
            return self.tf.Summary.Value(**kwargs)
        summary_values = [summary_val(k, v) for k, v in kvs.items()]
        summary_values = [sv for sv in summary_values if sv is not None]
        summary = self.tf.Summary(value=summary_values)
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None

def make_output_format(format, ev_dir, log_suffix=''):
    if not os.path.exists(ev_dir):
      os.makedirs(ev_dir)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        return HumanOutputFormat(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    elif format == 'json':
        return JSONOutputFormat(osp.join(ev_dir, 'progress%s.json' % log_suffix))
    elif format == 'csv':
        return CSVOutputFormat(osp.join(ev_dir, 'progress%s.csv' % log_suffix))
    elif format == 'tensorboard':
        return TensorBoardOutputFormat(osp.join(ev_dir, 'tb%s' % log_suffix))
    else:
        raise ValueError('Unknown format specified: %s' % (format,))

# ================================================================
# API
# ================================================================

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    Logger.CURRENT.logkv(key, val)

def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    Logger.CURRENT.logkv_mean(key, val)

def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)

def dumpkvs():
    """
    Write all of the diagnostics from the current iteration

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dumpkvs()

def getkvs():
    return Logger.CURRENT.name2val


def log_internal(level=INFO, *args):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    print (args)
    Logger.CURRENT.log_internal(level, *args)

def log(*args):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    log_internal(INFO, *args)

def debug(*args):
    log_internal(DEBUG, *args)

def info(*args):
    log_internal(INFO, *args)

def warn(*args):
    log_internal(WARN, *args)

def error(*args):
    log_internal(ERROR, *args)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    Logger.CURRENT.set_level(level)

def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return Logger.CURRENT.get_dir()

record_tabular = logkv
dump_tabular = dumpkvs

class ProfileKV:
    """
    Usage:
    with logger.ProfileKV("interesting_scope"):
        code
    """
    def __init__(self, n):
        self.n = "wait_" + n
    def __enter__(self):
        self.t1 = time.time()
    def __exit__(self ,type, value, traceback):
        Logger.CURRENT.name2val[self.n] += time.time() - self.t1

def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with ProfileKV(n):
                return func(*args, **kwargs)
        return func_wrapper
    return decorator_with_name


# ================================================================
# Backend
# ================================================================

class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.level == DISABLED: return
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, *args):
      self.log_internal(level=INFO, *args)

    def log_internal(self, level=INFO, *args):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

def get_git_commit_hash():
    import subprocess
    p = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit, _ = p.communicate()
    git_commit = git_commit.strip().decode('utf-8')
    return git_commit

def save_git_diff_to_file(git_diff_file_path):
    import subprocess
    git_diff_file = open(git_diff_file_path, 'w')
    p = subprocess.Popen(['git', 'diff', '--patch', 'HEAD'], stdout=git_diff_file)
    p.wait()

def configure(dir=None, format_strs=None):
    if dir is None:
        dir = os.getenv('OPENAI_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(dir, str)
    if not os.path.exists(dir):
      os.makedirs(dir)

    log_suffix = ''
    rank = 0
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ['PMI_RANK', 'OMPI_COMM_WORLD_RANK']:
        if varname in os.environ:
            rank = int(os.environ[varname])
    if rank > 0:
        log_suffix = "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
        else:
            format_strs = os.getenv('OPENAI_LOG_FORMAT_MPI', 'log').split(',')
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
    log('Logging to %s'%dir)

    git_commit = get_git_commit_hash()
    log('Git commit: {}'.format(git_commit))
    git_diff_file_path = os.path.join(dir, 'git_diff_{}.patch'.format(git_commit))
    save_git_diff_to_file(git_diff_file_path)

def _configure_default_logger():
    format_strs = None
    # keep the old default of only writing to stdout
    if 'OPENAI_LOG_FORMAT' not in os.environ:
        format_strs = ['stdout']
    configure(format_strs=format_strs)
    Logger.DEFAULT = Logger.CURRENT

def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')

class scoped_configure(object):
    def __init__(self, dir=None, format_strs=None):
        self.dir = dir
        self.format_strs = format_strs
        self.prevlogger = None
    def __enter__(self):
        self.prevlogger = Logger.CURRENT
        configure(dir=self.dir, format_strs=self.format_strs)
    def __exit__(self, *args):
        Logger.CURRENT.close()
        Logger.CURRENT = self.prevlogger

# ================================================================

def _demo():
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    dir = "/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see b = 33.3")

    logkv("b", -2.5)
    dumpkvs()

    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()


# ================================================================
# Readers
# ================================================================

def read_json(fname):
    import pandas
    ds = []
    with open(fname, 'rt') as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pandas.DataFrame(ds)

def read_csv(fname):
    import pandas
    return pandas.read_csv(fname, index_col=None, comment='#')

def read_tb(path):
    """
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    import pandas
    import numpy as np
    from glob import glob
    from collections import defaultdict
    import tensorflow as tf
    if osp.isdir(path):
        fnames = glob(osp.join(path, "events.*"))
    elif osp.basename(path).startswith("events."):
        fnames = [path]
    else:
        raise NotImplementedError("Expected tensorboard file or directory containing them. Got %s"%path)
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for v in summary.summary.value:
                    pair = (summary.step, v.simple_value)
                    tag2pairs[v.tag].append(pair)
                maxstep = max(summary.step, maxstep)
    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx,tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step-1, colidx] = value
    return pandas.DataFrame(data, columns=tags)

# https://github.com/ikhlestov/tensorflow_profiling/blob/c1ccd69e1128c2da2ae491464538909fa4ded8ca/03_merged_timeline_example.py
class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

def _create_default_bins():
    v = 1E-12
    buckets = []
    neg_buckets = []
    while v < 1E20:
        buckets.append(v)
        neg_buckets.append(-v)
        v *= 1.1
    return neg_buckets[::-1] + [0] + buckets

g_default_bins = _create_default_bins()


# From tensorboardX.
def make_histogram(values, bins=g_default_bins, max_bins=None, debug_name=''):
    """Convert values into a histogram proto using logic from histogram.cc."""
    if values.size == 0:
        raise ValueError('[make_histogram, {}] The input has no element.'.format(debug_name))
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(counts, pad_width=[[0, subsampling - subsampling_remainder]],
                            mode="constant", constant_values=0)
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits

    # Find the first and the last bin defining the support of the histogram:
    cum_counts = np.cumsum(np.greater(counts, 0, dtype=np.int32))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1
    del cum_counts

    # TensorBoard only includes the right bin limits. To still have the leftmost limit
    # included, we include an empty bin left.
    # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
    # first nonzero-count bin:
    counts = counts[start - 1:end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start:end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError('[make_histogram, {}] The histogram is empty, please file a bug report.'.format(debug_name))

    sum_sq = values.dot(values)
    from tensorflow import HistogramProto
    return HistogramProto(min=values.min(),
                          max=values.max(),
                          num=len(values),
                          sum=values.sum(),
                          sum_squares=sum_sq,
                          bucket_limit=limits.tolist(),
                          bucket=counts.tolist())



# configure the default logger on import
#_configure_default_logger()

if __name__ == "__main__":
    _demo()
