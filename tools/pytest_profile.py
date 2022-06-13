# Adapted from line-profiler (BSD)

import builtins
import inspect
import linecache
import os
import sys
import line_profiler

import numpy as np
from pytest import console_main


def main():
    prof = line_profiler.LineProfiler()
    builtins.__dict__['profile'] = prof

    # Make sure the script's directory is on sys.path instead of just
    # kernprof.py's.
    retval = 1
    try:
        try:
            sys.argv[:] = ['pytest'] + sys.argv[1:]
            retval = console_main()
        except KeyboardInterrupt:
            pass
        except SystemExit as exc:
            retval = exc.code
    finally:
        lstats = prof.get_stats()
        stats = lstats.timings
        unit = lstats.unit
        stream = sys.stdout
        stream.write('Determining timing results...\n')
        total_times = list()
        keys = list()
        min_time = 1. / unit
        for key, val in stats.items():
            total_time = sum(t[2] for t in val) if len(val) else 0.
            if total_time >= min_time:
                keys.append(key)
                total_times.append(total_time)
        if len(keys) == 0:
            stream.write(f'No functions that tool longer than {min_time} s')
            return retval
        order = np.argsort(total_times)[::-1]
        items = [(ii, total_times[idx]) + keys[idx] + (stats[keys[idx]],)
                 for ii, idx in enumerate(order)]
        template = '%6s %9s %12s %8s %8s  %-s'
        all_time = np.sum(total_times)
        del total_times
        scalar = unit / 1e-6
        for ii, total_time, filename, start_lineno, func_name, timings in \
                items:
            d = {}
            total_time = 0.0
            linenos = []
            for lineno, nhits, time in timings:
                total_time += time
                linenos.append(lineno)

            stream.write(
                '*' * 120 +
                f'\n#{ii + 1}. {func_name}: {total_time * unit:g} s '
                f'({100 * total_time / all_time:0.2f}%)\n')
            if os.path.exists(filename):
                stream.write(f'In {filename} @ {start_lineno}\n')
                if os.path.exists(filename):
                    # Clear the cache to ensure that we get up-to-date results.
                    linecache.clearcache()
                all_lines = linecache.getlines(filename)
                sublines = inspect.getblock(all_lines[start_lineno - 1:])
            else:
                stream.write('\n')
                stream.write(f'Could not find file {filename}\n')
                stream.write('Are you sure you are running this program from the same directory\n')
                stream.write('that you ran the profiler from?\n')
                stream.write("Continuing without the function's contents.\n")
                # Fake empty lines so we can see the timings, if not the code.
                nlines = max(linenos) - min(min(linenos), start_lineno) + 1
                sublines = [''] * nlines
            for lineno, nhits, time in timings:
                d[lineno] = (nhits,
                            '%5.1f' % (time * scalar),
                            '%5.1f' % (float(time) * scalar / nhits),
                            '%5.1f' % (100 * time / total_time) )
            linenos = range(start_lineno, start_lineno + len(sublines))
            empty = ('', '', '', '')
            header = template % ('Line #', 'Hits', 'Time', 'Per Hit', '% Time',
                                'Line Contents')
            stream.write(header)
            stream.write('\n')
            for lineno, line in zip(linenos, sublines):
                nhits, time, per_hit, percent = d.get(lineno, empty)
                txt = template % (lineno, nhits, time, per_hit, percent,
                                line.rstrip('\n').rstrip('\r'))
                if percent and float(percent) > 1:
                    stream.write(txt)
                    stream.write('\n')
            stream.write('\n')
    return retval


if __name__ == '__main__':
    sys.exit(main())
