#!/usr/bin/env python3
"""
Step 1 of 2: raw smart-meter telemetry -> clean daily table + window index.

Standalone version of the `windowed_cleaning` notebook. It reads the raw consumption-interval records,
cleans them to ONE ROW PER METER AND DAY, and then decides - without copying any data - which 90-day
input / 60-day target windows are usable. Step 2 (`whole_pipeline_v9_index.py`) rebuilds the window
arrays from the two tables written here.

Why an index instead of saved windows
-------------------------------------
The previous version saved every window as two arrays (90 + 60 days). With stride 30 each calendar day
sits in about five windows, so the output was ~5x the size of the cleaned data, the shuffle that builds
the arrays was the slowest part of every batch, and every reader had to load all the duplicates. Now:

  daily table   <output>/daily_<tag>/batch=NNN     meter_id, calendar_day, daily_kwh (flatlines nulled),
                                                   raw_daily_kwh_before_cleaning, is_raw, interval_days,
                                                   flatline_type, flatline_run_id, flatline_run_length
                one row per meter and day, written ONCE
  window index  <output>/index_<tag>/batch=NNN     one tiny row per candidate (meter_id, origin_date):
                                                   complete, passes, n_null_in, max_input_run,
                                                   n_null_recent, n_null_tg, n_smeared_in, n_smeared_tg
                counts come from rolling sums over the daily series; `passes` is exactly the old
                three-rule filter

A window is never stored: origin_date + the daily table give it back (input = origin-89..origin,
target = origin+1..origin+60). The index keeps the COUNTS and not just the verdict, so a threshold
study (what if the gap limit were 3 days?) needs the cheap index step only, not the raw data.

Folder names carry the settings, so runs with different settings never mix and an interrupted run
resumes on the right data:
    daily_nb020_qnone_f4/                                 20 batches, no quality filter, flatline run >= 4
    index_nb020_qnone_f4_L90_H60_S30_G5_R30/              lookback 90, horizon 60, stride 30, gap limit 5,
                                                          last 30 input days must be complete
A changed index setting (lookback/horizon/stride/gap/recent) re-uses the existing daily table and only
runs the index step.

Stages
------
1. preprocess_household_consumption  (unchanged)
     - drop recordings covering more than 3 days (interval_days > 3)
     - spread 2- and 3-day intervals evenly over their days (is_raw = False)
     - run-length flatline detection on RAW days only; flatline runs of
       >= --min-flatline-run days are nulled (original value kept in
       raw_daily_kwh_before_cleaning, run id / length / type kept as audit columns)
2. build_daily_table: slim columns, written to daily_<tag>/batch=NNN
3. build_window_index: per meter, from the written daily table
     - candidate origins: first_day + LOOKBACK-1, then every STRIDE days, last origin <= last_day - HORIZON
     - complete  = every one of the 90 input and 60 target days has exactly one record
     - passes    = complete
                   AND no flatline run has more than --max-input-fragment null days inside the input window
                   AND no missing day in the 60-day target
                   AND no missing day in the last --recent-clean-days input days
   (the same rules as before; one corner case is now stricter: a window whose span contains a
    DUPLICATED calendar day is rejected, where the old code could accept it if a missing day happened
    to cancel the duplicate and then shifted the days by one position)
4. windows_by_origin_month.csv from the index (windows before / after the filters, and why they fail)

Credentials are read from the environment, never from this file:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT   (+ optional AWS_REGION)
They are only required when an input/output path starts with s3a://.

Memory-safe full run
--------------------
  * only the columns the pipeline needs are read; the interval columns are dropped after the day explode
  * --n-batches N processes the meters in N hash batches (pmod(hash(meter_id), N)), each written to
    <...>/batch=NNN and skipped on a re-run if its _SUCCESS file exists, so memory and shuffle disk are
    bounded by 1/N of the data and an interrupted run resumes
  * the index step needs Python workers with numpy, pandas and pyarrow (checked at start-up)
  * S3 uploads are buffered on disk instead of off-heap memory; --local-dir moves Spark's shuffle
    spill to a disk with room; the shuffle-partition count follows the batch size
  * --quality-filter eda|target drops intervals the master guide says not to use (default none)
  * --min-flatline-run is now really passed to the cleaning step (before, the option was parsed but the
    default 4 was always used)

Experiment tracking (MLflow, optional)
--------------------------------------
With --mlflow-uri http://host:5000 (or MLFLOW_TRACKING_URI in the environment) the run is recorded in the
experiment "electricity/data-prep" (see PrepTracking; mlflow_utils.py must sit next to this file):
one MLflow run per store, resumed by the next process while batches are still missing (so a loop of
--max-new-batches 1 processes is ONE run); every argument as a parameter; per batch the windows, retention,
failure reasons and seconds; per origin month the windows before / after the filters; memory, java / python
RSS and minimum free disk about once a minute; windows_by_origin_month.csv and the run log as artifacts;
tag pipeline_state = complete when every batch is built. Without a URI nothing changes. An MLflow failure
only prints one warning: it can never stop this job. Credentials for the server go in MLFLOW_TRACKING_USERNAME
/ MLFLOW_TRACKING_PASSWORD / MLFLOW_TRACKING_TOKEN, never in the URI; they are masked in the log.

Run log (survives a crash of the terminal / pod)
------------------------------------------------
Everything printed - Python, tracebacks and the Java/Spark output - is copied to
<--log-dir>/process_raw_<time>_<pid>.log (default ~/work/logs; keep it on the PERSISTENT disk), each line
fsync'd. The file also gets STAGE START/END/FAIL lines for every phase and every batch, a heartbeat line
every --heartbeat-secs (container memory, OOM-kill counter, python / java RSS, free disk, Spark task
progress), and a final EXIT line. After a crash:
    tail -n 40 ~/work/logs/process_raw_latest.log
    grep -E " (STAGE|SIG|EXIT|WARN|ERROR) " ~/work/logs/process_raw_latest.log | tail -n 30
A log WITHOUT an EXIT line was killed hard (SIGKILL / OOM-kill / pod restart); its last STAGE and HB
lines show the phase and the memory / disk state just before. The next run also prints how the previous
log ended. Closing the terminal (SIGHUP) is ignored by default so the job keeps running (--exit-on-hup
to change that). `kill -USR1 <pid>` writes the Python stacks of all threads into the log.
Credential values are masked in the log and never printed.

Example
-------
    source ~/work/ec_env.sh      # sets AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT
    python process_raw.py --sample-mode full --n-batches 20 --cores 6 --driver-memory 24g \
        --local-dir ~/work/spark_tmp --log-dir ~/work/logs \
        --output s3a://phase1/electricity/analytics/features/clean_daily_store

    # trial on one batch first (batch 0 of 20):
    python process_raw.py --sample-mode full --n-batches 20 --only-batch 0 --local-dir ~/work/spark_tmp

    # try another gap limit without touching the raw data (daily table is re-used, only the index is rebuilt):
    python process_raw.py --sample-mode full --n-batches 20 --max-input-fragment 3 --local-dir ~/work/spark_tmp
"""

import argparse
import atexit
import contextlib
import faulthandler
import glob
import os
import queue
import re
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, IntegerType, StructField, StructType

try:                                   # optional: experiment tracking (see mlflow_utils.py; off unless a URI is set)
    from mlflow_utils import Tracker
except ImportError:                    # mlflow_utils.py not next to this file: everything below becomes a no-op
    Tracker = None


# =============================================================================
# Crash-proof run log
# =============================================================================
# Everything the run prints (Python print, tracebacks, and the Java/Spark/py4j output, which is
# captured at the file-descriptor level) is copied into a text file on the persistent disk, and
# every line is fsync'd so it survives the pod being killed. On top of that the file gets:
#   STAGE START/END/FAIL  one pair per phase of the run, with the elapsed time
#   HB                    a heartbeat every --heartbeat-secs (every 5 s while memory > 85% of the
#                         container limit or any watched disk has < 15 GB free): container memory,
#                         OOM-kill counter, python/java RSS, free disk, Spark task progress
#   SIG / EXIT            a signal that reached the script / the final status of the run
# A log that ends WITHOUT an EXIT line was killed hard (SIGKILL, OOM-kill, pod restart): its last
# STAGE line says where it was and its last HB lines say what memory and disk were doing.
# Credential values are masked in the log; nothing here prints them.
GB = float(1 << 30)
_URL_USERINFO = re.compile(r"://[^/\s@]+@")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _hms(sec):
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cgroup_memory():
    """Container memory (cgroup v2 or v1): cur / lim / anon / file bytes and the OOM counters."""
    out = dict.fromkeys(("cur", "lim", "anon", "file", "oom", "oom_kill"))
    v = _read_text("/sys/fs/cgroup/memory.current") or _read_text("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if v and v.isdigit():
        out["cur"] = int(v)
    v = _read_text("/sys/fs/cgroup/memory.max") or _read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v and v.isdigit() and int(v) < (1 << 60):
        out["lim"] = int(v)
    kv = {}
    stat = _read_text("/sys/fs/cgroup/memory.stat") or _read_text("/sys/fs/cgroup/memory/memory.stat") or ""
    for line in stat.splitlines():
        p = line.split()
        if len(p) == 2 and p[1].isdigit():
            kv[p[0]] = int(p[1])
    out["anon"] = kv.get("anon", kv.get("rss"))
    out["file"] = kv.get("file", kv.get("cache"))
    ev = _read_text("/sys/fs/cgroup/memory.events") or _read_text("/sys/fs/cgroup/memory/memory.oom_control") or ""
    for line in ev.splitlines():
        p = line.split()
        if len(p) == 2 and p[1].isdigit() and p[0] in ("oom", "oom_kill"):
            out[p[0]] = int(p[1])
    return out


def process_tree_rss():
    """(RSS of this python process, summed RSS of all its descendants = the Spark JVM) in bytes."""
    root, page = os.getpid(), os.sysconf("SC_PAGE_SIZE")
    ppid, rss = {}, {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                s = f.read()
            ppid[int(name)] = int(s[s.rindex(")") + 2:].split()[1])
            with open(f"/proc/{name}/statm") as f:
                rss[int(name)] = int(f.read().split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue
    kids, grew = {root}, True
    while grew:
        grew = False
        for p, pp in ppid.items():
            if pp in kids and p not in kids:
                kids.add(p)
                grew = True
    return rss.get(root, 0), sum(rss.get(p, 0) for p in kids if p != root)


class RunLog:
    def __init__(self):
        self.t0 = time.time()
        self.path = self.fd = None
        self.hb_numbers = {}               # numbers of the last snapshot (for experiment tracking)
        self.on_heartbeat = None           # callable(dict) run after every heartbeat
        self.on_finish = None              # callable(status) run once at the end
        self.lock = threading.RLock()   # re-entrant: a signal handler may log while the main thread is logging
        self.stage_name, self.stage_t0 = "startup", self.t0
        self.last_stage = "startup"        # most recently started stage (what the EXIT line reports)
        self.spark = None
        self.watch_dirs = []
        self.secrets = []
        self.finished = False
        self.last_signal = None
        self.hb_secs = 30.0
        self._stop = threading.Event()
        self._capturing = False
        self._saved_out = self._saved_err = None
        self._term_dead = False
        self._term_dropped = 0
        self._drain_token = None
        self._drained = threading.Event()
        self._spark_busy = False

    # ---- file -------------------------------------------------------------------------------
    def open(self, log_dir):
        log_dir = os.path.abspath(os.path.expanduser(log_dir))
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            log_dir = os.path.abspath("logs")
            os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.path = os.path.join(log_dir, f"process_raw_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log")
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        latest = os.path.join(log_dir, "process_raw_latest.log")
        try:
            if os.path.lexists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(self.path), latest)
        except OSError:
            pass

    def _mask(self, text):
        for s in self.secrets:
            text = text.replace(s, "***")
        return _URL_USERINFO.sub("://***@", text)            # user:password@host in any URL (e.g. a tracking URI)

    def _append(self, data: bytes):
        if self.fd is None:
            return
        with self.lock:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(self.fd, view):]
                os.fsync(self.fd)
            except OSError:
                pass

    def emit(self, tag, msg, echo=None):
        """One timestamped line (or several, one per input line) straight into the log, fsync'd."""
        text = self._mask(str(msg))
        stamp = _now()
        lines = text.splitlines() or [""]
        self._append("".join(f"{stamp} {tag:<5} {l}\n" for l in lines).encode("utf-8", "replace"))
        if echo is None:
            echo = tag != "HB"
        if echo and self._capturing:
            self._to_terminal("".join(f"{stamp[11:]} {tag:<5} {l}\n" for l in lines).encode("utf-8", "replace"))

    @contextlib.contextmanager
    def stage(self, name):
        prev = (self.stage_name, self.stage_t0)
        self.stage_name, self.stage_t0 = name, time.time()
        t = self.stage_t0
        self.last_stage = name
        self.emit("STAGE", f"START {name}")
        try:
            yield
        except BaseException as e:
            msg = (str(e).splitlines() or [""])[0][:300]
            self.emit("STAGE", f"FAIL  {name} after {_hms(time.time() - t)}: {type(e).__name__}: {msg}")
            raise
        else:
            self.emit("STAGE", f"END   {name} ({_hms(time.time() - t)})")
        finally:
            self.stage_name, self.stage_t0 = prev

    # ---- terminal tee (file-descriptor level, so the JVM's output is caught too) ---------------
    def capture_output(self):
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            self._saved_out, self._saved_err = os.dup(1), os.dup(2)
        except OSError:
            self._saved_out = self._saved_err = None
        r, w = os.pipe()
        os.dup2(w, 1)
        os.dup2(w, 2)
        os.close(w)
        self._r = r
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(line_buffering=True)
            except Exception:
                pass
        self._termq = queue.Queue(maxsize=2000)
        self._capturing = True
        threading.Thread(target=self._pump, daemon=True, name="log-pump").start()
        self._term_thread = threading.Thread(target=self._terminal_writer, daemon=True, name="log-terminal")
        self._term_thread.start()

    def _to_terminal(self, data: bytes):
        try:
            self._termq.put_nowait(data)
        except queue.Full:
            self._term_dropped += 1
            if self._term_dropped == 1:
                self._append(f"{_now()} WARN  terminal is not accepting output; dropping terminal copies "
                             f"(the log file still gets everything)\n".encode())

    def _terminal_writer(self):
        # separate thread: a hung or vanished terminal can never block the log or the job
        while True:
            item = self._termq.get()
            if item is None:
                return
            if self._term_dead or self._saved_out is None:
                continue
            try:
                view = memoryview(item)
                while view:
                    view = view[os.write(self._saved_out, view):]
            except OSError as e:
                self._term_dead = True
                self._append(f"{_now()} WARN  cannot write to the terminal any more ({e!r}); it was probably "
                             f"closed. Logging to this file continues.\n".encode())

    def _pump(self):
        buf = b""
        while True:
            try:
                ready, _, _ = select.select([self._r], [], [], 1.0)
            except (OSError, ValueError):
                break
            if not ready:
                if buf:
                    self._emit_lines([buf])
                    buf = b""
                continue
            try:
                data = os.read(self._r, 65536)
            except OSError:
                break
            if not data:
                break
            tok = self._drain_token
            fwd = data.replace(tok + b"\n", b"") if tok else data
            for s in self.secrets:
                fwd = fwd.replace(s.encode(), b"***")
            if fwd:
                self._to_terminal(fwd)
            parts = re.split(rb"\r\n|\n|\r", buf + data)
            buf = parts.pop()
            self._emit_lines(parts)
        if buf:
            self._emit_lines([buf])

    def _emit_lines(self, lines):
        stamp, out = _now(), []
        for raw in lines:
            if not raw.strip():
                continue
            if self._drain_token and self._drain_token in raw:
                self._drained.set()
                continue
            out.append(f"{stamp} OUT   {self._mask(raw.decode('utf-8', 'replace'))}\n")
        if out:
            self._append("".join(out).encode("utf-8", "replace"))

    def drain(self, timeout=5.0):
        """Wait until everything printed so far has reached the log."""
        if not self._capturing:
            return
        self._drain_token = f"@@LOGDRAIN-{uuid.uuid4().hex}@@".encode()
        self._drained.clear()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.write(1, self._drain_token + b"\n")
        except (OSError, ValueError):
            return
        self._drained.wait(timeout)
        self._drain_token = None

    def release_output(self):
        if not self._capturing:
            return
        self.drain()
        self._capturing = False
        self._termq.put(None)
        self._term_thread.join(2.0)
        for saved, fd in ((self._saved_out, 1), (self._saved_err, 2)):
            if saved is not None:
                try:
                    os.dup2(saved, fd)
                except OSError:
                    pass

    # ---- signals / faults ------------------------------------------------------------------------
    def _on_signal(self, signum, frame):
        name = signal.Signals(signum).name
        self.last_signal = name
        self.emit("SIG", f"{name} received during stage '{self.stage_name}' after {_hms(time.time() - self.t0)}")
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    def _on_hup(self, signum, frame):
        self.emit("SIG", f"SIGHUP received (the terminal was closed) during stage '{self.stage_name}' - ignored, "
                         f"the job keeps running")

    def install_signals(self, exit_on_hup):
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        if exit_on_hup:
            signal.signal(signal.SIGHUP, self._on_signal)
        else:
            # Ignored BEFORE the JVM is launched, so the JVM inherits "ignore" (this is how nohup works):
            # closing the terminal does not kill Python or Spark.
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        self._fh = os.fdopen(os.dup(self.fd), "w")
        faulthandler.enable(file=self._fh, all_threads=True)      # SIGSEGV / SIGABRT / SIGBUS -> python stacks
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file=self._fh, all_threads=True)   # kill -USR1 <pid>: dump stacks

    def after_jvm_started(self, exit_on_hup):
        if not exit_on_hup:
            signal.signal(signal.SIGHUP, self._on_hup)   # the JVM already inherited SIG_IGN; Python now logs it

    # ---- heartbeat -------------------------------------------------------------------------------
    def snapshot(self):
        m, pressure = cgroup_memory(), False
        parts = [f"stage={self.stage_name!r} +{_hms(time.time() - self.stage_t0)}"]
        if m["cur"] is not None:
            s = f"cgroup {m['cur'] / GB:.1f}" + (f"/{m['lim'] / GB:.0f} GB" if m["lim"] else " GB")
            if m["anon"] is not None and m["file"] is not None:
                s += f" (anon {m['anon'] / GB:.1f}, file {m['file'] / GB:.1f})"
            parts.append(s)
            pressure |= bool(m["lim"] and m["cur"] > 0.85 * m["lim"])
        if m["oom_kill"] is not None:
            parts.append(f"oom_kill={m['oom_kill']}")
        py, kids = process_tree_rss()
        parts.append(f"rss python {py / GB:.1f} / java+children {kids / GB:.1f} GB")
        nums = {"hb/rss_python_gb": py / GB, "hb/rss_java_gb": kids / GB, "hb/load": os.getloadavg()[0]}
        if m["cur"] is not None:
            nums["hb/cgroup_gb"] = m["cur"] / GB
        if m["oom_kill"] is not None:
            nums["hb/oom_kill"] = m["oom_kill"]
        seen, disks = set(), []
        for p in self.watch_dirs:
            try:
                dev, free = os.stat(p).st_dev, shutil.disk_usage(p).free
            except OSError:
                continue
            if dev in seen:
                continue
            seen.add(dev)
            disks.append(f"{p} {free / GB:.0f} GB")
            pressure |= free < 15 * GB
            nums["hb/disk_free_min_gb"] = min(free / GB, nums.get("hb/disk_free_min_gb", free / GB))
        parts.append("disk free: " + " | ".join(disks))
        parts.append(f"load {os.getloadavg()[0]:.1f}")
        self.hb_numbers = nums
        return "  ".join(parts), pressure

    def _spark_line(self):
        """Task progress of the active Spark stages. Asked in a side thread with a timeout: if the JVM is
        too busy (GC thrash) to answer, that is logged instead of silencing the heartbeat."""
        if self.spark is None or self._spark_busy:
            return "spark status: previous query still unanswered (JVM busy?)" if self._spark_busy else None
        box = []

        def ask():
            try:
                tr = self.spark.sparkContext.statusTracker()
                out = []
                for sid in list(tr.getActiveStageIds())[:3]:
                    si = tr.getStageInfo(sid)
                    if si is not None:
                        out.append(f"stage {sid}: {si.numCompletedTasks}/{si.numTasks} done, "
                                   f"{si.numActiveTasks} running, {si.numFailedTasks} failed")
                box.append("spark " + ("; ".join(out) if out else "idle"))
            except Exception as e:                           # noqa: BLE001
                box.append(f"spark status unavailable: {type(e).__name__}")
            finally:
                self._spark_busy = False

        self._spark_busy = True
        t = threading.Thread(target=ask, daemon=True, name="hb-spark")
        t.start()
        t.join(5.0)
        return box[0] if box else "spark status: no answer within 5 s (JVM busy? GC?)"

    def _hb_loop(self):
        wait = self.hb_secs
        while not self._stop.wait(wait):
            wait = self.hb_secs
            try:
                line, pressure = self.snapshot()
                self.emit("HB", line + ("  !! PRESSURE" if pressure else ""))
                if pressure:
                    wait = min(wait, 5.0)
                if self.on_heartbeat:
                    self.on_heartbeat(dict(self.hb_numbers))
                sp = self._spark_line()
                if sp:
                    self.emit("HB", sp)
            except Exception as e:                           # noqa: BLE001
                self.emit("WARN", f"heartbeat error: {e!r}")

    # ---- start / finish --------------------------------------------------------------------------
    def summarize_previous(self):
        """What the previous run's log says about how it ended (needs no EXIT line = killed hard)."""
        try:
            cands = sorted(p for p in glob.glob(os.path.join(self.log_dir, "process_raw_2*.log")) if p != self.path)
            if not cands:
                return
            prev = cands[-1]
            size = os.path.getsize(prev)
            with open(prev, "rb") as f:
                f.seek(max(0, size - 262144))
                tail = f.read().decode("utf-8", "replace").splitlines()
            pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (STAGE|HB|EXIT)\s+(.*)$")
            last = {}
            for line in tail:
                m = pat.match(line)
                if m:
                    last[m.group(2)] = (m.group(1), m.group(3))
                    if m.group(2) == "STAGE" and not m.group(3).startswith("END"):
                        last["STAGE_OPEN"] = (m.group(1), m.group(3))
            name = os.path.basename(prev)
            if "EXIT" in last:
                self.emit("PREV", f"previous run {name} ended: [{last['EXIT'][0]}] {last['EXIT'][1]}")
            else:
                self.emit("PREV", f"previous run {name} has NO exit line: it was killed hard (SIGKILL / OOM-kill / "
                                  f"pod restart) - or it is still running. Do not run two at once.")
                if "STAGE" in last:
                    self.emit("PREV", f"  its last stage line: [{last['STAGE'][0]}] {last['STAGE'][1]}")
                if "HB" in last:
                    self.emit("PREV", f"  its last heartbeat: [{last['HB'][0]}] {last['HB'][1]}")
        except Exception as e:                               # noqa: BLE001
            self.emit("WARN", f"could not read the previous log: {e!r}")

    def banner(self, args):
        self.emit("INFO", f"log file: {self.path}")
        self.emit("INFO", f"command: {' '.join(shlex.quote(a) for a in sys.argv)}")
        self.emit("INFO", f"pid {os.getpid()} | host {socket.gethostname()} | cwd {os.getcwd()} | "
                          f"time zone {time.strftime('%Z %z')}")
        self.emit("INFO", "arguments: " + ", ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
        self.emit("INFO", f"python {sys.version.split()[0]} | pyspark {pyspark.__version__} | "
                          f"JAVA_HOME={os.getenv('JAVA_HOME', '(unset)')}")
        java = os.path.join(os.getenv("JAVA_HOME", ""), "bin", "java")
        try:
            r = subprocess.run([java if os.path.exists(java) else "java", "-version"],
                               capture_output=True, text=True, timeout=15)
            lines = [l for l in (r.stderr or r.stdout).strip().splitlines() if not l.startswith("Picked up")]
            self.emit("INFO", "java: " + " / ".join(lines[:2]))
        except Exception as e:                               # noqa: BLE001
            self.emit("WARN", f"java -version failed: {e!r}")
        env = ", ".join(f"{v}={'set' if os.getenv(v) else 'MISSING'}"
                        for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT"))
        self.emit("INFO", f"credentials in environment (values are never logged): {env}")
        self.emit("INFO", "baseline: " + self.snapshot()[0])

    def start(self, args):
        self.hb_secs = max(2.0, float(args.heartbeat_secs))
        self.secrets = [v for k in ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN",
                                  "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN")
                        if (v := os.getenv(k)) and len(v) >= 6]
        self.open(args.log_dir)
        self.watch_dirs = [os.path.abspath(args.local_dir) if args.local_dir else tempfile.gettempdir(),
                           "/", self.log_dir, os.getcwd()]
        self.capture_output()
        self.install_signals(args.exit_on_hup)
        threading.excepthook = lambda a: self.emit(
            "ERROR", f"thread {a.thread.name if a.thread else '?'} died: {a.exc_type.__name__}: {a.exc_value}")
        atexit.register(lambda: self.finish("ABNORMAL: interpreter exiting without the run finishing"))
        self.emit("INFO", f"run log: {self.path}   (symlink: {os.path.join(self.log_dir, 'process_raw_latest.log')})")
        self.summarize_previous()
        self.banner(args)
        threading.Thread(target=self._hb_loop, daemon=True, name="heartbeat").start()

    def finish(self, status):
        if self.finished:
            return
        self.finished = True
        self._stop.set()
        self.drain()
        self.stage_name = self.last_stage
        try:
            self.emit("HB", "final: " + self.snapshot()[0])
        except Exception:                                    # noqa: BLE001
            pass
        self.emit("EXIT", f"{status} | total {_hms(time.time() - self.t0)} | last stage: {self.last_stage}")
        cb, self.on_finish = self.on_finish, None
        if cb:
            try:
                cb(status)
            except Exception as e:                           # noqa: BLE001
                self.emit("WARN", f"experiment-tracking finish hook failed: {e!r}")
        self.release_output()


RUN = RunLog()

# =============================================================================
# Defaults (identical to the notebook)
# =============================================================================
INPUT_CONSUMPTION_S3 = "s3a://phase1/electricity/analytics/household/consumption_intervals/data/"
CLEAN_STORE_S3 = "s3a://phase1/electricity/analytics/features/clean_daily_store"   # daily_<tag>/ + index_<tag>/
# Old layout: one row per window with the 90 + 60 day arrays (what the 200k-household sample holds).
# Nothing here writes to it; the guard in run() only keeps --output from pointing at it by mistake.
LEGACY_CLEAN_WINDOWS_S3 = "s3a://phase1/electricity/analytics/features/clean_windows.parquet"

LOOKBACK_DAYS = 90
HORIZON_DAYS = 60
STRIDE_DAYS = 30

SAMPLE_MODE = "full"   # "full" | "random_sample"
N_HOUSEHOLDS = 200_000
SEED = 42

MIN_FLATLINE_RUN_LENGTH = 4     # runs of >= 4 identical raw days (i.e. > 3 days) become missing
MAX_INPUT_FRAGMENT_DAYS = 5     # drop windows with an input gap run longer than this
RECENT_CLEAN_DAYS = 30          # last N input days must have no missing value


# =============================================================================
# Spark
# =============================================================================
def create_spark_session(app_name="household-raw-cleaning", cores=8, driver_memory="24g") -> SparkSession:
    paths_need_s3 = True  # decided by caller via configure_s3; kept simple here
    del paths_need_s3
    raise NotImplementedError  # replaced below (kept out of the public API)


def _java_major():
    """Major version of the java that PySpark will start (8, 11, 17, 21 ...); None if it cannot be told."""
    java = os.path.join(os.getenv("JAVA_HOME", ""), "bin", "java")
    try:
        out = subprocess.run([java if os.path.exists(java) else "java", "-version"], capture_output=True,
                             text=True, timeout=30).stderr
        m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
        if not m:
            return None
        return int(m.group(2)) if m.group(1) == "1" else int(m.group(1))     # "1.8.0" -> 8
    except Exception:                                                        # noqa: BLE001
        return None


def _java_open_options():
    """Spark's Arrow transfer (applyInPandas) fails on Java 17+ with 'sun.misc.Unsafe or
    DirectByteBuffer.<init>(long, int) not available' unless java.nio is opened. --add-opens does not
    exist on Java 8, so it is only added for Java 9+."""
    major = _java_major()
    if major is not None and major >= 9:
        return " --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    return ""


def build_spark_session(use_s3: bool, app_name="household-raw-cleaning", cores=8, driver_memory="24g",
                        local_dir=None, shuffle_partitions=1024, s3_buffer="disk",
                        log_level="WARN") -> SparkSession:
    if use_s3:
        missing = [v for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT") if not os.getenv(v)]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {missing}")
        # Must be set BEFORE the JVM starts.
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            "--packages org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.261 pyspark-shell"
        )

    # the Python workers of the index step must be the interpreter that has numpy / pandas / pyarrow
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    builder = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        # --- memory & execution headroom ---
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.memory.fraction", "0.80")
        .config("spark.memory.storageFraction", "0.20")
        # --- window sorting & partition scaling ---
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", "256")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.ansi.enabled", "false")  # avoids DIVIDE_BY_ZERO crashes
        .config("spark.ui.showConsoleProgress", "false")  # the \r progress bar is unreadable in a log file
        # cleaning up the previous batch's shuffle files on a slow disk can keep the block manager busy for
        # minutes; the default 120 s RPC timeout then logs "Cannot receive any reply ... executor driver may
        # have been lost" (seen at the start of batch 2). Local mode: nothing is really lost.
        .config("spark.rpc.askTimeout", "900s")
        .config("spark.network.timeout", "900s")
        .config("spark.executor.heartbeatInterval", "60s")
        # --- GC for heavy window sorts; Arrow (the window-index step) needs direct-buffer access on Java 17+ ---
        .config(
            "spark.driver.extraJavaOptions",
            "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:G1ReservePercent=15 "
            "-Dio.netty.tryReflectionSetAccessible=true" + _java_open_options(),
        )
    )

    if local_dir:
        # shuffle spill and temp files; default is /tmp, which is often small. The S3A "disk" upload
        # buffer also defaults to /tmp, so it is moved here too.
        os.makedirs(os.path.join(local_dir, "s3a"), exist_ok=True)
        builder = (builder.config("spark.local.dir", local_dir)
                   .config("spark.hadoop.fs.s3a.buffer.dir", os.path.join(local_dir, "s3a")))

    if use_s3:
        builder = (
            builder.config("spark.hadoop.fs.s3a.endpoint", os.environ["S3_ENDPOINT"])
            .config("spark.hadoop.fs.s3a.endpoint.region", os.getenv("AWS_REGION", "us-east-1"))
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.fast.upload", "true")
            .config("spark.hadoop.fs.s3a.fast.upload.buffer", s3_buffer)   # "disk": no off-heap buffers
            .config("spark.hadoop.fs.s3a.threads.max", "128")
            .config("spark.hadoop.fs.s3a.connection.maximum", "256")
            .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
            .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
            .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
            .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400")
            .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
            .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            )
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    return spark


# =============================================================================
# Stage 1: interval records -> daily series with flatlines nulled
# =============================================================================
def preprocess_household_consumption(
    df: DataFrame,
    min_flatline_run_length: int = MIN_FLATLINE_RUN_LENGTH,
    quality_filter: str = "none",
) -> DataFrame:
    """
    1. Drops recordings covering more than 3 days (interval_days > 3).
    2. Averages multi-day consumption across 2- and 3-day intervals to daily
       dates, tagging each resulting day `is_raw` (True for genuine 1-day meter
       reads, False for interval-spread days).
    3. Run-length flatline detection on RAW days only. A run is a maximal
       sequence of calendar-consecutive, is_raw=True rows sharing an identical
       `daily_kwh`. Any is_raw=False day breaks the run on both sides, so
       spread-interval artefacts never form or extend a flatline.
         - `flatline_run_length`: length (days) of the run a row belongs to
         - `is_flatline`: run length >= min_flatline_run_length
         - `flatline_type`: "zero" or "nonzero" for qualifying runs, else null
    4. Treats flagged flatlines (zero and nonzero) as missing: the original
       value is kept in `raw_daily_kwh_before_cleaning`, `daily_kwh` is nulled
       and `missing_reason` = "flatline".
    """
    # 1. drop recordings covering more than 3 days
    df_filtered = df.filter(F.col("interval_days") <= 3)

    # optional: the master guide says quarantined intervals must not feed EDA or model targets
    if quality_filter == "eda":
        df_filtered = df_filtered.filter(F.col("is_eda_eligible"))
    elif quality_filter == "target":
        df_filtered = df_filtered.filter(F.col("is_model_target_eligible"))
    elif quality_filter != "none":
        raise ValueError(f"unknown quality_filter {quality_filter!r}")

    date_col = "calendar_day" if "calendar_day" in df_filtered.columns else "end_date"
    kwh_col = "interval_kwh" if "interval_kwh" in df_filtered.columns else "daily_kwh"

    # keep only what the pipeline uses: the other ~25 raw columns would otherwise be carried through
    # every window array and multiply memory and shuffle size
    df_filtered = df_filtered.select(*dict.fromkeys(["meter_id", date_col, kwh_col, "interval_days"]))

    # 2. explode multi-day intervals into daily rows + set is_raw
    df_daily = (
        df_filtered
        .withColumn(
            "_date_seq",
            F.expr(f"sequence(date_sub({date_col}, cast(interval_days as int) - 1), {date_col})"),
        )
        .withColumn("calendar_day", F.explode("_date_seq"))
        .withColumn("daily_kwh", F.col(kwh_col) / F.col("interval_days"))
        .withColumn("is_raw", F.col("interval_days") == 1)
        .drop("_date_seq")
    )
    # the original interval columns are not needed after the explode
    df_daily = df_daily.drop(*[c for c in {kwh_col, date_col} if c not in ("calendar_day", "daily_kwh")])

    # 3. run-length flatline detection on raw, calendar-consecutive days
    w = Window.partitionBy("meter_id").orderBy("calendar_day")

    prev_kwh = F.lag("daily_kwh", 1).over(w)
    prev_date = F.lag("calendar_day", 1).over(w)
    prev_is_raw = F.lag("is_raw", 1).over(w)

    # A row continues the previous row's run only if both rows are raw, the dates
    # are calendar-adjacent, and the value is identical.
    same_run = (
        F.col("is_raw")
        & prev_is_raw.eqNullSafe(True)
        & (F.datediff(F.col("calendar_day"), prev_date) == 1)
        & (F.col("daily_kwh") == prev_kwh)
    )
    new_run_flag = F.when(same_run, F.lit(0)).otherwise(F.lit(1))

    df_runs = df_daily.withColumn(
        "flatline_run_id",
        F.sum(new_run_flag).over(w.rowsBetween(Window.unboundedPreceding, 0)),
    )

    run_w = Window.partitionBy("meter_id", "flatline_run_id")
    df_runs = df_runs.withColumn("flatline_run_length", F.count("*").over(run_w))

    df_flagged = (
        df_runs
        .withColumn(
            "is_flatline",
            F.col("is_raw") & (F.col("flatline_run_length") >= min_flatline_run_length),
        )
        .withColumn(
            "flatline_type",
            F.when(
                F.col("is_flatline"),
                F.when(F.col("daily_kwh") == 0, F.lit("zero")).otherwise(F.lit("nonzero")),
            ).otherwise(F.lit(None).cast("string")),
        )
    )

    # 4. flatlines -> missing, keeping the audit trail
    return (
        df_flagged
        .withColumn("raw_daily_kwh_before_cleaning", F.col("daily_kwh"))
        .withColumn(
            "daily_kwh",
            F.when(F.col("is_flatline"), F.lit(None).cast(df_flagged.schema["daily_kwh"].dataType))
             .otherwise(F.col("daily_kwh")),
        )
        .withColumn(
            "missing_reason",
            F.when(F.col("is_flatline"), F.lit("flatline")).otherwise(F.lit(None).cast("string")),
        )
    )


# =============================================================================
# Stage 2: the clean daily table (one row per meter and day)
# =============================================================================
DAILY_COLUMNS = ["meter_id", "calendar_day", "daily_kwh", "raw_daily_kwh_before_cleaning", "is_raw",
                 "interval_days", "flatline_type", "flatline_run_id", "flatline_run_length"]


def build_daily_table(
    df: DataFrame,
    sample_mode: str = "full",
    n_households: int = None,
    seed: int = 42,
    quality_filter: str = "none",
    min_flatline_run: int = MIN_FLATLINE_RUN_LENGTH,
) -> DataFrame:
    """Raw interval records -> the slim clean daily table (see preprocess_household_consumption).
    daily_kwh is null on flatline days; the original value, the run id / length and the flatline type are
    kept for audit, so nothing is lost by writing only this table."""
    if sample_mode == "random_sample":
        if n_households is None or n_households <= 0:
            raise ValueError("n_households must be a positive integer when sample_mode='random_sample'.")
        sampled_meters = (
            df.select("meter_id").distinct().orderBy(F.rand(seed=seed)).limit(n_households)
        )
        df_filtered = df.join(sampled_meters, on="meter_id", how="inner")
    elif sample_mode == "full":
        df_filtered = df
    else:
        raise ValueError(f"Invalid sample_mode '{sample_mode}'. Expected 'full' or 'random_sample'.")

    df_daily = preprocess_household_consumption(
        df_filtered, min_flatline_run_length=min_flatline_run, quality_filter=quality_filter)
    missing = [c for c in DAILY_COLUMNS if c not in df_daily.columns]
    if missing:
        raise ValueError(f"preprocess_household_consumption output is missing expected columns: {missing}")
    return df_daily.select(*DAILY_COLUMNS)


# =============================================================================
# Stage 3: the window index (which origins give a usable window, and why not)
# =============================================================================
INDEX_COUNT_COLUMNS = ["n_null_in", "max_input_run", "n_null_recent", "n_null_tg", "n_smeared_in", "n_smeared_tg"]
INDEX_COLUMNS = ["meter_id", "origin_date", "complete", "passes"] + INDEX_COUNT_COLUMNS


def compute_window_index(day, is_null, is_smeared, run_id, lookback, horizon, stride,
                         max_input_fragment, recent_clean_days):
    """The window logic for ONE meter, on plain numpy arrays (no Spark; this is what the tests call).

    day         int64 epoch day of every daily record (any order)
    is_null     daily_kwh is null on that record (flatline or a null raw value)
    is_smeared  the day came from a 2-3 day interval spread evenly (not is_raw)
    run_id      flatline_run_id of the record (null days of the same run share it)

    Candidate origins are the old grid: first_day + lookback - 1, every `stride` days, up to
    last_day - horizon. For each origin the input window is origin-lookback+1 .. origin and the target
    window is origin+1 .. origin+horizon. Everything is a difference of prefix sums, so the cost is
    O(days + origins), never O(origins x 150).

    Returns None when the meter is too short for a single window, else a dict of equal-length arrays:
      origin_day, complete, passes, n_null_in, max_input_run, n_null_recent, n_null_tg,
      n_smeared_in, n_smeared_tg
    """
    day = np.asarray(day, dtype=np.int64)
    if day.size == 0:
        return None
    d0, d1 = int(day.min()), int(day.max())
    t_start, t_end = d0 + lookback - 1, d1 - horizon
    if t_start > t_end:
        return None
    origin_day = np.arange(t_start, t_end + 1, stride, dtype=np.int64)
    span = d1 - d0 + 1
    pos = day - d0

    rec = np.bincount(pos, minlength=span)                                   # records per calendar day
    nul = np.bincount(pos, weights=np.asarray(is_null, dtype=float), minlength=span) > 0
    sme = np.bincount(pos, weights=np.asarray(is_smeared, dtype=float), minlength=span) > 0
    dup = rec > 1

    def prefix(x):
        return np.concatenate(([0], np.cumsum(x, dtype=np.int64)))

    c_rec, c_dup, c_nul, c_sme = prefix(rec), prefix(dup), prefix(nul), prefix(sme)

    def wsum(c, a, b):                       # sum over positions a..b inclusive
        return c[b + 1] - c[a]

    o = origin_day - d0
    r = min(recent_clean_days, lookback)
    a_in, b_in = o - lookback + 1, o
    a_tg, b_tg = o + 1, o + horizon
    a_rc = o - r + 1

    complete = (
        (wsum(c_rec, a_in, b_in) == lookback) & (wsum(c_rec, a_tg, b_tg) == horizon)
        & (wsum(c_dup, a_in, b_in) == 0) & (wsum(c_dup, a_tg, b_tg) == 0)
    )
    n_null_in = wsum(c_nul, a_in, b_in)
    n_null_tg = wsum(c_nul, a_tg, b_tg)
    n_null_recent = wsum(c_nul, a_rc, b_in)
    n_smeared_in = wsum(c_sme, a_in, b_in)
    n_smeared_tg = wsum(c_sme, a_tg, b_tg)

    # Longest null run INSIDE the input window, per flatline run id (the old rule counted the null days
    # of each flatline_run_id in the window, so two different back-to-back flatlines are not merged).
    # The null days of one run are consecutive calendar days, so their count in the window is the
    # overlap of [run_start, run_end] with the window.
    max_run = np.zeros(origin_day.size, dtype=np.int64)
    nz = np.flatnonzero(np.asarray(is_null, dtype=bool))
    if nz.size:
        rid, p = np.asarray(run_id)[nz], pos[nz]
        order = np.argsort(rid, kind="stable")
        rid, p = rid[order], p[order]
        first = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
        s, e = np.minimum.reduceat(p, first), np.maximum.reduceat(p, first)
        ov = np.minimum(e[None, :], b_in[:, None]) - np.maximum(s[None, :], a_in[:, None]) + 1
        max_run = np.clip(ov, 0, None).max(axis=1)

    passes = complete & (max_run <= max_input_fragment) & (n_null_tg == 0) & (n_null_recent == 0)
    return dict(origin_day=origin_day, complete=complete, passes=passes, n_null_in=n_null_in,
                max_input_run=max_run, n_null_recent=n_null_recent, n_null_tg=n_null_tg,
                n_smeared_in=n_smeared_in, n_smeared_tg=n_smeared_tg)


def window_index_frame(meter_id, res):
    """compute_window_index result -> pandas frame in the Spark output schema (empty if res is None)."""
    if res is None:
        n = 0
        res = dict(origin_day=np.zeros(0, "int32"), complete=np.zeros(0, bool), passes=np.zeros(0, bool),
                   **{c: np.zeros(0, "int32") for c in INDEX_COUNT_COLUMNS})
    else:
        n = len(res["origin_day"])
    out = {"meter_id": [meter_id] * n, "origin_day": res["origin_day"].astype("int32"),
           "complete": res["complete"].astype(bool), "passes": res["passes"].astype(bool)}
    for c in INDEX_COUNT_COLUMNS:
        out[c] = res[c].astype("int32")
    frame = pd.DataFrame(out)
    if n == 0:                       # an empty float column cannot always be cast to the string / long meter_id type
        frame["meter_id"] = frame["meter_id"].astype(object)
    return frame


def build_window_index(daily: DataFrame, lookback_days, horizon_days, stride_days,
                       max_input_fragment_days, recent_clean_days) -> DataFrame:
    """Spark wrapper: one pandas group per meter, run through compute_window_index. `daily` is the
    clean daily table (read back from disk). Output columns: INDEX_COLUMNS."""
    meter_type = daily.schema["meter_id"].dataType
    schema = StructType(
        [StructField("meter_id", meter_type), StructField("origin_day", IntegerType()),
         StructField("complete", BooleanType()), StructField("passes", BooleanType())]
        + [StructField(c, IntegerType()) for c in INDEX_COUNT_COLUMNS])
    L, H, S, G, R = lookback_days, horizon_days, stride_days, max_input_fragment_days, recent_clean_days

    def per_meter(pdf):
        res = compute_window_index(pdf["day"].to_numpy(), pdf["is_null"].to_numpy(),
                                   pdf["is_smeared"].to_numpy(), pdf["run_id"].to_numpy(), L, H, S, G, R)
        return window_index_frame(pdf["meter_id"].iloc[0], res)

    epoch = F.to_date(F.lit("1970-01-01"))
    slim = daily.select(
        "meter_id",
        F.datediff("calendar_day", epoch).alias("day"),
        F.col("daily_kwh").isNull().alias("is_null"),
        F.coalesce(~F.col("is_raw"), F.lit(False)).alias("is_smeared"),
        F.col("flatline_run_id").cast("long").alias("run_id"),
    )
    idx = slim.groupBy("meter_id").applyInPandas(per_meter, schema)
    return (idx.withColumn("origin_date", F.expr("date_add(to_date('1970-01-01'), origin_day)"))
            .drop("origin_day").select(*INDEX_COLUMNS))


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=INPUT_CONSUMPTION_S3, help="raw consumption-interval parquet (dir or file)")
    p.add_argument("--output", default=CLEAN_STORE_S3,
                   help="store root; gets daily_<tag>/batch=NNN and index_<tag>/batch=NNN (see the module docstring)")
    p.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    p.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    p.add_argument("--stride-days", type=int, default=STRIDE_DAYS)
    p.add_argument("--sample-mode", choices=["full", "random_sample"], default=SAMPLE_MODE)
    p.add_argument("--n-households", type=int, default=N_HOUSEHOLDS, help="used with --sample-mode random_sample")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--min-flatline-run", type=int, default=MIN_FLATLINE_RUN_LENGTH)
    p.add_argument("--max-input-fragment", type=int, default=MAX_INPUT_FRAGMENT_DAYS)
    p.add_argument("--recent-clean-days", type=int, default=RECENT_CLEAN_DAYS)
    p.add_argument("--force", action="store_true", help="skip the memory / disk safety check at start-up")
    p.add_argument("--cores", type=int, default=4,
                   help="Spark threads; more threads = more tasks holding memory at once. Use fewer than the "
                        "machine's cores and keep driver-memory well below its RAM (leave ~30%% for the JVM "
                        "overhead and the OS)")
    p.add_argument("--driver-memory", default="12g")
    p.add_argument("--n-batches", type=int, default=1,
                   help="process the meters in N hash batches written to <output>/batch=NNN (needs "
                        "--sample-mode full); resumable. 1 = the original single pass")
    p.add_argument("--only-batch", type=int, default=None, help="run just this batch number (a trial run)")
    p.add_argument("--rebuild", action="store_true",
                   help="redo the daily table AND the index of batches that already have a _SUCCESS file")
    p.add_argument("--rebuild-index", action="store_true",
                   help="redo only the window index of batches that already have one (the daily table is kept)")
    p.add_argument("--max-new-batches", type=int, default=0,
                   help="stop after computing this many NEW batches (0 = no limit). Lets a shell loop run one batch "
                        "per process, so Spark's local files are dropped at each exit and a crash costs one batch")
    p.add_argument("--quality-filter", choices=["none", "eda", "target"], default="none",
                   help="drop intervals with is_eda_eligible=false (eda) or is_model_target_eligible=false "
                        "(target) before anything else; none keeps the old behaviour")
    p.add_argument("--local-dir", default=None, help="Spark shuffle/spill directory (needs lots of free space)")
    p.add_argument("--shuffle-partitions", type=int, default=None,
                   help="default: max(128, 1024 // n_batches)")
    p.add_argument("--s3-buffer", choices=["disk", "bytebuffer", "array"], default="disk")
    p.add_argument("--log-dir", default="~/work/logs",
                   help="every output line and a timestamped stage / heartbeat trail go to "
                        "<log-dir>/process_raw_<time>_<pid>.log (fsync'd, so it survives a crash); "
                        "process_raw_latest.log points at the newest one. Use a PERSISTENT disk")
    p.add_argument("--heartbeat-secs", type=float, default=30.0,
                   help="seconds between memory / disk / Spark-progress lines in the log (5 s while under pressure)")
    p.add_argument("--spark-log-level", default="WARN", choices=["ERROR", "WARN", "INFO", "DEBUG"],
                   help="Spark/JVM log level reaching the log; WARN keeps task-failure and executor-lost messages")
    p.add_argument("--mlflow-uri", default=None,
                   help="MLflow tracking server / store (default: the MLFLOW_TRACKING_URI environment variable). "
                        "Without one, experiment tracking is off. Credentials belong in MLFLOW_TRACKING_USERNAME / "
                        "_PASSWORD / _TOKEN, not in the URI")
    p.add_argument("--mlflow-experiment", default="electricity/data-prep", help="MLflow experiment name")
    p.add_argument("--mlflow-run-name", default=None, help="MLflow run name (default: process_raw <index tag>)")
    p.add_argument("--no-mlflow", action="store_true", help="switch experiment tracking off even if a URI is set")
    p.add_argument("--exit-on-hup", action="store_true",
                   help="by default closing the terminal (SIGHUP) is ignored, like nohup, and the job keeps "
                        "running; this makes it exit instead")
    p.add_argument("--report", action="store_true",
                   help="no longer needed: the index always holds the windows before and after the filters, so "
                        "windows_by_origin_month.csv has both columns for free (kept so old command lines work)")
    return p.parse_args()


def preflight(args):
    """Print what this container really has and stop if the settings cannot fit.
    In a Kubernetes / Docker pod `free` shows the NODE's memory; the pod is killed at its cgroup limit,
    which is what takes the Jupyter server down with the job."""
    import shutil
    gb = float(1 << 30)
    limit = None
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v.isdigit() and int(v) < (1 << 60):
                limit = int(v)
                break
        except OSError:
            pass
    try:
        node = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        node = None
    for mod in ("numpy", "pandas", "pyarrow"):
        try:
            __import__(mod)
        except ImportError:
            raise SystemExit(f"the window-index step runs in Spark Python workers and needs {mod}: "
                             f"pip install {mod}   (in the SAME environment that runs this script)")
    dm = args.driver_memory.lower()
    dm_gb = float(dm[:-1]) * {"g": 1, "m": 1 / 1024, "t": 1024}.get(dm[-1], 1) if dm[:-1].replace(".", "").isdigit() else None
    path = args.local_dir or "/tmp"
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    print(f"[preflight] container memory limit: {limit / gb:.1f} GB" if limit else
          "[preflight] no cgroup memory limit found",
          f"| node RAM: {node / gb:.1f} GB" if node else "",
          f"| driver-memory {args.driver_memory}, cores {args.cores}",
          f"| free disk at {path}: {free / gb:.0f} GB")
    allowed = limit or node
    if allowed and dm_gb and dm_gb * gb > 0.6 * allowed and not args.force:
        raise SystemExit(f"--driver-memory {args.driver_memory} is more than 60% of the {allowed / gb:.0f} GB this "
                         f"container may use; the JVM needs headroom beyond the heap. Try "
                         f"--driver-memory {max(2, int(0.5 * allowed / gb))}g, or pass --force.")
    if free < 100 * gb and not args.force:
        print(f"[preflight] WARNING: only {free / gb:.0f} GB free at {path}. Shuffle spill and S3 upload buffers "
              f"go there; use --local-dir on a larger disk, or more --n-batches so each batch is smaller.")


def path_exists(spark: SparkSession, path: str) -> bool:
    p = spark._jvm.org.apache.hadoop.fs.Path(path)
    return p.getFileSystem(spark._jsc.hadoopConfiguration()).exists(p)


def index_month_counts(idx: DataFrame, max_input_fragment: int):
    """Per origin month: candidates, complete windows, windows that pass, and the reasons the others fail
    (a window can fail for several reasons at once, so the three fail columns overlap)."""
    c = F.col("complete")
    i = lambda cond: F.sum(cond.cast("int"))                                  # noqa: E731
    return (idx.groupBy(F.date_format("origin_date", "yyyy-MM").alias("origin_month"))
            .agg(F.count("*").alias("candidates"), i(c).alias("windows_before_filters"),
                 i(F.col("passes")).alias("windows_after_filters"),
                 i(c & (F.col("max_input_run") > max_input_fragment)).alias("fail_input_gap"),
                 i(c & (F.col("n_null_tg") > 0)).alias("fail_target_missing"),
                 i(c & (F.col("n_null_recent") > 0)).alias("fail_recent_missing"))
            .toPandas())


class PrepTracking:
    """MLflow tracking of one process_raw.py invocation (all methods are no-ops when tracking is off).

    * One MLflow run per STORE (index tag): a run that is still "partial" (not every batch built yet) is
      resumed by the next process, so a shell loop with --max-new-batches 1 ends up as ONE run with one
      step per batch instead of 20 loose runs.
    * params: every command-line argument; tags: daily / index folder tags, store, code version, host;
    * metrics per batch (step = batch number): windows, retention, failure reasons, seconds per stage;
      per origin month (step = YYYYMM): windows before / after the filters, retention;
      heartbeat numbers (memory, java / python RSS, minimum free disk) about once a minute;
    * artifacts: windows_by_origin_month.csv and the run log, uploaded when the run ends;
    * state: tag pipeline_state = complete once every batch has a daily table and an index.
    The training script finds this run through the index tag (tag data_prep_run_id on its own run)."""

    def __init__(self, args, daily_tag, index_tag, store, n):
        self.tr, self.rid, self.complete, self.n = None, None, False, n
        self._last_hb = 0.0
        if Tracker is None:
            return
        tr = Tracker(args.mlflow_experiment, uri=args.mlflow_uri, enabled=False if args.no_mlflow else None)
        if not tr.enabled:
            return
        prev = tr.find_run(f"tags.index_tag = '{index_tag}' and tags.pipeline_state = 'partial'")
        self.rid = tr.begin(args.mlflow_run_name or f"process_raw {index_tag}",
                            resume_run_id=prev.info.run_id if prev else None)
        if not self.rid:
            return
        self.tr = tr
        tr.tags({"stage": "data_prep", "daily_tag": daily_tag, "index_tag": index_tag, "store": store,
                 "input": args.input, "pipeline_state": "partial", "log_file": RUN.path or "",
                 **tr.system_tags()})
        tr.params({k: v for k, v in vars(args).items() if not k.startswith("mlflow")})
        RUN.emit("INFO", f"MLflow run {self.rid} ({'resumed' if prev else 'new'}) in experiment "
                         f"{args.mlflow_experiment!r}")
        RUN.on_heartbeat = self._heartbeat
        RUN.on_finish = self._finish

    def _heartbeat(self, nums):
        if self.tr and time.time() - self._last_hb >= 60:
            self._last_hb = time.time()
            self.tr.metrics(nums, step=int(time.time() - RUN.t0), run_id=self.rid)

    def batch(self, b, month_counts, seconds):
        if not self.tr:
            return
        s = month_counts
        cand, comp, ok = (int(s[c].sum()) for c in ("candidates", "windows_before_filters", "windows_after_filters"))
        self.tr.metrics({
            "batch/candidates": cand, "batch/complete": comp, "batch/passes": ok,
            "batch/retention": ok / comp if comp else float("nan"),
            "batch/fail_input_gap": int(s["fail_input_gap"].sum()),
            "batch/fail_target_missing": int(s["fail_target_missing"].sum()),
            "batch/fail_recent_missing": int(s["fail_recent_missing"].sum()),
            **{f"batch/seconds_{k}": v for k, v in seconds.items()}}, step=b, run_id=self.rid)

    def months(self, tot, csv_path, n_batches_seen, loop_completed):
        if not self.tr:
            return
        for month, r in tot.iterrows():
            step = int(str(month).replace("-", ""))
            self.tr.metrics({"month/windows_before_filters": r["windows_before_filters"],
                             "month/windows_after_filters": r["windows_after_filters"],
                             "month/retention": r["retention"]}, step=step, run_id=self.rid)
        self.tr.metrics({"total/windows_before_filters": tot["windows_before_filters"].sum(),
                         "total/windows_after_filters": tot["windows_after_filters"].sum(),
                         "total/retention": tot["windows_after_filters"].sum() / max(1, tot["windows_before_filters"].sum()),
                         "total/batches_seen": n_batches_seen}, run_id=self.rid)
        self.tr.artifact(csv_path, run_id=self.rid)
        self.complete = bool(loop_completed and n_batches_seen == self.n)

    def _finish(self, status):
        if not self.tr:
            return
        st = ("FINISHED" if status.startswith("CLEAN EXIT") else
              "KILLED" if status.startswith("INTERRUPTED") else "FAILED")
        self.tr.tags({"exit_status": status[:240], "pipeline_state": "complete" if self.complete else "partial"},
                     run_id=self.rid)
        if RUN.path:
            self.tr.artifact(RUN.path, subdir="logs", run_id=self.rid)
        self.tr.end(st)


def run(args):
    n = max(1, args.n_batches)
    if n > 1 and args.sample_mode != "full":
        raise SystemExit("--n-batches > 1 needs --sample-mode full (a random sample is drawn over all meters)")
    if args.only_batch is not None and not (0 <= args.only_batch < n):
        raise SystemExit(f"--only-batch must be between 0 and {n - 1}")
    if args.output.rstrip("/") == LEGACY_CLEAN_WINDOWS_S3 and not args.force:
        raise SystemExit(f"--output {LEGACY_CLEAN_WINDOWS_S3} holds the old window-array format (the 200k sample "
                         f"used by the old training script). Use a store folder such as {CLEAN_STORE_S3}, "
                         f"or pass --force.")

    # settings are part of the folder names, so different settings never mix
    daily_tag = f"nb{n:03d}_q{args.quality_filter}_f{args.min_flatline_run}"
    if args.sample_mode == "random_sample":
        daily_tag += f"_rs{args.n_households}s{args.seed}"
    index_tag = (f"{daily_tag}_L{args.lookback_days}_H{args.horizon_days}_S{args.stride_days}"
                 f"_G{args.max_input_fragment}_R{args.recent_clean_days}")
    store = args.output.rstrip("/")
    daily_root, index_root = f"{store}/daily_{daily_tag}", f"{store}/index_{index_tag}"
    print(f"[layout] daily table : {daily_root}/batch=NNN\n[layout] window index: {index_root}/batch=NNN")

    trk = PrepTracking(args, daily_tag, index_tag, store, n)
    with RUN.stage("preflight (memory / disk check)"):
        preflight(args)
    use_s3 = args.input.startswith("s3a://") or args.output.startswith("s3a://")
    with RUN.stage("spark session (starts the JVM; the first run downloads the hadoop-aws jars)"):
        spark = build_spark_session(
            use_s3, cores=args.cores, driver_memory=args.driver_memory, local_dir=args.local_dir,
            shuffle_partitions=args.shuffle_partitions or max(128, 1024 // n), s3_buffer=args.s3_buffer,
            log_level=args.spark_log_level)
        RUN.spark = spark
        RUN.after_jvm_started(args.exit_on_hup)
        RUN.emit("INFO", f"Spark {spark.version} up | master {spark.sparkContext.master} | "
                         f"default parallelism {spark.sparkContext.defaultParallelism}")

    df_raw = None                                  # opened lazily: an index-only run never touches the raw data
    print(f"Daily table + window index: lookback={args.lookback_days} horizon={args.horizon_days} "
          f"stride={args.stride_days} gap limit={args.max_input_fragment} recent-clean={args.recent_clean_days} "
          f"sample_mode={args.sample_mode} batches={n} quality_filter={args.quality_filter} "
          f"min flatline run={args.min_flatline_run}")

    counts = []
    batches = [args.only_batch] if args.only_batch is not None else list(range(n))
    batch_secs, n_new = [], 0                      # n_new: batches whose DAILY table was computed now
    loop_completed = True
    for k, b in enumerate(batches):
        daily_b = f"{daily_root}/batch={b:03d}"
        index_b = f"{index_root}/batch={b:03d}"
        tag = f"[batch {b + 1}/{n}]" if n > 1 else "[run]"
        label = f"batch {b + 1:03d}/{n}" if n > 1 else "run"
        t_batch = time.time()

        with RUN.stage(f"{label}: check for earlier results"):
            daily_done = (not args.rebuild) and path_exists(spark, daily_b + "/_SUCCESS")
            index_done = (daily_done and not args.rebuild_index
                          and path_exists(spark, index_b + "/_SUCCESS"))
        print(f"{tag} daily table: {'present' if daily_done else 'to build'} | "
              f"window index: {'present' if index_done else 'to build'}", flush=True)

        sec = {}
        if not daily_done:
            t_stage = time.time()
            if df_raw is None:
                with RUN.stage("open raw parquet (lists the S3 files)"):
                    print(f"Reading raw intervals from {args.input}")
                    df_raw = spark.read.option("recursiveFileLookup", "true").parquet(args.input)
            # Spark is lazy: the plan is only built here; the clean-up and the S3 upload run in the write stage.
            with RUN.stage(f"{label}: daily table: plan"):
                df_b = df_raw if n == 1 else df_raw.filter(F.pmod(F.hash("meter_id"), F.lit(n)) == b)
                df_daily = build_daily_table(
                    df_b, sample_mode=args.sample_mode, n_households=args.n_households, seed=args.seed,
                    quality_filter=args.quality_filter, min_flatline_run=args.min_flatline_run)
            with RUN.stage(f"{label}: daily table: compute + write to {daily_b}"):
                print(f"{tag} writing the clean daily table to {daily_b}", flush=True)
                df_daily.write.mode("overwrite").parquet(daily_b)
            sec["daily"] = time.time() - t_stage

        if not index_done:
            t_stage = time.time()
            with RUN.stage(f"{label}: window index: plan"):
                df_idx = build_window_index(
                    spark.read.parquet(daily_b),
                    lookback_days=args.lookback_days, horizon_days=args.horizon_days,
                    stride_days=args.stride_days, max_input_fragment_days=args.max_input_fragment,
                    recent_clean_days=args.recent_clean_days)
            with RUN.stage(f"{label}: window index: compute + write to {index_b}"):
                print(f"{tag} writing the window index to {index_b}", flush=True)
                df_idx.write.mode("overwrite").parquet(index_b)
            sec["index"] = time.time() - t_stage

        # windows per origin month, read back from the index that was written (a few columns only)
        with RUN.stage(f"{label}: read back counts"):
            ac = index_month_counts(spark.read.parquet(index_b), args.max_input_fragment)
            ac["batch"] = b
            counts.append(ac)
        print(f"{tag} {int(ac['candidates'].sum()):,} candidate windows | "
              f"{int(ac['windows_before_filters'].sum()):,} complete | "
              f"{int(ac['windows_after_filters'].sum()):,} pass the filters", flush=True)

        if sec:                                    # built in THIS process (a skipped batch was tracked when it was built)
            sec["total"] = time.time() - t_batch
            trk.batch(b, ac, sec)
        if not daily_done or not index_done:
            batch_secs.append(time.time() - t_batch)
        n_new += int(not daily_done)
        if args.max_new_batches and n_new >= args.max_new_batches:
            RUN.emit("INFO", f"--max-new-batches {args.max_new_batches} reached, stopping after {label}")
            loop_completed = k == len(batches) - 1
            break
        todo = len(batches) - k - 1
        if batch_secs and todo:
            avg = sum(batch_secs) / len(batch_secs)
            RUN.emit("INFO", f"{label} took {_hms(batch_secs[-1])} | {todo} batch(es) left | "
                             f"average {_hms(avg)} -> about {_hms(avg * todo)} to go")

    if counts:
        with RUN.stage("write windows_by_origin_month.csv"):
            tot = pd.concat(counts).drop(columns="batch").groupby("origin_month").sum()
            tot["retention"] = (tot["windows_after_filters"]
                                / tot["windows_before_filters"].where(tot["windows_before_filters"] > 0)).round(4)
            tot.sort_index().to_csv("windows_by_origin_month.csv")
            trk.months(tot.sort_index(), os.path.abspath("windows_by_origin_month.csv"), len(counts),
                       loop_completed and args.only_batch is None)
            print(f"\nwindows by origin month saved to {os.path.abspath('windows_by_origin_month.csv')} "
                  f"(total after filters: {int(tot['windows_after_filters'].sum()):,} of "
                  f"{int(tot['windows_before_filters'].sum()):,} complete windows)")

    with RUN.stage("spark stop"):
        RUN.spark = None
        spark.stop()
    print("Done.")


def main():
    args = parse_args()
    RUN.start(args)
    try:
        run(args)
    except KeyboardInterrupt:
        RUN.finish("INTERRUPTED (Ctrl-C)")
        raise SystemExit(130)
    except SystemExit as e:
        if e.code in (None, 0):
            RUN.finish("CLEAN EXIT")
            raise
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
        sig = f" | signal {RUN.last_signal}" if RUN.last_signal else ""
        RUN.finish(f"STOPPED code={e.code!r}{sig}")
        raise SystemExit(e.code if isinstance(e.code, int) else 1)
    except BaseException as e:                               # noqa: BLE001
        traceback.print_exc()
        RUN.finish(f"FAILED {type(e).__name__}: {(str(e).splitlines() or [''])[0][:300]}")
        raise SystemExit(1)
    else:
        RUN.finish("CLEAN EXIT")


if __name__ == "__main__":
    main()