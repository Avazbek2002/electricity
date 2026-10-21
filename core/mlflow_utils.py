"""
Experiment tracking and model registry for every entry point (MLflow), fail-safe.

    Tracker(experiment, uri=None, enabled=None)
        enabled=None: on when a URI is given or MLFLOW_TRACKING_URI is set (and mlflow is installed and the server
        answers); enabled=False: off; enabled=True without a URI: a local SQLite store (~/work/mlflow.db).
        Any tracking error is printed as a warning and never stops the job.

    run_script(main, experiment)      wraps a script's main(): one run with the command line as params, git /
                                      host tags, the printed log, the files in --out-dir as artifacts, and
                                      optional metrics read from a result table.

Credentials come only from the environment: MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (or
MLFLOW_TRACKING_TOKEN). A URI with user:password@ in it is accepted but the credentials are moved to those variables
and never printed.

Data rule (analytical datasets guide): customer rows stay in the analytical environment. Before a table is logged,
files whose header has a customer key column are refused, and rows of node tables with fewer than
MIN_CUSTOMERS households are dropped from the logged copy.
"""
import contextlib
import getpass
import io
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOCAL_STORE = "sqlite:///" + os.path.expanduser("~/work/mlflow.db")
CUSTOMER_COLUMNS = {"customer_key", "series_key", "meter_id", "consumer_id", "customer_id", "account_id",
                    "personal_account", "sub_consumer_id"}
COUNT_COLUMNS = ("households", "n_households", "households_per_day", "customers", "n_customers", "household_count")
MIN_CUSTOMERS = 10
TABLE_EXT = (".csv", ".tsv")
LOG_EXT = (".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".md", ".png", ".svg", ".jpg", ".pdf", ".html", ".log")
SECRET_ENV = ("MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID",
              "VAULT_AWS_SECRET_ACCESS_KEY", "VAULT_AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN")


# ----------------------------------------------------------------------------- small helpers
def _mask(text):
    """Remove credentials from a string (URI user:password@, values of secret environment variables)."""
    s = re.sub(r"(https?://)[^/@\s]+@", r"\1***@", str(text))
    for k in SECRET_ENV:
        v = os.getenv(k)
        if v and len(v) >= 6:
            s = s.replace(v, "***")
    return s


def _clean_uri(uri):
    """Expand ~ and move user:password@ out of an http(s) URI into the environment."""
    uri = str(uri).strip()
    if uri.startswith("sqlite:///~"):
        uri = "sqlite:///" + os.path.expanduser(uri[len("sqlite:///"):])
    elif uri.startswith("~"):
        uri = os.path.expanduser(uri)
    m = re.match(r"(https?://)([^/@\s]+)@(.*)", uri)
    if m:
        user, _, pw = m.group(2).partition(":")
        from urllib.parse import unquote
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", unquote(user))
        if pw:
            os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", unquote(pw))
        print("[mlflow] credentials in the URI were moved to MLFLOW_TRACKING_USERNAME / _PASSWORD; "
              "put them there instead of in the URI")
        uri = m.group(1) + m.group(3)
    return uri.rstrip("/")


def _key(k):
    """MLflow keys allow letters, digits, _ - . space /; anything else becomes _."""
    return re.sub(r"[^\w\-. /]", "_", str(k))[:250]


def _num(v):
    try:
        f = float(v.item() if hasattr(v, "item") else v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                                  # noqa: BLE001
        return ""


def table_guard(path):
    """(ok, path_to_log, note). Refuses customer-level tables; drops node rows under MIN_CUSTOMERS households."""
    if not path.lower().endswith(TABLE_EXT):
        return True, path, ""
    sep = "\t" if path.lower().endswith(".tsv") else ","
    try:
        with open(path, newline="") as fh:
            header = [c.strip().strip('"').lower() for c in fh.readline().rstrip("\r\n").split(sep)]
    except (OSError, UnicodeDecodeError):
        return True, path, ""
    bad = sorted(CUSTOMER_COLUMNS & set(header))
    if bad:
        return False, None, f"has customer key column(s) {bad}"
    counts = [c for c in COUNT_COLUMNS if c in header]
    if not counts:
        return True, path, ""
    import pandas as pd
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception:                                                  # noqa: BLE001
        return True, path, ""
    col = next(c for c in df.columns if str(c).strip().lower() == counts[0])
    small = pd.to_numeric(df[col], errors="coerce") < MIN_CUSTOMERS
    if not small.any():
        return True, path, ""
    d = tempfile.mkdtemp(prefix="mlflow_guard_")
    out = os.path.join(d, os.path.basename(path))
    df[~small].to_csv(out, index=False, sep=sep)
    return True, out, f"{int(small.sum())} row(s) with {col} < {MIN_CUSTOMERS} left out"


# ----------------------------------------------------------------------------- the tracker
class Tracker:
    """Thin, fail-safe wrapper over the MLflow fluent API and MlflowClient. Every method is a no-op when off."""

    def __init__(self, experiment, uri=None, enabled=None):
        self.experiment, self.enabled, self.client, self.mlflow = experiment, False, None, None
        self.root_run_id, self.experiment_id, self.uri = None, None, ""
        self._warned = 0
        uri = uri or os.getenv("MLFLOW_TRACKING_URI")
        if enabled is False or (enabled is None and not uri) or os.getenv("MLFLOW_DISABLE") == "1":
            return
        try:
            import mlflow
        except ImportError:
            print("[mlflow] not installed: tracking off")
            return
        uri = _clean_uri(uri or LOCAL_STORE)
        if uri.startswith("sqlite:///"):
            os.makedirs(os.path.dirname(uri[len("sqlite:///"):]) or ".", exist_ok=True)
        mlflow.set_tracking_uri(uri)
        os.environ["MLFLOW_TRACKING_URI"] = uri              # worker processes and child scripts see the same server
        saved = {k: os.environ.get(k) for k in ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_TIMEOUT")}
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"], os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "1", "20"
        try:                                                   # first contact: fail fast if the server is down
            self.client = mlflow.MlflowClient()
            exp = self.client.get_experiment_by_name(experiment)
            if exp is None:                                    # local store: artifacts next to the database
                loc = None
                if uri.startswith("sqlite:///"):
                    loc = "file://" + os.path.join(os.path.dirname(uri[len("sqlite:///"):]), "mlartifacts")
                self.experiment_id = self.client.create_experiment(experiment, artifact_location=loc)
            else:
                self.experiment_id = exp.experiment_id
        except Exception as e:                                 # noqa: BLE001
            print(f"[mlflow] WARNING: {_mask(uri)} not usable ({type(e).__name__}: {_mask(str(e))[:300]}); "
                  f"tracking off for this run")
            return
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.mlflow, self.uri, self.enabled = mlflow, uri, True
        print(f"[mlflow] tracking to {_mask(uri)} | experiment {experiment!r}")

    # ---------------------------------------------------------------- plumbing
    def _warn(self, msg):
        self._warned += 1
        if self._warned <= 20:
            print(f"[mlflow] WARNING: {_mask(msg)[:400]}")
        elif self._warned == 21:
            print("[mlflow] (further tracking warnings suppressed)")

    def _safe(self, what, fn, *a, **k):
        if not self.enabled:
            return None
        try:
            return fn(*a, **k)
        except Exception as e:                                 # noqa: BLE001
            self._warn(f"{what} failed: {type(e).__name__}: {e}")
            return None

    def _rid(self, run_id=None):
        if run_id:
            return run_id
        r = self.mlflow.active_run() if self.enabled else None
        return r.info.run_id if r else self.root_run_id

    # ---------------------------------------------------------------- runs
    @contextlib.contextmanager
    def run(self, name, nested=False, tags=None):
        """with tr.run("train ...") as run_id: ...  (run_id None when off). Nested automatically inside a run."""
        if not self.enabled:
            yield None
            return
        r = self._safe("start run", self.mlflow.start_run, run_name=name, experiment_id=self.experiment_id,
                       nested=bool(nested or self.mlflow.active_run()),
                       tags={_key(k): _mask(str(v))[:5000] for k, v in (tags or {}).items()})
        if r is None:
            yield None
            return
        if self.root_run_id is None:
            self.root_run_id = r.info.run_id
        status = "FINISHED"
        try:
            yield r.info.run_id
        except KeyboardInterrupt:
            status = "KILLED"
            raise
        except SystemExit as e:
            status = "FINISHED" if e.code in (0, None) else "FAILED"
            raise
        except BaseException:
            status = "FAILED"
            raise
        finally:
            self._safe("end run", self.mlflow.end_run, status=status)

    def begin(self, name, resume_run_id=None, tags=None):
        """Start (or resume) a run outside a with-block; returns its id. Close with end()."""
        def go():
            if resume_run_id:
                r = self.mlflow.start_run(run_id=resume_run_id)
            else:
                r = self.mlflow.start_run(run_name=name, experiment_id=self.experiment_id,
                                          nested=bool(self.mlflow.active_run()),
                                          tags={_key(k): _mask(str(v))[:5000] for k, v in (tags or {}).items()})
            return r.info.run_id
        rid = self._safe("start run", go)
        if rid and self.root_run_id is None:
            self.root_run_id = rid
        return rid

    def end(self, status="FINISHED"):
        if self.enabled:
            self._safe("end run", self.mlflow.end_run, status=status)

    # ---------------------------------------------------------------- logging
    def params(self, d, run_id=None):
        if not self.enabled or not d:
            return
        from mlflow.entities import Param
        rid = self._rid(run_id)
        items = [(_key(k), str(v)[:6000]) for k, v in d.items()]

        def go(chunk):
            try:
                self.client.log_batch(rid, params=[Param(k, v) for k, v in chunk])
            except Exception:                                  # noqa: BLE001  (e.g. resumed run: value changed)
                have = self.client.get_run(rid).data.params
                new = [(k, v) for k, v in chunk if k not in have]
                changed = {f"param_changed.{k}": v for k, v in chunk if k in have and have[k] != v}
                if new:
                    self.client.log_batch(rid, params=[Param(k, v) for k, v in new])
                if changed:
                    self.tags(changed, run_id=rid)
        for i in range(0, len(items), 100):
            self._safe("log params", go, items[i:i + 100])

    def metrics(self, d, step=None, run_id=None):
        if not self.enabled or not d:
            return
        from mlflow.entities import Metric
        rid, ts = self._rid(run_id), int(time.time() * 1000)
        ms = [Metric(_key(k), v, ts, int(step or 0)) for k, v in ((k, _num(v)) for k, v in d.items()) if v is not None]
        for i in range(0, len(ms), 1000):
            self._safe("log metrics", self.client.log_batch, rid, metrics=ms[i:i + 1000])

    def curve(self, key, values, run_id=None):
        """One metric over steps 0..n-1 (training curves)."""
        self.series(key, [(None, i, v) for i, v in enumerate(values)], run_id=run_id)

    def series(self, key, points, run_id=None):
        """points: (unix seconds or None, step, value); drawn over real time in the UI when seconds are given."""
        if not self.enabled:
            return
        from mlflow.entities import Metric
        rid, now = self._rid(run_id), int(time.time() * 1000)
        ms = [Metric(_key(key), f, int(t * 1000) if t is not None else now, int(s))
              for t, s, v in points if (f := _num(v)) is not None]
        for i in range(0, len(ms), 1000):
            self._safe("log series", self.client.log_batch, rid, metrics=ms[i:i + 1000])

    def tags(self, d, run_id=None):
        if not self.enabled or not d:
            return
        from mlflow.entities import RunTag
        rid = self._rid(run_id)
        self._safe("set tags", self.client.log_batch, rid,
                   tags=[RunTag(_key(k), _mask(str(v))[:5000]) for k, v in d.items()])

    def artifact(self, path, subdir=None, run_id=None):
        """Log one file (tables pass the customer-data guard first)."""
        if not self.enabled:
            return
        if not path or not os.path.exists(path):
            self._warn(f"artifact {path} does not exist")
            return
        ok, src, note = table_guard(path)
        if not ok:
            print(f"[mlflow] NOT logged {os.path.basename(path)}: {note}")
            return
        if note:
            print(f"[mlflow] {os.path.basename(path)}: {note}")
        self._safe(f"log artifact {os.path.basename(path)}", self.client.log_artifact, self._rid(run_id), src, subdir)

    def artifacts_dir(self, folder, subdir=None, run_id=None, ext=LOG_EXT, max_file_mb=50, max_total_mb=500,
                      everything=False):
        """Log a folder's files (recursively), keeping its layout. everything=True: every file (model folders),
        otherwise only report-like extensions; files over max_file_mb and anything past max_total_mb are skipped."""
        if not self.enabled or not folder or not os.path.isdir(folder):
            return
        total, skipped = 0.0, []
        for base, _, files in os.walk(folder):
            for f in sorted(files):
                p = os.path.join(base, f)
                mb = os.path.getsize(p) / 1e6
                if not everything and (not f.lower().endswith(ext) or f.startswith(".")):
                    continue
                if (not everything and mb > max_file_mb) or total + mb > max_total_mb:
                    skipped.append(f"{f} ({mb:.0f} MB)")
                    continue
                rel = os.path.relpath(base, folder)
                sd = "/".join(x for x in (subdir, None if rel == "." else rel) if x) or None
                self.artifact(p, subdir=sd, run_id=run_id)
                total += mb
        if skipped:
            print(f"[mlflow] not logged (size): {', '.join(skipped[:10])}{' ...' if len(skipped) > 10 else ''}")

    def dataframe(self, df, name, run_id=None):
        """Log a DataFrame as a CSV artifact at `name` (e.g. 'evaluation/test_system_daily.csv')."""
        if not self.enabled or df is None:
            return
        d = tempfile.mkdtemp(prefix="mlflow_df_")
        p = os.path.join(d, os.path.basename(name))
        import pandas as pd
        df.to_csv(p, index=not isinstance(df.index, pd.RangeIndex))
        self.artifact(p, subdir=os.path.dirname(name) or None, run_id=run_id)

    def text(self, text, name, run_id=None):
        if not self.enabled:
            return
        self._safe("log text", self.client.log_text, self._rid(run_id), _mask(text), name)

    def system_tags(self):
        commit = _git("rev-parse", "HEAD")
        return {"git_commit": commit or "unknown", "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))).lower(),
                "host": socket.gethostname(), "user": getpass.getuser(), "python": platform.python_version(),
                "conda_env": os.path.basename(os.getenv("CONDA_PREFIX", "")) or "-",
                "command": _mask(" ".join(shlex.quote(a) for a in sys.argv))[:4000],
                "mlflow.source.git.commit": commit or "unknown"}

    # ---------------------------------------------------------------- lookup
    def get(self, run_id):
        return self._safe("get run", self.client.get_run, run_id)

    def find_run(self, filter_string, experiment=None):
        """Latest run in this experiment (or `experiment`, a name) matching an MLflow filter string, or None."""
        if not self.enabled:
            return None
        if experiment is not None:
            return self.find_run_in(experiment, filter_string)
        rs = self._safe("search runs", self.client.search_runs, [self.experiment_id], filter_string,
                        order_by=["attributes.start_time DESC"], max_results=1)
        return rs[0] if rs else None

    def find_run_in(self, experiment, filter_string):
        if not self.enabled:
            return None
        exp = self._safe("get experiment", self.client.get_experiment_by_name, experiment)
        if exp is None:
            return None
        rs = self._safe("search runs", self.client.search_runs, [exp.experiment_id], filter_string,
                        order_by=["attributes.start_time DESC"], max_results=1)
        return rs[0] if rs else None

    def download_artifact(self, run_id, path):
        """Local path of a run's artifact, or None."""
        if not self.enabled or not run_id:
            return None
        import mlflow.artifacts
        return self._safe(f"download {path}", mlflow.artifacts.download_artifacts, run_id=run_id, artifact_path=path,
                          dst_path=tempfile.mkdtemp(prefix="mlflow_dl_"))

    # ---------------------------------------------------------------- registry
    def ensure_registered_model(self, name, description=None, tags=None):
        if not self.enabled:
            return None

        def go():
            try:
                return self.client.get_registered_model(name)
            except Exception:                                  # noqa: BLE001  (does not exist yet)
                return self.client.create_registered_model(name, tags={k: str(v) for k, v in (tags or {}).items()},
                                                           description=description)
        return self._safe("registered model", go)

    def register(self, model_uri, name, tags=None, description=None, wait_seconds=120):
        """New version of `name` from a logged model (models:/m-..., runs:/<id>/<path>) or an artifact folder URI.
        Returns the version number, or None."""
        if not self.enabled or not model_uri:
            return None

        def go():
            self.ensure_registered_model(name)
            t = {k: str(v)[:5000] for k, v in (tags or {}).items()}
            try:
                mv = self.mlflow.register_model(model_uri, name, tags=t)
            except Exception as e:                             # noqa: BLE001  (plain folder in MLflow 3, ...)
                src, run_id = model_uri, None
                m = re.match(r"runs:/([^/]+)/(.*)", model_uri)
                if m:
                    run_id = m.group(1)
                    src = self.client.get_run(run_id).info.artifact_uri.rstrip("/") + "/" + m.group(2)
                print(f"[mlflow] register_model: {type(e).__name__}; registering the artifact folder directly")
                mv = self.client.create_model_version(name, source=src, run_id=run_id, tags=t,
                                                      description=description)
            if description and getattr(mv, "description", None) != description:
                self.client.update_model_version(name, mv.version, description=description)
            t0 = time.time()
            while getattr(mv, "status", "READY") not in ("READY", "FAILED_REGISTRATION") and time.time() - t0 < wait_seconds:
                time.sleep(2)
                mv = self.client.get_model_version(name, mv.version)
            return int(mv.version)
        return self._safe(f"register {name}", go)

    def set_version_tags(self, name, version, tags):
        for k, v in (tags or {}).items():
            self._safe("version tag", self.client.set_model_version_tag, name, str(version), _key(k), str(v)[:5000])

    def set_alias(self, name, alias, version):
        self._safe(f"alias {alias}", self.client.set_registered_model_alias, name, alias, str(version))

    def delete_alias(self, name, alias):
        self._safe(f"delete alias {alias}", self.client.delete_registered_model_alias, name, alias)

    def version_by_alias(self, name, alias):
        """(version, run_id) of the version holding `alias`, or (None, None)."""
        if not self.enabled:
            return None, None
        try:
            mv = self.client.get_model_version_by_alias(name, alias)
            return int(mv.version), mv.run_id
        except Exception:                                      # noqa: BLE001  (no such model / alias)
            return None, None

    def version_info(self, name, version):
        return self._safe("get version", self.client.get_model_version, name, str(version))


# ----------------------------------------------------------------------------- scripts
class _Tee(io.TextIOBase):
    def __init__(self, stream, path):
        self.stream, self.fh = stream, open(path, "w", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.stream.fileno()


def _argv_params(argv):
    """--key value / --key=value / --flag / positional -> params."""
    out, pos, i = {}, 0, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k, eq, v = a[2:].partition("=")
            if not eq:
                vals = []
                while i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    vals.append(argv[i + 1])
                    i += 1
                v = " ".join(vals) if vals else "true"
            out[k.replace("-", "_")] = _mask(v)
        else:
            out[f"arg{pos}"] = _mask(a)
            pos += 1
        i += 1
    return out


def _flag_value(argv, flags):
    for i, a in enumerate(argv):
        for f in flags:
            if a == f and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(f + "="):
                return a.split("=", 1)[1]
    return None


def table_metrics(path, keys, values, where=None, prefix=""):
    """Metrics from a result CSV: one metric per row and value column, named <prefix><key1>/<key2>/<value>_h<h>.
    `keys` are the columns that name the row (a 'horizon' column becomes the _h<h> suffix); `where`: {col: value}."""
    import pandas as pd
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    for c, v in (where or {}).items():
        if c in df:
            df = df[df[c].astype(str) == str(v)]
    out = {}
    for r in df.itertuples(index=False):
        r = r._asdict()
        name = "/".join(str(r[k]) for k in keys if k != "horizon" and k in r)
        h = f"_h{r['horizon']}" if "horizon" in keys and "horizon" in r else ""
        for v in values:
            if v in r and _num(r[v]) is not None:
                out[f"{prefix}{name}/{v}{h}".replace("//", "/")] = r[v]
    return out


def run_script(main, experiment="electricity/analysis", stage="analysis", out_flags=("--out-dir", "--work-dir", "--out"),
               metrics_from=None, model_files=False):
    """Run main() inside one MLflow run. Extra flags understood (and removed before main sees them):
    --no-mlflow, --mlflow-uri URI, --mlflow-experiment NAME, --mlflow-run-name NAME.
    metrics_from(out_dir) -> {name: value} logs result metrics; model_files=True logs every file in the out dir."""
    argv = sys.argv[1:]
    opts = {}
    for flag in ("--mlflow-uri", "--mlflow-experiment", "--mlflow-run-name"):
        v = _flag_value(argv, (flag,))
        if v is not None:
            opts[flag] = v
            argv = [a for i, a in enumerate(argv) if not (a == flag or a.startswith(flag + "=")
                                                           or (i > 0 and argv[i - 1] == flag))]
    off = "--no-mlflow" in argv
    argv = [a for a in argv if a != "--no-mlflow"]
    sys.argv = sys.argv[:1] + argv
    if any(a in ("-h", "--help") for a in argv):
        return main()
    script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    tr = Tracker(opts.get("--mlflow-experiment", experiment), uri=opts.get("--mlflow-uri"),
                 enabled=False if off else None)
    if not tr.enabled:
        return main()
    out_dir = _flag_value(argv, out_flags)
    out_dir = os.path.expanduser(out_dir) if out_dir else None
    if out_dir and out_dir.lower().endswith(LOG_EXT):           # --out pointing at a file
        out_dir = os.path.dirname(out_dir) or "."
    log_path = os.path.join(tempfile.mkdtemp(prefix="mlflow_log_"), "stdout.txt")
    name = opts.get("--mlflow-run-name") or f"{script} {time.strftime('%Y-%m-%d %H:%M')}"
    t0 = time.time()
    old_out, old_err = sys.stdout, sys.stderr
    with tr.run(name, tags={"stage": stage, "script": script, **tr.system_tags()}):
        tr.params(_argv_params(argv))
        sys.stdout = _Tee(old_out, log_path)
        sys.stderr = _Tee(old_err, log_path + ".err")
        try:
            result = main()
        finally:
            sys.stdout.flush(), sys.stderr.flush()
            sys.stdout, sys.stderr = old_out, old_err
            tr.metrics({"seconds": time.time() - t0})
            if out_dir and os.path.isdir(out_dir):
                tr.artifacts_dir(out_dir, subdir="outputs", everything=model_files,
                                 max_total_mb=2000 if model_files else 500)
            if metrics_from and out_dir:
                try:
                    tr.metrics(metrics_from(out_dir))
                except Exception as e:                           # noqa: BLE001
                    print(f"[mlflow] WARNING: result metrics not read ({type(e).__name__}: {e})")
            tr.artifact(log_path, subdir="logs")
            if os.path.getsize(log_path + ".err"):
                tr.artifact(log_path + ".err", subdir="logs")
    return result
