"""Where the pipeline keeps everything between steps: the on-premise MinIO (S3), or a local folder in tests.

PIPELINE_STORE   s3://<bucket>/<prefix>  (MinIO; endpoint and keys from AWS_ENDPOINT_URL / AWS_ACCESS_KEY_ID /
                 AWS_SECRET_ACCESS_KEY / AWS_REGION) or a local path / file:// URI

Every dataset is a folder with Parquet parts and a `_SUCCESS` marker written last; readers refuse folders without it.
A write goes to `<folder>.tmp-<pid>` first and replaces the folder only when complete (re-runs are clean).
"""
import io
import json
import os
import posixpath
import time

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs as pafs


class Store:
    def __init__(self, uri=None):
        uri = uri or os.environ.get("PIPELINE_STORE")
        if not uri:
            raise SystemExit("PIPELINE_STORE is not set (s3://bucket/prefix or a local folder)")
        if uri.startswith("s3://"):
            ep = os.environ.get("AWS_ENDPOINT_URL", "").rstrip("/")
            scheme, host = ep.split("://", 1) if "://" in ep else ("https", ep)
            self.fs = pafs.S3FileSystem(access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                                        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                                        session_token=os.environ.get("AWS_SESSION_TOKEN"),
                                        endpoint_override=host or None, scheme=scheme,
                                        region=os.environ.get("AWS_REGION", "us-east-1"))
            self.root = uri[len("s3://"):].rstrip("/")
        else:
            self.fs = pafs.LocalFileSystem()
            self.root = os.path.abspath(uri[len("file://"):] if uri.startswith("file://") else uri)
        self.uri = uri

    def path(self, *parts):
        return posixpath.join(self.root, *[str(p) for p in parts])

    def exists(self, *parts):
        return self.fs.get_file_info(self.path(*parts)).type != pafs.FileType.NotFound

    def done(self, *parts):
        return self.exists(*parts, "_SUCCESS")

    def write_table(self, df, *parts, meta=None):
        """DataFrame -> <folder>/part-0.parquet + _SUCCESS (with row count and meta); replaces an earlier write."""
        final = self.path(*parts)
        tmp = f"{final}.tmp-{os.getpid()}"
        self._rm(tmp)
        self.fs.create_dir(tmp, recursive=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, posixpath.join(tmp, "part-0.parquet"), filesystem=self.fs, compression="zstd")
        info = dict(rows=int(len(df)), columns=list(df.columns), written_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **(meta or {}))
        with self.fs.open_output_stream(posixpath.join(tmp, "_SUCCESS")) as f:
            f.write(json.dumps(info, default=str).encode())
        self._rm(final)
        self.fs.move(tmp, final)
        return info

    def read_table(self, *parts, columns=None):
        if not self.done(*parts):
            raise FileNotFoundError(f"{self.path(*parts)} has no _SUCCESS (missing or incomplete)")
        files = [f.path for f in self.fs.get_file_info(pafs.FileSelector(self.path(*parts)))
                 if f.path.endswith(".parquet")]
        return pa.concat_tables([pq.read_table(p, filesystem=self.fs, columns=columns) for p in sorted(files)]).to_pandas()

    def success_info(self, *parts):
        with self.fs.open_input_stream(self.path(*parts, "_SUCCESS")) as f:
            return json.loads(f.read().decode() or "{}")

    def read_json(self, *parts, default=None):
        p = self.path(*parts)
        if self.fs.get_file_info(p).type == pafs.FileType.NotFound:
            return default
        with self.fs.open_input_stream(p) as f:
            return json.loads(f.read().decode())

    def write_json(self, obj, *parts):
        p = self.path(*parts)
        self.fs.create_dir(posixpath.dirname(p), recursive=True)
        tmp = p + ".tmp"
        with self.fs.open_output_stream(tmp) as f:
            f.write(json.dumps(obj, indent=2, default=str).encode())
        self.fs.move(tmp, p)

    def list_dirs(self, *parts):
        base = self.path(*parts)
        if self.fs.get_file_info(base).type == pafs.FileType.NotFound:
            return []
        return sorted(posixpath.basename(f.path) for f in self.fs.get_file_info(pafs.FileSelector(base))
                      if f.type == pafs.FileType.Directory and ".tmp-" not in f.path)

    def _rm(self, p):
        if self.fs.get_file_info(p).type != pafs.FileType.NotFound:
            self.fs.delete_dir(p)
