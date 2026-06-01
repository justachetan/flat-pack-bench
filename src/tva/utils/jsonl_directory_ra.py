from __future__ import annotations
import json, io
import os.path as osp
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union, Tuple, Dict, List, Iterable

from pycocotools.mask import decode

Pathish = str | Path
Processor = Callable[[dict[str, Any]], Optional[dict[str, Any]]]

class JsonlDirRA:
    """
    Random-access streaming processor over a DIRECTORY of uncompressed .jsonl files.

    - Treats all .jsonl files in the directory (in the format `clip_<clip_index>.jsonl` 
        sorted by name) as one logical stream.
    - Builds per-file sparse byte-offset indices every `index_every` lines (default 1).
    - Supports optional subsampling via `skip`, while always keeping lines where
      `prompt_frame_idx == frame_idx`.
    - Random access (`__getitem__`) and streaming iteration operate on the
      subsampled view when `skip > 1`.
    - Accepts `retain_indices` to force specific raw line indices into the
      subsampled view regardless of `skip`.
    - Supports:
        * Iteration over entire corpus (lazy, streaming)
        * Global seeks:     iter_from(global_raw_index), ra[i], ra[i:j]
        * Per-file seeks:   get_at(file_key, local_line) via ra[(file_key, local_line)]
            - file_key can be an integer file index or the filename string.
    - “Raw line” means line numbers in source files before filtering/processing.

    Notes:
      * Only uncompressed .jsonl files are supported for fast seeking.
      * If you need compression + seeking, consider Zstandard with seekable frames.
    """

    def __init__(
        self,
        dir_path: Pathish,
        process: Processor = lambda d: decode_jsonl_line(d),
        *,
        encoding: str = "utf-8",
        skip_errors: bool = True,
        index_every: int = 1,
        file_glob: str = "*.jsonl",
        skip: int = 1,
        retain_indices: Optional[Iterable[int]] = None,
    ):
        self.dir_path = Path(dir_path)
        self.encoding = encoding
        self.process = process
        self.skip_errors = skip_errors
        self._index_every = max(1, index_every)
        if skip < 1:
            raise ValueError("skip must be >= 1")
        self._skip = int(skip)
        self._retain_indices_spec = (
            tuple(int(i) for i in retain_indices) if retain_indices is not None else None
        )
        self._retain_indices: set[int] = set()
        self._full_len_cache: Optional[int] = None

        if not self.dir_path.is_dir():
            raise ValueError(f"Not a directory: {self.dir_path}")

        # Discover files
        # import ipdb; ipdb.set_trace()
        files = sorted(self.dir_path.glob(file_glob), 
                       key=lambda p: int(osp.basename(p.name).split(".")[0].split("_")[-1]))
        files = [p for p in files if p.is_file() and p.suffix == ".jsonl"]
        if not files:
            raise ValueError(f"No .jsonl files found in {self.dir_path} with pattern {file_glob}")
        self._files: List[Path] = files
        self._file_name_to_idx: Dict[str, int] = {p.name: i for i, p in enumerate(self._files)}

        # Per-file indices and state
        self._file_offsets: List[List[int]] = []   # per-file list of checkpoint byte offsets
        self._file_line_counts: List[int] = []     # number of raw lines per file
        self._fh: Dict[int, io.BufferedReader] = {}  # lazily opened file handles keyed by file idx

        # Build indices
        self._build_indices()
        self._refresh_line_metadata()
        self._skip_scan_fh: Optional[io.BufferedReader] = None
        self._reset_skip_state()
        self._init_retain_indices()

    # ---------- indexing builders ----------
    def _open_file(self, fi: int) -> io.BufferedReader:
        fh = self._fh.get(fi)
        if fh is None or fh.closed:
            fh = self._files[fi].open("rb", buffering=1024 * 1024)
            self._fh[fi] = fh
        return fh

    def _close_all(self):
        for fh in self._fh.values():
            try:
                if fh and not fh.closed:
                    fh.close()
            except Exception:
                pass
        self._fh.clear()

    def _build_indices(self):
        self._file_offsets.clear()
        self._file_line_counts.clear()

        for fi, path in enumerate(self._files):
            fh = self._open_file(fi)
            fh.seek(0)
            offsets: List[int] = []
            line_no = 0
            while True:
                pos = fh.tell()
                bline = fh.readline()
                if not bline:
                    break
                if line_no % self._index_every == 0:
                    offsets.append(pos)
                line_no += 1
            self._file_offsets.append(offsets)
            self._file_line_counts.append(line_no)
            fh.seek(0)

    def _refresh_line_metadata(self) -> None:
        """Recompute cumulative line counts after (re)building per-file indices."""
        self._cum_counts = [0]
        running = 0
        for cnt in self._file_line_counts:
            running += cnt
            self._cum_counts.append(running)
        self._total_raw_lines = running

    # ---------- properties ----------
    @property
    def files(self) -> List[Path]:
        return self._files

    @property
    def total_files(self) -> int:
        return len(self._files)

    @property
    def file_line_counts(self) -> List[int]:
        """Raw line counts for each file (before processing/filtering)."""
        return self._file_line_counts

    @property
    def total_raw_lines(self) -> int:
        return self._total_raw_lines

    # ---------- helpers: mapping & seeking ----------
    def _global_to_file_local(self, global_idx: int) -> Tuple[int, int]:
        """
        Map global raw index -> (file_idx, local_line).
        """
        if global_idx < 0 or global_idx >= self.total_raw_lines:
            raise IndexError("global index out of range")
        # binary search over cumulative counts
        lo, hi = 0, len(self._cum_counts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum_counts[mid] <= global_idx < self._cum_counts[mid + 1]:
                fi = mid
                local = global_idx - self._cum_counts[mid]
                return fi, local
            if global_idx < self._cum_counts[mid]:
                hi = mid
            else:
                lo = mid + 1
        # fallback (shouldn't hit)
        fi = max(0, min(len(self._files) - 1, lo - 1))
        local = global_idx - self._cum_counts[fi]
        return fi, local

    def _seek_file_line(self, fi: int, local_line: int) -> io.BufferedReader:
        """
        Seek file `fi` to start of local_line using nearest checkpoint.
        Returns the (positioned) file handle.
        """
        if fi < 0 or fi >= self.total_files:
            raise IndexError("file index out of range")
        if local_line < 0 or local_line >= self._file_line_counts[fi]:
            raise IndexError("local line out of range")

        fh = self._open_file(fi)
        # find nearest checkpoint
        idx_every = self._index_every
        ck = local_line // idx_every
        ck_line = ck * idx_every
        fh.seek(self._file_offsets[fi][ck])
        # fast-forward to exact line
        for _ in range(local_line - ck_line):
            if not fh.readline():
                break
        return fh

    def reset_index_every(self, index_every: int, rebuild: bool = True) -> None:
        """
        Update `index_every` and optionally rebuild sparse indices.
        Rebuilding is required whenever the value changes to keep offsets valid.
        """
        if index_every < 1:
            raise ValueError("index_every must be >= 1")

        new_every = int(index_every)
        prev_every = self._index_every
        changed = new_every != prev_every

        if changed and not rebuild:
            raise ValueError("rebuild must be True when changing index_every")

        self._index_every = new_every
        if rebuild or changed:
            self._build_indices()
            self._refresh_line_metadata()
            self._reset_skip_state()
            self._init_retain_indices()

    def _reset_skip_state(self) -> None:
        fh = getattr(self, "_skip_scan_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
        self._skip_cache = []
        self._skip_records = []
        self._skip_scan_pos = 0
        self._skip_scan_fi = 0
        self._skip_scan_local = 0
        self._skip_scan_fh = None
        self._skip_next_regular_state = 0
        self._skip_scan_complete = False
        self._skip_scan_initialized = False
        self._retain_indices = set()
        self._full_len_cache = None

    def _init_retain_indices(self) -> None:
        spec = self._retain_indices_spec
        self._retain_indices.clear()
        if not spec:
            return
        total = self._total_raw_lines
        if total <= 0:
            return
        for raw_idx in spec:
            idx = int(raw_idx)
            if idx < 0:
                idx = total + idx
            if 0 <= idx < total:
                self._retain_indices.add(idx)

    # ---------- skip helpers ----------
    def _open_skip_scan_handle(self, fi: int, local_line: int) -> io.BufferedReader:
        fh = self._skip_scan_fh
        if fh is not None and not fh.closed:
            fh.close()
        fh = self._files[fi].open("rb", buffering=1024 * 1024)
        idx_every = self._index_every
        ck = local_line // idx_every
        ck_line = ck * idx_every
        offsets = self._file_offsets[fi]
        if ck >= len(offsets):
            ck = max(0, len(offsets) - 1)
            ck_line = ck * idx_every
        fh.seek(offsets[ck])
        for _ in range(local_line - ck_line):
            if not fh.readline():
                break
        self._skip_scan_fh = fh
        return fh

    def _prepare_skip_scan(self) -> None:
        if self._skip_scan_complete or self._skip <= 1:
            return
        if self._skip_scan_pos >= self.total_raw_lines:
            self._skip_scan_complete = True
            return
        if not self._skip_scan_initialized:
            fi, local = self._global_to_file_local(self._skip_scan_pos)
            self._skip_scan_fi = fi
            self._skip_scan_local = local
            self._open_skip_scan_handle(fi, local)
            self._skip_scan_initialized = True
            if self._skip_next_regular_state < self._skip_scan_pos:
                rem = self._skip_scan_pos % self._skip
                if rem == 0:
                    self._skip_next_regular_state = self._skip_scan_pos
                else:
                    self._skip_next_regular_state = self._skip_scan_pos + (self._skip - rem)
        elif self._skip_scan_fh is None or self._skip_scan_fh.closed:
            self._open_skip_scan_handle(self._skip_scan_fi, self._skip_scan_local)

    def _scan_next_skip_item(self) -> Optional[Tuple[int, dict[str, Any]]]:
        if self._skip <= 1 or self._skip_scan_complete:
            return None

        total = self.total_raw_lines
        while self._skip_scan_pos < total:
            self._prepare_skip_scan()
            fh = self._skip_scan_fh
            if fh is None:
                self._skip_scan_complete = True
                return None

            bline = fh.readline()
            if not bline:
                self._skip_scan_fi += 1
                if self._skip_scan_fi >= self.total_files:
                    self._skip_scan_complete = True
                    self._skip_scan_pos = total
                    fh.close()
                    self._skip_scan_fh = None
                    return None
                self._skip_scan_local = 0
                self._open_skip_scan_handle(self._skip_scan_fi, 0)
                continue

            cur_idx = self._skip_scan_pos
            self._skip_scan_pos += 1
            self._skip_scan_local += 1

            s = bline.strip()
            out = None
            forced_match = False
            if s:
                try:
                    obj = json.loads(s.decode(self.encoding))
                    forced_match = obj.get("prompt_frame_idx") == obj.get("frame_idx")
                    out = self.process(obj)
                except Exception:
                    if not self.skip_errors:
                        raise
                    out = None

            next_regular = self._skip_next_regular_state
            include_regular = False
            while cur_idx > next_regular:
                next_regular += self._skip
            if cur_idx == next_regular:
                include_regular = True
                next_regular += self._skip

            explicit_keep = cur_idx in self._retain_indices
            keep = out is not None and (include_regular or forced_match or explicit_keep)
            self._skip_next_regular_state = next_regular

            if keep:
                assert out is not None  # for type checkers
                self._skip_cache.append(cur_idx)
                self._skip_records.append(out)
                return cur_idx, out

        self._skip_scan_complete = True
        if self._skip_scan_fh is not None:
            try:
                self._skip_scan_fh.close()
            except Exception:
                pass
            self._skip_scan_fh = None
        return None

    def _ensure_skip_cache(self, logical_idx: int) -> bool:
        if self._skip <= 1:
            return True
        while len(self._skip_records) <= logical_idx and not self._skip_scan_complete:
            if self._scan_next_skip_item() is None:
                break
        return len(self._skip_records) > logical_idx

    def _ensure_all_skip_cached(self) -> None:
        if self._skip <= 1:
            return
        while not self._skip_scan_complete:
            if self._scan_next_skip_item() is None:
                break

    def _iter_skip_logical_range(self, start: int, stop: Optional[int]) -> Iterator[dict[str, Any]]:
        if start < 0:
            raise ValueError("start must be non-negative")
        idx = start
        while stop is None or idx < stop:
            while len(self._skip_records) <= idx:
                if self._scan_next_skip_item() is None:
                    return
            yield self._skip_records[idx]
            idx += 1

    # ---------- iteration helpers ----------
    def _iter_processed_range(
        self,
        start: int,
        stop: Optional[int],
        *,
        apply_skip: bool,
    ) -> Iterator[dict[str, Any]]:
        total = self.total_raw_lines
        if total == 0:
            return
        if start < 0:
            start = 0
        if start >= total:
            return
        if stop is None or stop > total:
            stop = total
        if stop <= start:
            return

        fi, local = self._global_to_file_local(start)
        fh = self._seek_file_line(fi, local)
        cur_idx = start
        if apply_skip and self._skip > 1:
            rem = cur_idx % self._skip
            if rem == 0:
                next_regular = cur_idx
            else:
                next_regular = cur_idx + (self._skip - rem)
        else:
            next_regular = None

        while cur_idx < stop:
            bline = fh.readline()
            if not bline:
                fi += 1
                if fi >= self.total_files:
                    break
                fh = self._open_file(fi)
                fh.seek(0)
                continue

            s = bline.strip()
            out = None
            forced_match = False
            if s:
                try:
                    obj = json.loads(s.decode(self.encoding))
                    forced_match = obj.get("prompt_frame_idx") == obj.get("frame_idx")
                    out = self.process(obj)
                except Exception:
                    if not self.skip_errors:
                        raise
                    out = None

            if apply_skip and self._skip > 1:
                include_regular = False
                while next_regular is not None and cur_idx > next_regular:
                    next_regular += self._skip
                if next_regular is not None and cur_idx == next_regular:
                    include_regular = True
                    next_regular += self._skip
                explicit_keep = cur_idx in self._retain_indices
                should_emit = out is not None and (include_regular or forced_match or explicit_keep)
            else:
                should_emit = out is not None

            if should_emit:
                yield out

            cur_idx += 1

    # ---------- iteration ----------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate processed records across all files from the start."""
        if self._skip > 1:
            yield from self._iter_skip_logical_range(0, None)
        else:
            yield from self._iter_processed_range(0, None, apply_skip=False)

    def iter_from(self, global_raw_start: int) -> Iterator[dict[str, Any]]:
        """
        Iterate processed records starting at a GLOBAL raw line index.
        """
        yield from self._iter_processed_range(
            global_raw_start,
            None,
            apply_skip=self._skip > 1,
        )

    def iter_slice(self, start: int, stop: int) -> Iterator[dict[str, Any]]:
        """
        Iterate processed records whose GLOBAL raw index is in [start, stop).
        """
        if stop is not None:
            stop = max(0, min(stop, self.total_raw_lines))
        start = max(0, start)
        yield from self._iter_processed_range(
            start,
            stop,
            apply_skip=self._skip > 1,
        )

    # ---------- accessors ----------
    def get(self, global_raw_index: int) -> Optional[dict[str, Any]]:
        """
        Return processed record at GLOBAL raw index (or None if filtered/blank/bad).
        """
        if global_raw_index < 0:
            global_raw_index = self.total_raw_lines + global_raw_index
        if global_raw_index < 0 or global_raw_index >= self.total_raw_lines:
            return None
        fi, local = self._global_to_file_local(global_raw_index)
        return self.get_at(fi, local)

    def get_at(self, file_key: Union[int, str], local_line: int) -> Optional[dict[str, Any]]:
        """
        Return processed record at (file_key, local_line).
          - file_key: int file index (0..total_files-1) or filename string.
        """
        if isinstance(file_key, str):
            fi = self._file_name_to_idx.get(file_key)
            if fi is None:
                raise KeyError(f"Unknown file name: {file_key}")
        else:
            fi = int(file_key)

        if fi < 0 or fi >= self.total_files:
            raise IndexError("file index out of range")
        if local_line < 0 or local_line >= self._file_line_counts[fi]:
            raise IndexError("local line out of range")

        fh = self._seek_file_line(fi, local_line)
        bline = fh.readline()
        if not bline:
            return None
        s = bline.strip()
        if not s:
            return None
        try:
            obj = json.loads(s.decode(self.encoding))
            out = self.process(obj)
            return out
        except Exception:
            if not self.skip_errors:
                raise
            return None

    def close(self):
        self._close_all()
        if self._skip_scan_fh is not None:
            try:
                self._skip_scan_fh.close()
            except Exception:
                pass
            self._skip_scan_fh = None

    # ---------- pythonic indexing ----------
    def __getitem__(self, key):
        """
        Indexing support:
          - ra[i]            : processed record at GLOBAL raw index i (int)
          - ra[i:j]          : iterator over processed records in [i, j) (slice, step=1)
          - ra[(f, j)]       : processed record at file f, local line j
                                (f can be int file index or filename str)
        """
        # Tuple addressing: (file_key, local_line)
        if isinstance(key, tuple) and len(key) == 2:
            file_key, local_line = key
            return self.get_at(file_key, local_line)

        # Global int
        if isinstance(key, int):
            if self._skip <= 1:
                raw_idx = key
                if raw_idx < 0:
                    raw_idx = self.total_raw_lines + raw_idx
                if raw_idx < 0 or raw_idx >= self.total_raw_lines:
                    raise IndexError("index out of range")
                return self.get(raw_idx)

            logical_idx = key
            if logical_idx < 0:
                self._ensure_all_skip_cached()
                logical_idx = len(self._skip_records) + logical_idx
            if logical_idx < 0:
                raise IndexError("index out of range")
            if not self._ensure_skip_cache(logical_idx):
                raise IndexError("index out of range")
            return self._skip_records[logical_idx]

        # Global slice
        if isinstance(key, slice):
            if self._skip <= 1:
                start, stop, step = key.indices(self.total_raw_lines)
                if step != 1:
                    raise ValueError("slice steps other than 1 are not supported")
                return self.iter_slice(start, stop)

            if key.step not in (None, 1):
                raise ValueError("slice steps other than 1 are not supported")
            start = key.start if key.start is not None else 0
            stop = key.stop
            if start < 0 or (stop is not None and stop < 0):
                self._ensure_all_skip_cached()
                length = len(self._skip_records)
                if start < 0:
                    start = length + start
                if stop is not None and stop < 0:
                    stop = length + stop
            if start < 0:
                start = 0
            if stop is not None and stop < start:
                return iter(())
            return self._iter_skip_logical_range(start, stop)

        raise TypeError(
            "indices must be int, slice, or (file_key, local_line) tuple"
        )

    def __len__(self) -> int:
        """
        Number of processed records produced by this view.
        """
        if self._skip > 1:
            self._ensure_all_skip_cached()
            return len(self._skip_records)

        if self._full_len_cache is not None:
            return self._full_len_cache

        count = 0
        for _ in self._iter_processed_range(0, None, apply_skip=False):
            count += 1
        self._full_len_cache = count
        return count

    # ---------- context manager convenience ----------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# -------------------- Example usage --------------------

# Define your per-line processing:
def my_process(rec: dict) -> Optional[dict]:
    # Keep as-is or filter/transform
    # Example: ensure 'idx' exists and normalize text
    if "idx" not in rec:
        return None
    out = dict(rec)
    if "text" in out and isinstance(out["text"], str):
        out["text"] = out["text"].strip()
    return out

def decode_jsonl_line(rec: dict, jumble_map: dict=None, decode_rle=False) -> Optional[dict]:
    if "masks" in rec and isinstance(rec["masks"], dict):
        out = dict(rec)
        out["masks"] = {k: v if not decode_rle else decode(v) for k, v in rec["masks"].items()}
        if jumble_map is not None:
            tmp = {k: out["masks"][jumble_map.get(k, k)] for k in jumble_map if jumble_map.get(k, k) in out["masks"]}
            out["masks"] = tmp
        return out
    return rec


# Build over a directory of .jsonl files:
# ra = JsonlDirRA("/path/to/jsonl_dir", process=my_process, index_every=1)

# Iterate everything lazily:
# for r in ra:
#     ...

# Global access:
# first = ra[0]
# last  = ra[-1]
# for r in ra[100:200]:
#     ...

# Per-file access (by index or filename):
# r1 = ra[(0, 42)]                        # file 0, line 42
# r2 = ra[("part-0003.jsonl", 17)]        # file named 'part-0003.jsonl', line 17

# Clean up (also automatic if used in `with`):
# ra.close()
