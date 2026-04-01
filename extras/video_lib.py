"""
extras/video_lib.py — Static video library for Ecko.

Loads videos from a root videos/ folder.
  General videos:          videos/*.{mp4,webm,mkv,mov,...}
  Per-character videos:    videos/<char_name>/*.{mp4,...}

Sidecar tag files (manually created, same format as image tags):
  video_name.mp4  ->  video_name.txt
  Format: "makima, office, rain, night"

Videos are served via URL (/videos/<path>) — never base64 encoded.
Result dicts contain 'url' (the serve path) instead of 'uri'.
"""

import os
import random
import re


_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}

_STOP_WORDS = frozenset(
    "a an the and or but is are was were be been being have has had do does did "
    "will would could should may might shall i me my we our you your he she it "
    "they them their this that these those what which who how when where why "
    "to of in on at by for from with as into out up about if so then than "
    "just like also very much more some any all no not solo".split()
)

_TAG_EXACT_SCORE    = 2.0
_TAG_PARTIAL_SCORE  = 1.0
_STEM_EXACT_SCORE   = 1.0
_STEM_PARTIAL_SCORE = 0.5


# ── Module-level helpers ──────────────────────────────────────────────────────

def _load_dir(path: str) -> list[str]:
    if not path or not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.splitext(f.lower())[1] in _VIDEO_EXTS
    )


def _stem_words(filename: str) -> frozenset[str]:
    stem = os.path.splitext(os.path.basename(filename).lower())[0]
    return frozenset(w for w in re.split(r"[\s_\-\d]+", stem) if len(w) > 1)


def _parse_tag_file(txt_path: str) -> frozenset[str]:
    """
    Parse a sidecar .txt tag file. Same format as image tags.
    Parenthetical source qualifiers are stripped.
    """
    if not os.path.exists(txt_path):
        return frozenset()
    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read().strip()
    except Exception:
        return frozenset()

    tokens: set[str] = set()
    for phrase in raw.split(","):
        phrase = phrase.strip().lower()
        if not phrase:
            continue
        base = re.sub(r"\\?\([^)]*\\?\)", "", phrase).strip()
        if not base:
            continue
        for word in re.split(r"[\s_\-]+", base):
            word = re.sub(r"[^\w]", "", word)
            if len(word) > 1 and word not in _STOP_WORDS:
                tokens.add(word)
    return frozenset(tokens)


def _score_video(path: str, tags: frozenset[str], keywords: list[str]) -> float:
    stems = _stem_words(path)
    score = 0.0
    for kw in keywords:
        if kw in tags:
            score += _TAG_EXACT_SCORE
        elif any((t.startswith(kw) or kw.startswith(t)) for t in tags if len(t) > 2 and len(kw) > 2):
            score += _TAG_PARTIAL_SCORE
        if kw in stems:
            score += _STEM_EXACT_SCORE
        elif any((w.startswith(kw) or kw.startswith(w)) for w in stems if len(w) > 3 and len(kw) > 3):
            score += _STEM_PARTIAL_SCORE
    return score


# ── VideoLibrary class ────────────────────────────────────────────────────────

class VideoLibrary:
    """
    Two-tier video pool (general + per-character) with sidecar tag support.
    Videos are served by URL, never base64 encoded.
    """

    def __init__(self) -> None:
        self._root: str = ""
        self._general: list[str] = []
        self._char_cache: dict[str, list[str]] = {}
        self._tag_index: dict[str, frozenset[str]] = {}

    # ── Tag loading ───────────────────────────────────────────────────────────

    def _load_tags(self, video_path: str) -> frozenset[str]:
        stem      = os.path.splitext(video_path)[0]
        directory = os.path.dirname(video_path)
        basename  = os.path.basename(stem)
        candidates = [
            stem + ".txt",
            os.path.join(directory, "\\" + basename + ".txt"),
        ]
        for txt in candidates:
            tags = _parse_tag_file(txt)
            if tags:
                self._tag_index[video_path] = tags
                return tags
        self._tag_index[video_path] = frozenset()
        return frozenset()

    def _index_dir(self, paths: list[str]) -> None:
        tagged = sum(1 for p in paths if self._load_tags(p))
        if paths:
            print(f"[VIDEO_LIB]   {tagged}/{len(paths)} videos have tag files")

    def _tags(self, path: str) -> frozenset[str]:
        if path not in self._tag_index:
            return self._load_tags(path)
        return self._tag_index[path]

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(self, root_dir: str) -> int:
        self._root      = root_dir
        self._tag_index = {}
        self._general   = _load_dir(root_dir)
        self._char_cache = {}

        if self._general:
            print(f"[VIDEO_LIB] Indexing {len(self._general)} general videos...")
        self._index_dir(self._general)

        if os.path.isdir(root_dir):
            for entry in os.listdir(root_dir):
                sub = os.path.join(root_dir, entry)
                if os.path.isdir(sub):
                    vids = _load_dir(sub)
                    if vids:
                        key = entry.lower()
                        self._char_cache[key] = vids
                        print(f"[VIDEO_LIB] Indexing {len(vids)} videos for '{key}'...")
                        self._index_dir(vids)

        total = self.count
        char_summary = ", ".join(f"{len(v)} for '{k}'" for k, v in self._char_cache.items())
        msg = f"{len(self._general)} general"
        if char_summary:
            msg += f" + {char_summary}"
        print(f"[VIDEO_LIB] Ready — {msg} ({self.tag_count()} with tags)")
        return total

    def reload(self, char_name: str = "") -> int:
        if char_name:
            key = char_name.lower()
            for p in self._char_cache.get(key, []):
                self._tag_index.pop(p, None)
            vids = _load_dir(os.path.join(self._root, key))
            self._char_cache[key] = vids
            self._index_dir(vids)
            return len(vids)
        return self.load(self._root)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pool(self, char_name: str = "") -> list[str]:
        key = (char_name or "").lower()
        return self._char_cache.get(key, []) + self._general

    @property
    def count(self) -> int:
        return len(self._general) + sum(len(v) for v in self._char_cache.values())

    def char_count(self, char_name: str) -> int:
        return len(self._char_cache.get(char_name.lower(), []))

    def tag_count(self) -> int:
        return sum(1 for t in self._tag_index.values() if t)

    def _build_result(self, path: str, keywords: list[str], is_random: bool) -> dict:
        """Build result dict. Uses URL (rel_path) not base64."""
        tags  = self._tags(path)
        stems = _stem_words(path)
        rel   = os.path.relpath(path, self._root).replace(os.sep, "/")

        matched = []
        for kw in keywords:
            if kw in tags or kw in stems:
                matched.append(kw)
            elif any(
                (t.startswith(kw) or kw.startswith(t))
                for t in (tags | stems) if len(t) > 2 and len(kw) > 2
            ):
                matched.append(kw)

        return {
            "url":              f"/videos/{rel}",   # serve URL — no base64
            "filename":         os.path.basename(path),
            "rel_path":         rel,
            "stem_words":       sorted(stems),
            "tags":             sorted(tags),
            "matched_keywords": matched,
            "is_random":        is_random,
        }

    # ── Public pick API ───────────────────────────────────────────────────────

    def pick_random(self, char_name: str = "") -> dict | None:
        key  = (char_name or "").lower()
        pool = self._char_cache.get(key) or self._general
        if not pool:
            return None
        return self._build_result(random.choice(pool), keywords=[], is_random=True)

    def pick_by_keywords(self, keywords: list[str], char_name: str = "") -> dict | None:
        pool = self._pool(char_name)
        if not pool:
            return None
        scored = [(p, _score_video(p, self._tags(p), keywords)) for p in pool]
        best   = max(s for _, s in scored)
        if best == 0:
            result = self.pick_random(char_name)
            if result:
                result["is_random"] = True
            return result
        candidates = [p for p, s in scored if s == best]
        return self._build_result(random.choice(candidates), keywords=keywords, is_random=False)

    def pick_for_reply(self, reply_text: str, char_name: str = "") -> dict | None:
        words    = re.findall(r"[a-z]+", reply_text.lower())
        keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        if not keywords:
            return self.pick_random(char_name)
        return self.pick_by_keywords(keywords, char_name)

    def list_char_videos(self, char_name: str) -> list[str]:
        return [os.path.basename(p) for p in self._char_cache.get(char_name.lower(), [])]
