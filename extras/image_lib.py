"""
extras/image_lib.py — Static image library for Ecko.

Loads images from a root images/ folder.
  General images:          images/*.{jpg,png,gif,webp,...}
  Per-character images:    images/<char_name>/*.{jpg,png,...}

Sidecar tag files (WD1.5 tagger output):
  image_name.jpg  ->  image_name.txt  (same directory, same stem)
  Format: "makima (chainsaw man), solo, bangs, long hair"

  Tags are parsed at load time and cached.  Matching a tag token scores
  2x vs a filename stem word, so tagged images rank far above untagged ones
  when the keyword is present in the tags.

Pick logic:
  pick_random(char_name)           - random from char pool, falls back to general
  pick_by_keywords(words, char)    - score filenames + tags against keywords
  pick_for_reply(reply, char)      - tokenise reply, call pick_by_keywords

All pick methods return a result dict with keys:
  uri, filename, rel_path, stem_words, tags, matched_keywords, is_random
"""

import base64
import mimetypes
import os
import random
import re


# ── Constants ─────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif"}

_STOP_WORDS = frozenset(
    "a an the and or but is are was were be been being have has had do does did "
    "will would could should may might shall i me my we our you your he she it "
    "they them their this that these those what which who how when where why "
    "to of in on at by for from with as into out up about if so then than "
    "just like also very much more some any all no not solo".split()
)

# Score weights
_TAG_EXACT_SCORE    = 2.0   # exact token match in sidecar tags
_TAG_PARTIAL_SCORE  = 1.0   # prefix/suffix partial match in tags
_STEM_EXACT_SCORE   = 1.0   # exact token match in filename stem
_STEM_PARTIAL_SCORE = 0.5   # prefix/suffix partial match in stem


# ── Module-level helpers ──────────────────────────────────────────────────────

def _load_dir(path: str) -> list[str]:
    """Return sorted list of image full-paths in a directory (non-recursive)."""
    if not path or not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.splitext(f.lower())[1] in _IMAGE_EXTS
    )


def _to_data_uri(path: str) -> str | None:
    """Read an image file and return a base64 data URI."""
    try:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[IMAGE_LIB] Failed to read {path!r}: {e}")
        return None


def _stem_words(filename: str) -> frozenset[str]:
    """Extract word tokens from a filename stem (splits on _, -, spaces, digits)."""
    stem = os.path.splitext(os.path.basename(filename).lower())[0]
    return frozenset(w for w in re.split(r"[\s_\-\d]+", stem) if len(w) > 1)


def _parse_tag_file(txt_path: str) -> frozenset[str]:
    """
    Parse a WD1.5 sidecar .txt file into a frozenset of normalised tokens.

    Input:  "makima (chainsaw man), solo, long hair, red eyes"
    Output: frozenset({"makima", "chainsaw", "man", "long", "hair", "red", "eyes"})

    Each comma-separated phrase is handled as:
      - Strip parenthetical qualifiers like (source) from the base tag
      - Extract words inside the parentheses too (e.g. "chainsaw man")
      - Tokenise everything, filter stop words and single chars
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
        # Strip parenthetical source/fandom qualifiers entirely.
        # e.g. "makima (chainsaw man)" -> "makima"
        # The paren content describes the SOURCE, not visual content.
        base = re.sub(r"\\?\([^)]*\\?\)", "", phrase).strip()
        if not base:
            continue
        for word in re.split(r"[\s_\-]+", base):
            word = re.sub(r"[^\w]", "", word)
            if len(word) > 1 and word not in _STOP_WORDS:
                tokens.add(word)

    return frozenset(tokens)


def _score_image(path: str, tags: frozenset[str], keywords: list[str]) -> float:
    """
    Score one image against keywords using both sidecar tags and filename stems.
    Tags are authoritative (2x weight); filename stems are a weaker fallback.
    Tag and stem scoring are additive — if a keyword hits both it scores both.
    """
    stems = _stem_words(path)
    score = 0.0

    for kw in keywords:
        # Tags (high weight)
        if kw in tags:
            score += _TAG_EXACT_SCORE
        elif any(
            (t.startswith(kw) or kw.startswith(t))
            for t in tags if len(t) > 2 and len(kw) > 2
        ):
            score += _TAG_PARTIAL_SCORE

        # Filename stems (lower weight, additive)
        if kw in stems:
            score += _STEM_EXACT_SCORE
        elif any(
            (w.startswith(kw) or kw.startswith(w))
            for w in stems if len(w) > 3 and len(kw) > 3
        ):
            score += _STEM_PARTIAL_SCORE

    return score


# ── ImageLibrary class ────────────────────────────────────────────────────────

class ImageLibrary:
    """
    Two-tier image pool (general + per-character), with WD1.5 sidecar tag support.

    Tag index maps each image path -> frozenset of tag tokens, loaded at startup.
    Scoring: tag exact match 2pt, tag partial 1pt, stem exact 1pt, stem partial 0.5pt.
    """

    def __init__(self) -> None:
        self._root: str = ""
        self._general: list[str] = []
        self._char_cache: dict[str, list[str]] = {}
        self._tag_index: dict[str, frozenset[str]] = {}

    # ── Tag loading ───────────────────────────────────────────────────────────

    def _load_tags(self, image_path: str) -> frozenset[str]:
        stem = os.path.splitext(image_path)[0]
        directory = os.path.dirname(image_path)
        basename  = os.path.basename(stem)
        # Try plain name first, then backslash-prefixed (WD tagger output convention)
        candidates = [
            stem + ".txt",
            os.path.join(directory, "\\" + basename + ".txt"),
        ]
        for txt in candidates:
            tags = _parse_tag_file(txt)
            if tags:
                self._tag_index[image_path] = tags
                return tags
        self._tag_index[image_path] = frozenset()
        return frozenset()

    def _index_dir(self, paths: list[str]) -> None:
        tagged = sum(1 for p in paths if self._load_tags(p))
        if paths:
            print(f"[IMAGE_LIB]   {tagged}/{len(paths)} images have tag files")

    def _tags(self, path: str) -> frozenset[str]:
        if path not in self._tag_index:
            return self._load_tags(path)
        return self._tag_index[path]

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(self, root_dir: str) -> int:
        """Load the library, indexing all sidecar .txt tag files. Returns image count."""
        self._root = root_dir
        self._tag_index = {}
        self._general = _load_dir(root_dir)
        self._char_cache = {}

        if self._general:
            print(f"[IMAGE_LIB] Indexing {len(self._general)} general images...")
        self._index_dir(self._general)

        if os.path.isdir(root_dir):
            for entry in os.listdir(root_dir):
                sub = os.path.join(root_dir, entry)
                if os.path.isdir(sub):
                    imgs = _load_dir(sub)
                    if imgs:
                        key = entry.lower()
                        self._char_cache[key] = imgs
                        print(f"[IMAGE_LIB] Indexing {len(imgs)} images for '{key}'...")
                        self._index_dir(imgs)

        total = self.count
        char_summary = ", ".join(
            f"{len(v)} for '{k}'" for k, v in self._char_cache.items()
        )
        msg = f"{len(self._general)} general"
        if char_summary:
            msg += f" + {char_summary}"
        print(f"[IMAGE_LIB] Ready — {msg} ({self.tag_count()} with tags)")
        return total

    def reload(self, char_name: str = "") -> int:
        """Reload a single character's pool, or everything if char_name is empty."""
        if char_name:
            key = char_name.lower()
            # Clear old tag entries for this char's images
            for p in self._char_cache.get(key, []):
                self._tag_index.pop(p, None)
            imgs = _load_dir(os.path.join(self._root, key))
            self._char_cache[key] = imgs
            self._index_dir(imgs)
            return len(imgs)
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

    def _build_result(self, path: str, keywords: list[str], is_random: bool) -> dict | None:
        """Assemble the full result dict for a chosen image."""
        uri = _to_data_uri(path)
        if not uri:
            return None
        tags  = self._tags(path)
        stems = _stem_words(path)
        rel   = os.path.relpath(path, self._root).replace(os.sep, "/")

        # Which keywords actually contributed to the score?
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
            "uri":              uri,
            "filename":         os.path.basename(path),
            "rel_path":         rel,
            "stem_words":       sorted(stems),
            "tags":             sorted(tags),       # all WD tag tokens for this image
            "matched_keywords": matched,
            "is_random":        is_random,
        }

    # ── Public pick API ───────────────────────────────────────────────────────

    def pick_random(self, char_name: str = "") -> dict | None:
        """Random image from char pool (falls back to general)."""
        key  = (char_name or "").lower()
        pool = self._char_cache.get(key) or self._general
        if not pool:
            return None
        return self._build_result(random.choice(pool), keywords=[], is_random=True)

    def pick_by_keywords(self, keywords: list[str], char_name: str = "") -> dict | None:
        """
        Score every image against keywords using tags (2x) + filename stems (1x).
        Returns the highest-scoring image, random on tie.
        Falls back to pick_random if nothing scores above zero.
        """
        pool = self._pool(char_name)
        if not pool:
            return None

        scored = [(p, _score_image(p, self._tags(p), keywords)) for p in pool]
        best   = max(s for _, s in scored)

        if best == 0:
            result = self.pick_random(char_name)
            if result:
                result["is_random"] = True
            return result

        candidates = [p for p, s in scored if s == best]
        return self._build_result(random.choice(candidates), keywords=keywords, is_random=False)

    def pick_for_reply(self, reply_text: str, char_name: str = "") -> dict | None:
        """Tokenise reply_text and call pick_by_keywords."""
        words    = re.findall(r"[a-z]+", reply_text.lower())
        keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        if not keywords:
            return self.pick_random(char_name)
        return self.pick_by_keywords(keywords, char_name)

    def list_char_images(self, char_name: str) -> list[str]:
        return [os.path.basename(p) for p in self._char_cache.get(char_name.lower(), [])]
