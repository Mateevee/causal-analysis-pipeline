# =============================================================================
# WATCH PAIRS — Tracking pairs of interest throughout the pipeline
# =============================================================================

class WatchPairs:
    """
    Registers and tracks pairs (source_entity, source_property, target_entity,
    target_property) of interest throughout the pipeline's entire filter
    cascade.

    Accepted input format (two equivalent formats):

      1. List of tuples:
            [
                ('Entity_B', 'X_2', 'Entity_7', 'Y_3'),
                ('Entity_A', 'load', 'Entity_3', 'speed'),
            ]

      2. List of strings in the format "source_entity  source_property  →  target_entity  target_property"
            [
                'Entity_B       X_2             →  Entity_7          Y_3',
                'Entity_A       load            →  Entity_3          speed',
            ]
         The "→" (or "->") separator is required. Extra spaces are ignored.

    Usage:
        wp = WatchPairs([
            ('Entity_B', 'X_2', 'Entity_7', 'Y_3'),
            'Entity_A  load  →  Entity_3  speed',
        ])
        results = run_pipeline(df, watch_pairs=wp)
    """

    # Accepted separators between source and target in the string notation
    _SEPARATORS = ('→', '->')

    def __init__(self, pairs):
        """
        Parameters
        ----------
        pairs : list[tuple | str]
            List of pairs in tuple or string format (see class docstring).
        """
        self._pairs: list[tuple[str, str, str, str]] = []
        for item in (pairs or []):
            self._pairs.append(self._parse(item))

        # Event history per pair: {(so,sp,to,tp): list[dict]}
        self._log: dict[tuple, list] = {p: [] for p in self._pairs}

    # ── Parsing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(item) -> tuple:
        """Converts a string or tuple into (source_entity, source_property, target_entity, target_property)."""
        if isinstance(item, (tuple, list)):
            if len(item) != 4:
                raise ValueError(
                    f"Entity pair tuple must have 4 elements "
                    f"(source_entity, source_property, target_entity, target_property), "
                    f"got: {item}"
                )
            return tuple(str(x).strip() for x in item)

        if isinstance(item, str):
            raw = item
            sep_used = None
            for sep in WatchPairs._SEPARATORS:
                if sep in raw:
                    sep_used = sep
                    break
            if sep_used is None:
                raise ValueError(
                    f"Pair string does not contain a separator ('→' or '->'): {repr(raw)}\n"
                    f"Expected format: 'source_entity  source_property  →  target_entity  target_property'"
                )
            left, right = raw.split(sep_used, 1)
            left_parts  = left.strip().split()
            right_parts = right.strip().split()
            if len(left_parts) < 2 or len(right_parts) < 2:
                raise ValueError(
                    f"Could not extract (entity, property) from both sides "
                    f"of the separator in: {repr(raw)}\n"
                    f"Expected format: 'source_entity  source_property  →  target_entity  target_property'"
                )
            return (left_parts[0], left_parts[1], right_parts[0], right_parts[1])

        raise TypeError(
            f"watch_pairs element must be str or tuple, got {type(item)}: {item!r}"
        )

    # ── Lookup ───────────────────────────────────────────────────────────────

    def _key(self, source_entity, source_property, target_entity, target_property):
        return (str(source_entity).strip(), str(source_property).strip(),
                str(target_entity).strip(), str(target_property).strip())

    def is_watched(self, source_entity, source_property, target_entity, target_property) -> bool:
        return self._key(source_entity, source_property, target_entity, target_property) in self._log

    def is_watched_key(self, src_key: str, tgt_key: str) -> bool:
        """Checks using keys in the 'entity__property' format."""
        try:
            so, sp = src_key.split('__', 1)
            to, tp = tgt_key.split('__', 1)
            return self.is_watched(so, sp, to, tp)
        except Exception:
            return False

    def get_key_from_keys(self, src_key: str, tgt_key: str):
        """Returns the (source_entity, source_property, target_entity, target_property) tuple if the pair is watched, otherwise None."""
        try:
            so, sp = src_key.split('__', 1)
            to, tp = tgt_key.split('__', 1)
            k = self._key(so, sp, to, tp)
            return k if k in self._log else None
        except Exception:
            return None

    # ── Event logging ───────────────────────────────────────────────────────

    def record(self, source_entity, source_property, target_entity, target_property,
               stage: str, status: str, reason: str = ''):
        """
        Records an event for the pair.

        Parameters
        ----------
        stage  : stage name, e.g.: 'xcorr', 'TE', 'Granger_rescue', 'Pearson_rescue'
        status : 'passed', 'removed', 'rescued', 'not_rescued'
        reason : detailed explanation of the reason
        """
        k = self._key(source_entity, source_property, target_entity, target_property)
        if k not in self._log:
            return
        self._log[k].append({'stage': stage, 'status': status, 'reason': reason})

    def record_by_keys(self, src_key: str, tgt_key: str,
                       stage: str, status: str, reason: str = ''):
        """Variant that accepts 'entity__property' keys."""
        k = self.get_key_from_keys(src_key, tgt_key)
        if k is None:
            return
        self._log[k].append({'stage': stage, 'status': status, 'reason': reason})

    # ── Report printing ───────────────────────────────────────────────────────

    def _fmt_pair(self, k):
        so, sp, to, tp = k
        return f"{so}({sp})  →  {to}({tp})"

    def print_report(self):
        """Prints a detailed report tracking all pairs."""
        SEP  = "=" * 70
        SEP2 = "-" * 70
        print(f"\n{SEP}")
        print("  WATCH PAIRS TRACKING REPORT")
        print(SEP)

        for k, events in self._log.items():
            print(f"\n  Watched pair: {self._fmt_pair(k)}")
            if not events:
                print("    (no event recorded — the pair may not have been evaluated)")
                continue
            for ev in events:
                icon = {
                    'passed':      '✔',
                    'removed':     '✘',
                    'rescued':     '↑',
                    'not_rescued': '✘',
                }.get(ev['status'], '?')
                status_str = ev['status'].upper().replace('_', ' ')
                print(f"    [{ev['stage']}] {icon} {status_str}", end='')
                if ev['reason']:
                    print(f": {ev['reason']}")
                else:
                    print()
            print(f"  {SEP2}")

        print()

    # ── Iteration over pairs ────────────────────────────────────────────────

    def __iter__(self):
        return iter(self._pairs)

    def __len__(self):
        return len(self._pairs)

    def __repr__(self):
        return f"WatchPairs({self._pairs})"
