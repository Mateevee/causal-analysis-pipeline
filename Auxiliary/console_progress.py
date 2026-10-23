"""
console_progress.py — FIXED progress footer pinned to the bottom of the
terminal ("scenario X of N", "model Y of M"), used by the runner
(main.py::run_all) so that progress stays always visible while the
pipeline's normal log (thousands of prints scattered across Pipeline/*,
ModelComparison/*, etc.) keeps scrolling above it, without needing to
change any of those prints.

Only works on terminals with ANSI/VT100 support and with stdout attached
to a real TTY. If stdout is redirected (file, pipe, `> log.txt`) or the
terminal is too short, start()/update()/stop() become silent no-ops and
the runner just prints its normal log.
"""

import atexit
import os
import shutil
import sys

_CSI = "\x1b["


def _enable_windows_vt() -> None:
    """On Windows, enables ANSI/VT sequence processing in the console
    (needed on legacy conhost.exe; Windows Terminal and PowerShell 7+
    already have native support). No effect on other OSes; never raises
    an exception — worst case, the console has no fixed footer, it does
    not break the runner."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass


class PinnedProgress:
    """
    Fixed `n_lines`-line footer at the bottom of the terminal.

    Usage:
        progress = PinnedProgress(n_lines=2)
        progress.start()
        ...
        progress.update(["SCENARIO 3/14 (name)", "MODEL 2/5 (patch)"])
        ...  # normal prints from the rest of the pipeline keep scrolling above
        ...
        progress.stop()  # restores the terminal (also runs via atexit,
                          # as a safety net if the process dies before
                          # calling stop() explicitly)
    """

    def __init__(self, n_lines: int = 2):
        self.n_lines = n_lines
        self._active = False
        self._enabled = sys.stdout.isatty()
        self._rows = None

    def start(self) -> None:
        if not self._enabled:
            return
        _enable_windows_vt()
        try:
            _cols, rows = shutil.get_terminal_size(fallback=(120, 40))
            if rows <= self.n_lines + 5:
                self._enabled = False  # terminal too short — not worth it
                return
            self._rows = rows
            scroll_bottom = rows - self.n_lines
            # pushes what is already on screen upward (does not erase
            # scrollback), freeing the last n_lines lines for the footer,
            # and only then reserves the remainder (top) as the scroll region
            sys.stdout.write("\n" * self.n_lines)
            sys.stdout.write(f"{_CSI}1;{scroll_bottom}r")
            sys.stdout.write(f"{_CSI}{scroll_bottom};1H")
            sys.stdout.flush()
            self._active = True
            atexit.register(self.stop)
        except Exception:
            self._enabled = False

    def update(self, lines) -> None:
        """lines: str or a list of up to n_lines strings. Redraws the
        fixed footer without moving the cursor away from where the normal
        log is writing (position save/restore around the redraw)."""
        if not (self._enabled and self._active):
            return
        if isinstance(lines, str):
            lines = [lines]
        try:
            _cols, rows = shutil.get_terminal_size(fallback=(120, 40))
            base_row = rows - self.n_lines
            sys.stdout.write("\x1b7")  # save cursor position+attributes
            for i in range(self.n_lines):
                text = lines[i] if i < len(lines) else ""
                text = text[:max(_cols - 1, 0)]  # avoid wrapping the fixed line
                sys.stdout.write(f"{_CSI}{base_row + i + 1};1H{_CSI}K{text}")
            sys.stdout.write("\x1b8")  # restore cursor position
            sys.stdout.flush()
        except Exception:
            pass

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        try:
            rows = self._rows or shutil.get_terminal_size(fallback=(120, 40)).lines
            sys.stdout.write(f"{_CSI}r")           # remove the scroll region
            sys.stdout.write(f"{_CSI}{rows};1H\n")  # cursor at the end of the screen
            sys.stdout.flush()
        except Exception:
            pass
