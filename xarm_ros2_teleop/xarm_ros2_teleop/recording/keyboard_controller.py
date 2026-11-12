"""Read episode-control keys directly from the launch terminal."""

import os
import select
import termios
import threading
import tty
from collections.abc import Callable

_ESCAPE_SEQUENCE_TIMEOUT_SEC = 0.03


class KeyboardController:
    """Forward terminal keys as scalar recorder commands.

    Args:
        device: Character-device path, normally ``/dev/tty``.
        callback: Function receiving ``staging``, ``start``, ``save``,
            ``discard`` or ``finish``. The callback runs on the keyboard
            thread and must not call ROS APIs directly.
    """

    def __init__(
        self,
        device: str,
        callback: Callable[[str], None],
        *,
        decoder: Callable[[bytes], str | None] | None = None,
    ) -> None:
        """Open ``device`` and prepare a stoppable background reader."""

        self.device = str(device)
        self.callback = callback
        self.decoder = decoder or decode_key
        self.stop_event = threading.Event()
        self.settings_lock = threading.Lock()
        self.stream = open(self.device, "rb", buffering=0)
        self.original_settings = termios.tcgetattr(self.stream.fileno())
        self.thread = threading.Thread(target=self._run, name="episode-keyboard", daemon=True)

    def start(self) -> None:
        """Enter cbreak mode and start reading one terminal sequence at a time."""

        tty.setcbreak(self.stream.fileno())
        self.thread.start()

    def close(self) -> None:
        """Stop the reader and restore the terminal settings exactly once."""

        if self.stream.closed:
            return
        self.stop_event.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=0.3)
        self._restore_terminal()

    def _run(self) -> None:
        """Decode bytes until shutdown; unknown keys are intentionally ignored."""

        file_descriptor = self.stream.fileno()
        try:
            while not self.stop_event.is_set():
                ready, _, _ = select.select([file_descriptor], [], [], 0.1)
                if not ready:
                    continue
                first = os.read(file_descriptor, 1)
                if not first:
                    return
                following = b""
                if first == b"\x1b":
                    following = _read_available(file_descriptor, 2)
                command = self.decoder(first + following)
                if command is not None:
                    self.callback(command)
                if command == "finish":
                    return
        finally:
            self._restore_terminal()

    def _restore_terminal(self) -> None:
        """Restore cbreak changes and close the terminal stream idempotently."""

        with self.settings_lock:
            if self.stream.closed:
                return
            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self.original_settings)
            self.stream.close()


def decode_key(sequence: bytes) -> str | None:
    """Map one terminal byte sequence to a recorder command string.

    Args:
        sequence: Raw terminal bytes with shape ``(1..3,)``.

    Returns:
        ``staging``, ``start``, ``save``, ``discard`` or ``finish``;
        unsupported input returns ``None``.
    """

    return {
        b"g": "staging",
        b"G": "staging",
        b"s": "start",
        b"S": "start",
        b"o": "optimal",
        b"O": "optimal",
        b"f": "failure",
        b"F": "failure",
        b"\x1b[D": "discard",
        b"\x1bOD": "discard",
        b"\x1b[C": "save",
        b"\x1bOC": "save",
        b"\x1b": "finish",
    }.get(sequence)


def _read_available(file_descriptor: int, maximum_bytes: int) -> bytes:
    """Read up to ``maximum_bytes`` belonging to one terminal escape sequence."""

    result = bytearray()
    for _ in range(maximum_bytes):
        ready, _, _ = select.select(
            [file_descriptor], [], [], _ESCAPE_SEQUENCE_TIMEOUT_SEC
        )
        if not ready:
            break
        result.extend(os.read(file_descriptor, 1))
    return bytes(result)
