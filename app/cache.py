import hashlib
from collections import OrderedDict

MAX_SIZE = 256


class ResponseCache:
    """Simple in-memory LRU cache keyed by question hash."""

    def __init__(self, max_size: int = MAX_SIZE):
        self._store: OrderedDict[str, str] = OrderedDict()
        self._max = max_size

    @staticmethod
    def _key(question: str) -> str:
        return hashlib.sha256(question.strip().lower().encode()).hexdigest()

    def get(self, question: str) -> str | None:
        key = self._key(question)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(self, question: str, answer: str) -> None:
        key = self._key(question)
        self._store[key] = answer
        self._store.move_to_end(key)
        if len(self._store) > self._max:
            self._store.popitem(last=False)


response_cache = ResponseCache()
