from .environment import NOT_WINDOWS


class MsgType:
    # Emojis have to be encoded as bytes to not break the terminal on Windows
    @property
    def DOING(self) -> str:
        return b"\xf0\x9f\xa4\x96".decode() if NOT_WINDOWS else "DOING:"

    @property
    def GOOD(self) -> str:
        return b"\xe2\x9c\x85".decode() if NOT_WINDOWS else "DONE:"

    @property
    def FAIL(self) -> str:
        return b"\xf0\x9f\x9a\xa8".decode() if NOT_WINDOWS else "FAILED:"

    @property
    def WARN(self) -> str:
        return b"\xf0\x9f\x9a\xa7".decode() if NOT_WINDOWS else "WARNING:"

    @property
    def SYNC(self) -> str:
        return b"\xf0\x9f\x9a\x82".decode() if NOT_WINDOWS else "SYNCING:"

    @property
    def PY(self) -> str:
        return b"\xf0\x9f\x90\x8d".decode() if NOT_WINDOWS else ""

    @property
    def CLEAN(self) -> str:
        return b"\xf0\x9f\xa7\xb9".decode() if NOT_WINDOWS else "CLEANING:"

    @property
    def TEST(self) -> str:
        return b"\xf0\x9f\xa7\xaa".decode() if NOT_WINDOWS else "TESTING:"

    @property
    def COMMUNICATE(self) -> str:
        return b"\xf0\x9f\x93\xa3".decode() if NOT_WINDOWS else "COMMUNICATING:"

    @property
    def EXAMINE(self) -> str:
        return b"\xf0\x9f\x94\x8d".decode() if NOT_WINDOWS else "VIEWING:"


msg_type = MsgType()


def echo_header(msg: str):
    print(f"\n--- {msg} ---")
