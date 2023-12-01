import platform

NOT_WINDOWS = platform.system() != "Windows"


def on_ovartaci() -> bool:
    import platform

    if platform.node() == "RMAPPS1279":
        print(f"\n{msg_type.GOOD} On Ovartaci")
        return True

    print(f"\n{msg_type.GOOD} Not on Ovartaci")
    return False
