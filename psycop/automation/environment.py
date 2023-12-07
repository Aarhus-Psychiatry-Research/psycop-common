import platform

NOT_WINDOWS = platform.system() != "Windows"


def on_ovartaci() -> bool:
    import platform

    if platform.node() == "RMAPPS1279":
        print("On Ovartaci")
        return True

    print("Not on Ovartaci")
    return False
