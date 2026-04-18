def get_channels(width_mult=1.0):
    base = [24, 116, 232, 464, 1024]

    return [int(x * width_mult) for x in base]
