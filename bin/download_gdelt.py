try:
    import gdelt
except ImportError:
    raise ImportError(
        "We require `gdelt` package to get access to GDELT dataset"
    )

gd = gdelt.gdelt()
DATE_RANGE = ["2018 Jan 1", "2018 Jan 31"]
results = gd.Search(DATE_RANGE, coverage=True, output="json")
pass
