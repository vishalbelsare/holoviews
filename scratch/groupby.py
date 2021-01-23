from holoviews import Path

p = Path(
    [{"a": 0, "b": 1, "c": [0]}, {"a": 0, "b": 2, "c": [0]}, {"a": 1, "b": 3, "c": [0]}, {"a": 1, "b": 4, "c": [0]}],
    kdims=["a", "b"], vdims=["c"]
)

grouped = p.interface.groupby(p, ["a"])
print(grouped)

grouped = p.groupby(["a", "b"])
print(grouped)
