_base_ = ['./flashocc-r50-car-grid.py']

model = dict(
    wocc=True,
    wdet3d=False,
)
