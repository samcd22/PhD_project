import bisect
import numpy as np
import pandas as pd
from data_processing.box_gridder import BoxGridder


def reference_gridding(data, grid_size, log_in, log_out):
    """The original point-by-point algorithm, kept here as the oracle."""
    def ffloor(x, base):
        return base * np.floor(x / base)
    cols = ['x', 'y', 'z']
    grid = []
    for d, c in enumerate(cols):
        lo = ffloor(data[c].min(), grid_size[d]); hi = ffloor(data[c].max(), grid_size[d]) + grid_size[d]
        grid.append(np.arange(lo, hi + grid_size[d], grid_size[d]))
    sums, counts = {}, {}
    for _, row in data.iterrows():
        key = []
        for d, c in enumerate(cols):
            idx = bisect.bisect(grid[d], row[c])
            key.append((grid[d][idx - 1] + grid[d][idx]) / 2)
        key = tuple(key)
        v = 10 ** row['C'] if log_in else row['C']
        sums[key] = sums.get(key, 0.0) + v; counts[key] = counts.get(key, 0) + 1
    rows = [(k[0], k[1], k[2], sums[k] / counts[k], counts[k]) for k in sums]
    ref = pd.DataFrame(rows, columns=['x', 'y', 'z', 'C', 'counts']).sort_values(cols).reset_index(drop=True)
    if log_out:
        ref['C'] = np.log10(ref['C'])
    return ref


def test_vectorised_gridding_matches_reference(workdir):
    rng = np.random.default_rng(7)
    n = 600
    data = pd.DataFrame({'x': rng.uniform(150, 1300, n), 'y': rng.uniform(-900, 700, n),
                         'z': rng.uniform(0, 220, n), 'C': rng.normal(9.3, 0.5, n)})
    data.loc[0, ['x', 'y', 'z']] = [300.0, -100.0, 25.0]  # points exactly on box boundaries go to the upper box
    bg = BoxGridder(data, grid_size=[100, 100, 25], output_path=str(workdir / 'grid'),
                    input_data_logarithmised=True, output_data_logarithmised=True)
    got = bg.processed_data.sort_values(['x', 'y', 'z']).reset_index(drop=True)
    ref = reference_gridding(data, [100, 100, 25], True, True)
    assert len(got) == len(ref)
    pd.testing.assert_frame_equal(got[['x', 'y', 'z', 'counts']].astype(float), ref[['x', 'y', 'z', 'counts']].astype(float))
    assert np.allclose(got['C'].to_numpy(), ref['C'].to_numpy())
    assert got['counts'].sum() == n
    assert all(len(s) == c for s, c in zip(got['samples'], got['counts']))


def test_gridding_is_fast_on_large_input(workdir):
    import time
    rng = np.random.default_rng(1)
    n = 200_000
    data = pd.DataFrame({'x': rng.uniform(150, 1300, n), 'y': rng.uniform(-900, 700, n),
                         'z': rng.uniform(0, 220, n), 'C': rng.normal(9.3, 0.5, n)})
    t0 = time.time()
    BoxGridder(data, grid_size=[100, 100, 25], output_path=str(workdir / 'grid_big'),
               input_data_logarithmised=True, output_data_logarithmised=True)
    assert time.time() - t0 < 30
