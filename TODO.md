1. sum no-axis path: replace hardcoded (1, 1) with rank-derived all-ones — (1,) \* ndim conceptually.
2. test_axis_reduction line 198: back to (1,) (and 192's (6,).sum() also (1,) — both 1D lines now agree, the seam closes).
3. test_reduce_ops: its (2,2) expectations become (1,1).
4. mean/max/min: same treatment as sum — or, if you're deferring them, make the ignored axis honest with a NotImplementedError rather than a silent wrong answer.
5. One comment line at the top of the shape tests naming the convention, so the next flip-flop temptation hits documentation instead of code.
6. Value assertions for the axis paths (5,7,9 / 6,15 from your (2,3)) — still the only thing standing between max-of-indices-style bugs and a green suite.
