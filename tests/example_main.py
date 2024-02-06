from src import bench
bench.bench(num_samples=1000, num_features=100, fix_comp_time=1, reg_or_cls="reg")

bench.bench(num_samples=1000, num_features=100, fix_comp_time=1, reg_or_cls="cls")
