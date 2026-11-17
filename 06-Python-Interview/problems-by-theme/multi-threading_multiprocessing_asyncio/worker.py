def calculate(n):
    print(f"Starting {n}")
    total = 0
    for i in range(50_000_000):
        total += i
    print(f"Finished {n}")
    return total
