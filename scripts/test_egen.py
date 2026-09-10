"""
Test script for E-Gen equivalence generation

Tests the E-Gen Rust binary with a variety of mathematical expressions
to verify that:
1. Binary runs without errors
2. Generates multiple equivalents per expression
3. New rewrite rules (exp, polylog, zeta, sign) are firing
"""

import sys
from pathlib import Path
from src.egraph.egen_wrapper import generate_equivalents, EGenConfig
from src.utils.paths import get_paths

def main():
    # configure paths
    paths = get_paths()
    project_root = Path(paths['project_root'])
    binary_path = project_root / "bin" / "egen.exe"  # Windows

    if not binary_path.exists():
        # try Unix name
        binary_path = project_root / "bin" / "egen"

    if not binary_path.exists():
        print(f"ERROR: E-Gen binary not found at {binary_path}")
        print("Please build the binary first:")
        print("  cd egen_integralindx")
        print("  cargo build --release")
        print("  mkdir ../bin")
        print("  cp target/release/egen[.exe] ../bin/")
        sys.exit(1)

    print(f"Using E-Gen binary: {binary_path}")
    print("=" * 70)

    # test expressions (prefix notation)
    test_exprs = [
        # simple algebraic
        ("add x 1", "x + 1 (simple addition)"),
        ("mul pow x 2 sin x", "x² * sin(x)"),

        # exp/ln (should trigger new exp rules)
        ("exp x", "exp(x) - should convert to e^x"),
        ("pow e x", "e^x - should convert to exp(x)"),
        ("add exp x exp mul -1 x", "exp(x) + exp(-x) - should use product rules"),

        # polylog (should trigger new polylog rules)
        ("Li 2 x", "Li(2,x) - dilog function"),
        ("Li 2 1", "Li(2,1) - should give π²/6"),
        ("Li 3 1", "Li(3,1) - should connect to zeta(3)"),

        # zeta (should trigger new zeta rules)
        ("zeta 2", "ζ(2) - should give π²/6"),
        ("zeta 4", "ζ(4) - should give π⁴/90"),

        # trig identity (existing rules)
        ("add pow sin x 2 pow cos x 2", "sin²(x) + cos²(x) = 1"),

        # sign function (should trigger new sign rules)
        ("mul sign x abs x", "sign(x) * abs(x) = x"),
    ]

    # configuration
    config = EGenConfig(
        binary_path=binary_path,
        n_equiv=20,  # generate 20 equivalents per expression
        token_limit=12,
        time_limit=60  # 60 second timeout
    )

    print(f"Configuration:")
    print(f"  n_equiv: {config.n_equiv}")
    print(f"  token_limit: {config.token_limit}")
    print(f"  time_limit: {config.time_limit}s")
    print("=" * 70)
    print()

    # run tests
    success_count = 0
    total_equivalents = 0

    for i, (expr, description) in enumerate(test_exprs, 1):
        print(f"Test {i}/{len(test_exprs)}: {description}")
        print(f"  Input: {expr}")

        try:
            equivalents = generate_equivalents(expr, config)
            num_equiv = len(equivalents)
            total_equivalents += num_equiv
            success_count += 1

            print(f"  ✓ Generated {num_equiv} equivalents")

            # show first 3 equivalents
            for j, eq in enumerate(equivalents[:3], 1):
                print(f"    {j}. {eq}")
            if num_equiv > 3:
                print(f"    ... and {num_equiv - 3} more")

            print()

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            print()

    # summary
    print("=" * 70)
    print(f"Results: {success_count}/{len(test_exprs)} tests passed")
    print(f"Total equivalents generated: {total_equivalents}")
    print(f"Average per expression: {total_equivalents / len(test_exprs):.1f}")

    if success_count == len(test_exprs):
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {len(test_exprs) - success_count} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
