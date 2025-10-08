# E-Gen for IntegralIndx

**Forked from**: [hongbozheng/E-Gen](https://github.com/hongbozheng/E-Gen)

This is a customized fork of E-Gen with added rewrite rules for mathematical expression equivalence generation.

## Modifications

### Custom Rewrite Rules (`src/math.rs`)
Added:
- Polylogarithm identities: `Li(1,x) = -ln(1-x)`, `Li(2,1) = π²/6`
- UNK_FUNC handling for unknown functions
- Extended trig/hyperbolic identities: `sin²(x)+cos²(x)=1`, `cosh²(x)-sinh²(x)=1`
- Rational function decomposition
- Logarithm/exponential rules


## Building

### Prerequisites
Install Rust using [`rustup`](https://www.rust-lang.org/tools/install):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build Release Binary
```bash
cd egen_integralindx
cargo build --release
```

Binary will be at: `target/release/egen` (or `egen.exe` on Windows)

### Install to Project
```bash
# From project root
cp egen_integralindx/target/release/egen bin/egen
```

## Usage

### Command Line
```bash
# Single expression
bin/egen -n 20 -l 12 -t 300 "(+ x 1)"

# From file
bin/egen -n 20 -l 12 -t 300 -i input.txt -o output.txt
```

### Python Wrapper
```python
from pathlib import Path
from src.egraph import EGenConfig, generate_equivalents

config = EGenConfig(
    binary_path=Path("bin/egen"),
    n_equiv=20,
    token_limit=12,
    time_limit=300
)

equivalents = generate_equivalents("(+ x x)", config)
print(equivalents)  # ['(* 2 x)', '(* x 2)', ...]
```

## Parameters

- `-n` **n_equiv**: Number of equivalent expressions to generate (default: 20)
- `-l` **token_limit**: Initial token limit for saturation (default: 10)
- `-m` **max_token_limit**: Maximum token limit (default: 12)
- `-t` **time_limit**: Time limit in seconds (default: 300)
- `-e` **expression**: Single expression to process
- `-i` **input**: Input file with multiple expressions
- `-o` **output**: Output file for results

## Original E-Gen

**Paper**: "E-Gen: Leveraging E-Graphs to Improve Continuous Representations of Symbolic Expressions" by Zheng et al. (2024)

**Original Repository**: https://github.com/hongbozheng/E-Gen

**Transformer Model**: https://github.com/hongbozheng/transformer

## License

MIT (same as original E-Gen)
