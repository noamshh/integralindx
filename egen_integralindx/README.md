# E-Gen for IntegralIndx

**Forked from**: [hongbozheng/E-Gen](https://github.com/hongbozheng/E-Gen)

Customized E-Gen fork for integral expression equivalence generation with extended mathematical support.

## Modifications

**Language Extensions** (`src/math.rs`):
- Added `Li` (polylogarithm) and `zeta` (Riemann zeta) functions to language definition
- Added rewrite rules including:
  - Polylogarithm identities and special values
  - Extended trigonometric and hyperbolic identities
  - Logarithm and exponential simplification rules
  - Power laws and algebraic simplifications
  - Rational function decomposition

For general E-Gen documentation and theory, see the [original repository](https://github.com/hongbozheng/E-Gen).


## Building

### Prerequisites
Install Rust using [`rustup`](https://www.rust-lang.org/tools/install):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build Release Binary
```bash
cd egen_integralindx
carge clean
cargo build --release
```

Binary will be at: `target/release/egen` (or `egen.exe` on Windows)

### Install to Project
```bash
# From project root
cp egen_integralindx/target/release/egen bin/egen
```

## Usage

```bash
bin/egen -f -n 20 -l 10 -m 10 -t 400 -i input.txt -o output.txt
```
- each line in input.txt contains an s-expression
## Parameters

- `-n` **n_equiv**: Number of equivalent expressions to generate (default: 20)
- `-l` **token_limit**: Initial token limit for saturation (default: 10)
- `-m` **max_token_limit**: Maximum token limit (default: 12)
- `-t` **time_limit**: Time limit in seconds (default: 300)
- `-e` **expression**: Single expression to process
- `-i` **input**: Input file with multiple expressions
- `-o` **output**: Output file for results

## Original E-Gen

**Paper**: "E-Gen: Leveraging E-Graphs to Improve Continuous Representations of Symbolic Expressions" by Zheng et al. (2025)

**Original Repository**: https://github.com/hongbozheng/E-Gen

**Transformer Model**: https://github.com/hongbozheng/transformer

## License

MIT (same as original E-Gen)
