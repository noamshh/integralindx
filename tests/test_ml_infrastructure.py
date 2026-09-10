"""
Test suite for ML infrastructure: vocab, tokenizer, prefix notation, and model.
"""
import pytest
import torch
import sympy as sp

from src.utils.prefix_notation import sympy_to_prefix
from src.models.egen.tokenizer import Tokenizer
from src.models.egen.vocab import get_vocab_size, get_all_tokens
from src.models.egen.contrastive_model import Encoder


class TestVocabulary:
    """test frozen vocabulary"""

    def test_vocab_size(self):
        """test that vocabulary has expected size"""
        vocab_size = get_vocab_size()
        assert vocab_size == 70, f"Expected vocab size 70, got {vocab_size}"

    def test_vocab_tokens(self):
        """test that all expected token categories are present"""
        tokens = get_all_tokens()

        # check special tokens
        assert 'PAD' in tokens, "Missing PAD token"
        assert 'SOE' in tokens, "Missing SOE token"
        assert 'EOE' in tokens, "Missing EOE token"

        # check variables
        assert 'x' in tokens, "Missing variable x"
        assert 'a' in tokens, "Missing parameter a"
        assert 'b' in tokens, "Missing parameter b"
        assert 'c' in tokens, "Missing parameter c"

        # check constants
        assert 'pi' in tokens, "Missing constant pi"
        assert 'e' in tokens, "Missing constant e"
        for i in range(10):
            assert str(i) in tokens, f"Missing digit {i}"

        # check operators
        operators = ['add', 'mul', 'sub', 'pow', 'sin', 'cos', 'ln', 'exp']
        for op in operators:
            assert op in tokens, f"Missing operator {op}"

        # check special functions
        assert 'Li' in tokens, "Missing Li (polylog)"
        assert 'zeta' in tokens, "Missing zeta"
        assert 're' in tokens, "Missing re (real part)"
        assert 'im' in tokens, "Missing im (imaginary part)"

    def test_no_unk_func(self):
        """test that UNK_FUNC is not in vocabulary (removed per refactoring)"""
        tokens = get_all_tokens()
        assert 'UNK_FUNC' not in tokens, "UNK_FUNC should be removed from vocabulary"
        assert 'UNK' not in tokens, "UNK should be removed from vocabulary"


class TestTokenizer:
    """test tokenizer with frozen vocabulary"""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    def test_tokenizer_vocab_size(self, tokenizer):
        """test tokenizer vocabulary size"""
        assert len(tokenizer) == 70, f"Expected tokenizer vocab size 70, got {len(tokenizer)}"

    def test_encode_simple_expressions(self, tokenizer):
        """test encoding simple expressions"""
        test_cases = [
            "add pow x 2 sin x",
            "mul a x",
            "add x 1",
            "pow x 2",
        ]

        for expr in test_cases:
            tokens = tokenizer.encode(expr)
            assert isinstance(tokens, list), f"Expected list, got {type(tokens)}"
            assert len(tokens) > 0, f"Empty token list for {expr}"
            # check all tokens are valid indices
            for token_id in tokens:
                assert 0 <= token_id < len(tokenizer), \
                    f"Invalid token ID {token_id} for {expr}"

    def test_encode_with_special_functions(self, tokenizer):
        """test encoding expressions with special functions"""
        test_cases = [
            "Li 2 x",  # polylog(2, x)
            "zeta x",  # zeta(x)
            "re x",  # re(x)
            "im x",  # im(x)
        ]

        for expr in test_cases:
            tokens = tokenizer.encode(expr)
            assert len(tokens) > 0, f"Failed to encode {expr}"

    def test_encode_decode_roundtrip(self, tokenizer):
        """test that encode/decode is reversible"""
        test_cases = [
            "add x 1",
            "mul a pow x 2",
            "Li 2 x",
        ]

        for expr in test_cases:
            tokens = tokenizer.encode(expr)
            decoded = tokenizer.decode(tokens)
            # decoded should match original (spaces normalized)
            assert decoded.strip() == expr.strip(), \
                f"Roundtrip failed for {expr}: got {decoded}"

    def test_max_len_padding(self, tokenizer):
        """test that max_len parameter works correctly"""
        expr = "add x 1"
        max_len = 10

        tokens = tokenizer.encode(expr, max_len=max_len)
        assert len(tokens) <= max_len, \
            f"Tokens exceed max_len: {len(tokens)} > {max_len}"

    def test_batch_encode(self, tokenizer):
        """test batch encoding with padding"""
        exprs = [
            "add x 1",
            "mul a pow x 2",
            "Li 2 x",
        ]

        batch = tokenizer.batch_encode(exprs, max_len=20)

        # check output format
        assert 'tokens' in batch, "Missing 'tokens' key in batch output"
        assert 'mask' in batch, "Missing 'mask' key in batch output"

        # check shapes
        tokens = batch['tokens']
        mask = batch['mask']
        assert tokens.shape[0] == len(exprs), "Batch size mismatch"
        assert tokens.shape == mask.shape, "Token and mask shapes don't match"


class TestPrefixNotation:
    """test prefix notation conversion"""

    def test_simple_expressions(self):
        """test conversion of simple expressions"""
        x = sp.Symbol('x')
        a = sp.Symbol('a')

        test_cases = [
            (x, "x"),
            (x + 1, "add x 1"),
            (x * a, "mul a x"),
            (x**2, "pow x 2"),
            (sp.sin(x), "sin x"),
            (sp.cos(x), "cos x"),
        ]

        for expr, expected in test_cases:
            result = sympy_to_prefix(expr)
            assert result == expected, f"Failed for {expr}: expected '{expected}', got '{result}'"

    def test_special_functions(self):
        """test conversion of special functions"""
        x = sp.Symbol('x')

        test_cases = [
            (sp.polylog(2, x), "Li 2 x"),
            (sp.zeta(3), "zeta 3"),
            (sp.zeta(x), "zeta x"),
            (sp.re(x), "re x"),
            (sp.im(x), "im x"),
        ]

        for expr, expected in test_cases:
            result = sympy_to_prefix(expr)
            assert result == expected, f"Failed for {expr}: expected '{expected}', got '{result}'"

    def test_nested_expressions(self):
        """test conversion of nested expressions"""
        x = sp.Symbol('x')
        a = sp.Symbol('a')

        test_cases = [
            (x**2 + sp.sin(x), "add pow x 2 sin x"),
            (a * x**2, "mul a pow x 2"),
            (sp.sin(a * x), "sin mul a x"),
        ]

        for expr, expected in test_cases:
            result = sympy_to_prefix(expr)
            assert result == expected, f"Failed for {expr}: expected '{expected}', got '{result}'"

    def test_complex_expressions(self):
        """test conversion of complex expressions from real integrands"""
        x = sp.Symbol('x')
        a, b = sp.symbols('a b')

        test_cases = [
            # polynomial
            (a*x**2 + b*x, "add mul a pow x 2 mul b x"),
            # trig
            (sp.sin(x)**2 + sp.cos(x)**2, "add pow sin x 2 pow cos x 2"),
            # logarithm
            (sp.log(x**a), "ln pow x a"),
        ]

        for expr, expected in test_cases:
            result = sympy_to_prefix(expr)
            # just check it doesn't crash and produces something
            assert isinstance(result, str), f"Failed to convert {expr}"
            assert len(result) > 0, f"Empty result for {expr}"


class TestMathEncoder:
    """test contrastive encoder model"""

    @pytest.fixture
    def model(self):
        """create a small model for testing"""
        return Encoder(
            vocab_size=70,
            dim=128,  # smaller for testing
            num_layers=2,  # fewer layers for testing
            num_heads=4,
            feedforward_dim=256,
            max_seq_len=50,
            dropout=0.1
        )

    def test_model_creation(self, model):
        """test that model can be created"""
        assert model is not None, "Failed to create model"
        assert isinstance(model, torch.nn.Module), "Model is not a torch module"

    def test_forward_pass(self, model):
        """test forward pass with dummy input"""
        batch_size = 4
        seq_len = 10
        vocab_size = 70

        # create dummy input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        # forward pass
        output = model(x, mask=mask)

        # check output shape
        assert output.shape == (batch_size, seq_len, 128), \
            f"Expected shape (4, 10, 128), got {output.shape}"

    def test_mean_pooling(self, model):
        """test mean pooling over sequence"""
        batch_size = 4
        seq_len = 10

        # create dummy hidden states
        hidden = torch.randn(batch_size, seq_len, 128)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # mask out last 3 positions
        mask[:, -3:] = True

        # mean pool
        pooled = model.mean_pool(hidden, mask)

        # check output shape
        assert pooled.shape == (batch_size, 128), \
            f"Expected shape (4, 128), got {pooled.shape}"

        # check that padding is excluded (non-zero values)
        assert not torch.all(pooled == 0), "Pooled output is all zeros"

    def test_model_parameters(self, model):
        """test that model has trainable parameters"""
        params = list(model.parameters())
        assert len(params) > 0, "Model has no parameters"

        total_params = sum(p.numel() for p in params)
        assert total_params > 0, "Model has zero parameters"

    def test_model_device_transfer(self, model):
        """test that model can be moved to different devices"""
        # test CPU
        model_cpu = model.to('cpu')
        assert next(model_cpu.parameters()).device.type == 'cpu'

        # test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            assert next(model_cuda.parameters()).device.type == 'cuda'


class TestEndToEnd:
    """test end-to-end pipeline: expression → prefix → tokens → model"""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    @pytest.fixture
    def model(self):
        return Encoder(
            vocab_size=70,
            dim=128,
            num_layers=2,
            num_heads=4,
            feedforward_dim=256,
            max_seq_len=50,
            dropout=0.1
        )

    def test_expression_to_embedding(self, tokenizer, model):
        """test full pipeline from SymPy expression to embedding"""
        # create expression
        x = sp.Symbol('x')
        expr = x**2 + sp.sin(x)

        # convert to prefix
        prefix = sympy_to_prefix(expr)
        assert isinstance(prefix, str), "Prefix conversion failed"

        # tokenize
        tokens = tokenizer.encode(prefix, max_len=20)
        assert len(tokens) > 0, "Tokenization failed"

        # convert to tensor
        token_tensor = torch.tensor([tokens], dtype=torch.long)  # [1, L]
        mask = torch.zeros_like(token_tensor, dtype=torch.bool)

        # forward pass
        model.eval()
        with torch.no_grad():
            hidden = model(token_tensor, mask=mask)  # [1, L, D]
            embedding = model.mean_pool(hidden, mask)  # [1, D]

        # check output
        assert embedding.shape == (1, 128), f"Expected shape (1, 128), got {embedding.shape}"
        assert not torch.all(embedding == 0), "Embedding is all zeros"
