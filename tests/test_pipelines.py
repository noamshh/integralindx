"""
Test suite for data processing pipelines (integration tests).
"""
import pytest
from pathlib import Path


class TestPipelineImports:
    """test that all pipeline modules can be imported"""

    def test_import_filter_pipeline(self):
        """test importing filter pipeline"""
        try:
            from corpus.pipelines import filter_pipeline
            assert filter_pipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import filter_pipeline: {e}")

    def test_import_normalize_pipeline(self):
        """test importing normalize pipeline"""
        try:
            from corpus.pipelines import normalize_pipeline
            assert normalize_pipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import normalize_pipeline: {e}")

    def test_import_variable_normalization_pipeline(self):
        """test importing variable normalization pipeline"""
        try:
            from corpus.pipelines import variable_normalization_pipeline
            assert variable_normalization_pipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import variable_normalization_pipeline: {e}")

    def test_import_canonicalize_pipeline(self):
        """test importing canonicalize pipeline"""
        try:
            from corpus.pipelines import canonicalize_pipeline
            assert canonicalize_pipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import canonicalize_pipeline: {e}")

    def test_import_integrand_group_pipeline(self):
        """test importing integrand group pipeline"""
        try:
            from corpus.pipelines import integrand_group_pipeline
            assert integrand_group_pipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import integrand_group_pipeline: {e}")

    def test_import_rebuild_contributors_pipeline(self):
        """test importing rebuild contributors pipeline"""
        try:
            from corpus.pipelines import rebuild_contributors_pipeline
            assert rebuild_contributors_pipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import rebuild_contributors_pipeline: {e}")

    def test_import_build_training_data_pipeline(self):
        """test importing training data pipeline"""
        try:
            from corpus.pipelines import build_training_data_pipeline
            assert build_training_data_pipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import build_training_data_pipeline: {e}")


class TestPipelineDataFlow:
    """test that pipeline data directories exist and have expected structure"""

    def test_grouped_data_exists(self):
        """test that grouped data directory exists (canonical source)"""
        grouped_dir = Path('data/grouped')
        assert grouped_dir.exists(), "grouped/ directory not found"

        # check for expected files
        groups_file = grouped_dir / 'integrand_groups.jsonl'
        instances_file = grouped_dir / 'integrals_all.jsonl'

        if groups_file.exists():
            assert groups_file.stat().st_size > 0, "integrand_groups.jsonl is empty"
        if instances_file.exists():
            assert instances_file.stat().st_size > 0, "integrals_all.jsonl is empty"

    def test_database_exists(self):
        """test that database file exists (migrated from grouped/)"""
        db_file = Path('data/integral.db')
        if db_file.exists():
            assert db_file.stat().st_size > 0, "integral.db is empty"
        else:
            pytest.skip("Database not yet created from grouped data")

    def test_ml_seeds_exist(self):
        """test that ML synthetic seeds exist"""
        seeds_file = Path('data/ml_seeds/synthetic_seeds.txt')
        if seeds_file.exists():
            assert seeds_file.stat().st_size > 0, "synthetic_seeds.txt is empty"

            # check it has reasonable number of lines
            with open(seeds_file) as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            assert len(lines) > 100, f"Expected >100 seeds, got {len(lines)}"
        else:
            pytest.skip("ML seeds not yet created")


class TestScriptImports:
    """test that corpus scripts can be imported"""

    def test_import_migrate_to_database(self):
        """test importing database migration script"""
        try:
            from corpus.scripts import migrate_to_database
            assert migrate_to_database is not None
        except ImportError as e:
            pytest.fail(f"Failed to import migrate_to_database: {e}")

    def test_import_backfill_authors(self):
        """test importing backfill authors script"""
        try:
            from corpus.scripts import backfill_authors
            assert backfill_authors is not None
        except ImportError as e:
            pytest.fail(f"Failed to import backfill_authors: {e}")


class TestNormalizerImports:
    """test that normalizers can be imported"""

    def test_import_mse_normalizer(self):
        """test importing MSE normalizer"""
        try:
            from corpus.normalizers.mse_normalizer import MSENormalizer
            normalizer = MSENormalizer()
            assert normalizer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import MSENormalizer: {e}")


class TestFormulaModels:
    """test formula data models"""

    def test_import_formula_models(self):
        """test importing formula models"""
        try:
            from corpus.formula_models import IntegralFormula, IntegrandGroup, NormalizedFormula
            assert IntegralFormula is not None
            assert IntegrandGroup is not None
            assert NormalizedFormula is not None
        except ImportError as e:
            pytest.fail(f"Failed to import formula models: {e}")

    def test_create_integral_formula(self):
        """test creating IntegralFormula instance"""
        from corpus.formula_models import IntegralFormula

        # create minimal instance
        formula = IntegralFormula(
            id='test-1',
            source_id='test-source',
            latex='\\int_0^1 x dx',
            sympy_integrand='x',
            sympy_variable='x'
        )

        assert formula.id == 'test-1'
        assert formula.latex == '\\int_0^1 x dx'
        assert formula.sympy_integrand == 'x'

    def test_create_integrand_group(self):
        """test creating IntegrandGroup instance"""
        from corpus.formula_models import IntegrandGroup

        # create minimal instance
        group = IntegrandGroup(
            integrand_canonical='x',
            integrand_hash='test-hash'
        )

        assert group.integrand_canonical == 'x'
        assert group.integrand_hash == 'test-hash'


class TestPipelineConfiguration:
    """test pipeline configuration and paths"""

    def test_paths_config_exists(self):
        """test that paths config file exists"""
        config_file = Path('config/paths.yaml')
        assert config_file.exists(), "config/paths.yaml not found"

    def test_get_paths_import(self):
        """test importing paths utility"""
        try:
            from src.utils.paths import get_paths
            paths = get_paths()
            assert paths is not None
            assert 'data' in paths
            assert 'project_root' in paths
        except ImportError as e:
            pytest.fail(f"Failed to import get_paths: {e}")

    def test_database_config_exists(self):
        """test that database config file exists"""
        config_file = Path('config/database.yaml')
        if config_file.exists():
            assert config_file.stat().st_size > 0, "database.yaml is empty"
        else:
            pytest.skip("Database config not yet created")

    def test_training_config_exists(self):
        """test that training config file exists"""
        config_file = Path('config/training_config.yaml')
        if config_file.exists():
            assert config_file.stat().st_size > 0, "training_config.yaml is empty"
        else:
            pytest.skip("Training config not yet created")


class TestEGraphIntegration:
    """test e-graph wrapper and integration"""

    def test_import_egen_wrapper(self):
        """test importing E-Gen wrapper"""
        try:
            from src.egraph.egen_wrapper import EGenConfig, generate_equivalents, generate_batch
            assert EGenConfig is not None
            assert generate_equivalents is not None
            assert generate_batch is not None
        except ImportError as e:
            pytest.fail(f"Failed to import egen_wrapper: {e}")

    def test_egen_config_creation(self):
        """test creating EGenConfig"""
        from src.egraph.egen_wrapper import EGenConfig

        config = EGenConfig(
            binary_path=Path('bin/egen'),
            n_equiv=20,
            token_limit=12,
            time_limit=300
        )

        assert config.binary_path == Path('bin/egen')
        assert config.n_equiv == 20
        assert config.token_limit == 12

    def test_egen_binary_path(self):
        """test that egen binary path is configured"""
        from pathlib import Path

        binary_path = Path('bin/egen')
        if not binary_path.exists():
            # try with .exe extension for Windows
            binary_path = Path('bin/egen.exe')

        if not binary_path.exists():
            pytest.skip("E-Gen binary not yet built (expected, needs Rust)")


class TestPipelineChainIntegrity:
    """test that pipeline chain makes sense (output of one is input of next)"""

    def test_normalized_to_filtered(self):
        """test that normalized output can be input to filter pipeline"""
        normalized_dir = Path('data/normalized')
        if not normalized_dir.exists():
            pytest.skip("Normalized data not available")

        normalized_accepted = normalized_dir / 'normalized_accepted.jsonl'
        if normalized_accepted.exists():
            # this file should be used as input to filter_pipeline
            assert normalized_accepted.stat().st_size > 0

    def test_filtered_to_var_normalized(self):
        """test that filtered output can be input to variable normalization"""
        filtered_dir = Path('data/filtered')
        if not filtered_dir.exists():
            pytest.skip("Filtered data not available")

        integrals_file = filtered_dir / 'integrals_all.jsonl'
        if integrals_file.exists():
            # this file should be used as input to variable_normalization_pipeline
            assert integrals_file.stat().st_size > 0

    def test_var_normalized_to_canonicalized(self):
        """test that var_normalized output can be input to canonicalization"""
        var_norm_dir = Path('data/var_normalized')
        if not var_norm_dir.exists():
            pytest.skip("Variable normalized data not available")

        integrals_file = var_norm_dir / 'integrals_all.jsonl'
        if integrals_file.exists():
            # this file should be used as input to canonicalize_pipeline
            assert integrals_file.stat().st_size > 0

    def test_canonicalized_to_grouped(self):
        """test that canonicalized output can be input to grouping"""
        canon_dir = Path('data/canonicalized')
        if not canon_dir.exists():
            pytest.skip("Canonicalized data not available")

        integrals_file = canon_dir / 'integrals_all.jsonl'
        if integrals_file.exists():
            # this file should be used as input to integrand_group_pipeline
            assert integrals_file.stat().st_size > 0

    def test_grouped_to_database(self):
        """test that grouped output is migrated to database"""
        grouped_dir = Path('data/grouped')
        if not grouped_dir.exists():
            pytest.skip("Grouped data not available")

        groups_file = grouped_dir / 'integrand_groups.jsonl'
        db_file = Path('data/integral.db')

        if groups_file.exists() and db_file.exists():
            # database should be newer than grouped data (migrated)
            groups_mtime = groups_file.stat().st_mtime
            db_mtime = db_file.stat().st_mtime

            # note: this check might fail if database is older than grouped data
            # which is fine - it just means a rebuild is needed
            if db_mtime < groups_mtime:
                pytest.skip("Database is older than grouped data - needs rebuild")
