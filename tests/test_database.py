"""
Test suite for database operations.
"""
import pytest
import sqlite3
from pathlib import Path

from src.database.integral_db import IntegralDatabase


def instances_of(db, integrand_hash):
    group = db.get_group_by_hash(integrand_hash)
    return [db.get_integral_instance(i) for i in group.definite_instances + group.indefinite_instances]


class TestDatabaseConnection:
    """test database connection and basic operations"""

    @pytest.fixture
    def db(self):
        """create database connection for testing"""
        try:
            db_path = Path('data/integral.db')
            if not db_path.exists():
                pytest.skip("Database not available for testing")
            return IntegralDatabase(db_path)
        except Exception as e:
            pytest.skip(f"Failed to connect to database: {e}")

    def test_database_connection(self, db):
        """test that database connection works"""
        assert db is not None, "Database connection failed"

    def test_get_group_count(self, db):
        """test getting total number of integrand groups"""
        count = db.count_groups()
        assert count > 0, "No integrand groups found"
        assert isinstance(count, int), f"Expected int, got {type(count)}"

    def test_get_instance_count(self, db):
        """test getting total number of integral instances"""
        count = db.count_instances()
        assert count > 0, "No integral instances found"
        assert isinstance(count, int), f"Expected int, got {type(count)}"


class TestIntegrandGroups:
    """test integrand group queries"""

    @pytest.fixture
    def db(self):
        """create database connection for testing"""
        try:
            db_path = Path('data/integral.db')
            if not db_path.exists():
                pytest.skip("Database not available for testing")
            return IntegralDatabase(db_path)
        except Exception as e:
            pytest.skip(f"Failed to connect to database: {e}")

    def test_get_all_groups(self, db):
        """test retrieving all integrand groups"""
        groups = db.get_all_groups()
        assert len(groups) > 0, "No groups returned"

        # check structure of first group
        group = groups[0]
        assert hasattr(group, 'integrand_hash'), "Group missing integrand_hash"
        assert hasattr(group, 'integrand_canonical'), "Group missing integrand_canonical"

    def test_get_group_by_hash(self, db):
        """test retrieving single group by hash"""
        # get a sample group
        groups = db.get_all_groups(limit=1)
        if len(groups) == 0:
            pytest.skip("No groups available for testing")

        test_group = groups[0]
        test_hash = test_group.integrand_hash

        # retrieve by hash
        group = db.get_group_by_hash(test_hash)
        assert group is not None, f"Failed to retrieve group by hash {test_hash}"
        assert group.integrand_hash == test_hash, "Hash mismatch"

    def test_get_groups_paginated(self, db):
        """test paginated group retrieval"""
        # get first page
        page1 = db.get_all_groups(limit=10, offset=0)
        assert len(page1) > 0, "No groups in first page"
        assert len(page1) <= 10, "First page exceeds limit"

        # get second page
        page2 = db.get_all_groups(limit=10, offset=10)

        # pages should be different (if enough data)
        if len(page2) > 0:
            assert page1[0].integrand_hash != page2[0].integrand_hash, \
                "Pages contain same data"


class TestIntegralInstances:
    """test integral instance queries"""

    @pytest.fixture
    def db(self):
        """create database connection for testing"""
        try:
            db_path = Path('data/integral.db')
            if not db_path.exists():
                pytest.skip("Database not available for testing")
            return IntegralDatabase(db_path)
        except Exception as e:
            pytest.skip(f"Failed to connect to database: {e}")

    def test_get_instances_by_hash(self, db):
        """test retrieving instances for a group"""
        # get a sample group
        groups = db.get_all_groups(limit=1)
        if len(groups) == 0:
            pytest.skip("No groups available for testing")

        test_group = groups[0]
        test_hash = test_group.integrand_hash

        # get instances
        instances = instances_of(db, test_hash)
        assert len(instances) > 0, f"No instances found for hash {test_hash}"

        # check structure
        instance = instances[0]
        assert hasattr(instance, 'id'), "Instance missing id"
        assert hasattr(instance, 'integrand_hash'), "Instance missing integrand_hash"
        assert instance.integrand_hash == test_hash, "Hash mismatch"

    def test_get_instance_by_id(self, db):
        """test retrieving single instance by ID"""
        # get a sample instance
        groups = db.get_all_groups(limit=1)
        if len(groups) == 0:
            pytest.skip("No groups available for testing")

        test_hash = groups[0].integrand_hash
        instances = instances_of(db, test_hash)
        if len(instances) == 0:
            pytest.skip("No instances available for testing")

        test_id = instances[0].id

        # retrieve by ID
        instance = db.get_integral_instance(test_id)
        assert instance is not None, f"Failed to retrieve instance {test_id}"
        assert instance.id == test_id, "ID mismatch"


class TestCurationLog:
    """test curation log operations"""

    @pytest.fixture
    def db(self):
        """create database connection for testing"""
        try:
            db_path = Path('data/integral.db')
            if not db_path.exists():
                pytest.skip("Database not available for testing")
            return IntegralDatabase(db_path)
        except Exception as e:
            pytest.skip(f"Failed to connect to database: {e}")

    def test_curation_log_exists(self, db):
        """test that curation log table exists"""
        assert isinstance(db.get_curation_log(), list)

    def test_excluded_instances_filter(self, db):
        """test that excluded instances are filtered correctly"""
        # get all groups without exclusion
        all_groups = db.get_all_groups(exclude_curated=False)

        # get groups with exclusion
        filtered_groups = db.get_all_groups(exclude_curated=True)

        # filtered should be <= all (some might be excluded)
        assert len(filtered_groups) <= len(all_groups), \
            "Filtered count exceeds total count"


class TestDatabaseStatistics:
    """test database statistics and queries"""

    @pytest.fixture
    def db(self):
        """create database connection for testing"""
        try:
            db_path = Path('data/integral.db')
            if not db_path.exists():
                pytest.skip("Database not available for testing")
            return IntegralDatabase(db_path)
        except Exception as e:
            pytest.skip(f"Failed to connect to database: {e}")

    def test_group_instance_relationship(self, db):
        """test that group counts match instance counts"""
        total_groups = db.count_groups()
        total_instances = db.count_instances()

        # should have more instances than groups (instances grouped by integrand)
        assert total_instances >= total_groups, \
            f"Instance count ({total_instances}) less than group count ({total_groups})"

    def test_definite_vs_indefinite(self, db):
        """test counts of definite vs indefinite integrals"""
        # get sample instances
        groups = db.get_all_groups(limit=10)
        if len(groups) == 0:
            pytest.skip("No groups available for testing")

        definite_count = 0
        indefinite_count = 0

        for group in groups:
            instances = instances_of(db, group.integrand_hash)
            for instance in instances:
                if instance.integral_type == 'definite':
                    definite_count += 1
                else:
                    indefinite_count += 1

        # just check we have some data
        total = definite_count + indefinite_count
        assert total > 0, "No definite/indefinite classification found"

    def test_mse_metadata(self, db):
        """test MSE metadata retrieval"""
        # get sample instance with MSE data
        groups = db.get_all_groups(limit=5)
        if len(groups) == 0:
            pytest.skip("No groups available for testing")

        found_mse_data = False
        for group in groups:
            instances = instances_of(db, group.integrand_hash)
            for instance in instances[:3]:  # check first few
                if hasattr(instance, 'mse_question_id') and instance.mse_question_id:
                    found_mse_data = True
                    assert instance.mse_question_id > 0, "Invalid MSE question ID"
                    break
            if found_mse_data:
                break

        assert found_mse_data, "No MSE metadata found in sample"


class TestDatabaseIntegrity:
    """test database integrity and constraints"""

    @pytest.fixture
    def db(self):
        """create database connection for testing"""
        try:
            db_path = Path('data/integral.db')
            if not db_path.exists():
                pytest.skip("Database not available for testing")
            return IntegralDatabase(db_path)
        except Exception as e:
            pytest.skip(f"Failed to connect to database: {e}")

    def test_unique_hashes(self, db):
        """test that integrand hashes are unique"""
        groups = db.get_all_groups()
        hashes = [g.integrand_hash for g in groups]

        # check for duplicates
        unique_hashes = set(hashes)
        assert len(hashes) == len(unique_hashes), \
            f"Found duplicate hashes: {len(hashes)} total, {len(unique_hashes)} unique"

    def test_instance_group_relationship(self, db):
        """test that all instances belong to valid groups"""
        groups = db.get_all_groups(limit=10)
        if len(groups) == 0:
            pytest.skip("No groups available for testing")

        for group in groups:
            instances = instances_of(db, group.integrand_hash)
            assert len(instances) > 0, \
                f"Group {group.integrand_hash} has no instances"

            # all instances should have matching hash
            for instance in instances:
                assert instance.integrand_hash == group.integrand_hash, \
                    f"Instance hash mismatch: {instance.integrand_hash} != {group.integrand_hash}"

    def test_canonical_forms(self, db):
        """test that canonical forms are valid"""
        groups = db.get_all_groups(limit=20)
        if len(groups) == 0:
            pytest.skip("No groups available for testing")

        for group in groups:
            assert group.integrand_canonical, \
                f"Group {group.integrand_hash} missing canonical form"
            assert isinstance(group.integrand_canonical, str), \
                f"Canonical form is not string for {group.integrand_hash}"
            assert len(group.integrand_canonical) > 0, \
                f"Empty canonical form for {group.integrand_hash}"
