import logging
import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from src.database.schema import Base, IntegrandGroupModel, IntegralInstanceModel, MSEMetadataModel, CurationLogModel
from corpus.formula_models import IntegrandGroup, IntegralFormula

logger = logging.getLogger(__name__)


class IntegralDatabase:
    """Defines database access for integral groups and instances
    Args:
        db_path: path to SQLite database file
        fallback_to_jsonl: if True, fall back to jsonl loading if database doesn't exist"""
    def __init__(self, db_path: Path, fallback_to_jsonl: bool = True):
        self.db_path = db_path
        self.fallback_to_jsonl = fallback_to_jsonl
        self.engine = None
        self.Session = None
        if db_path.exists():
            self._init_database()
        elif not fallback_to_jsonl:
            raise FileNotFoundError(f"database not found: {db_path}")
        else:
            logger.warning(f"database not found: {db_path}, will fall back to jsonl if needed")

    def _init_database(self):
        self.engine = create_engine(
            f'sqlite:///{self.db_path}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool
        )
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"initialized database connection: {self.db_path}")

    def create_tables(self):
        if self.engine is None:
            self.engine = create_engine(
                f'sqlite:///{self.db_path}',
                connect_args={'check_same_thread': False},
                poolclass=StaticPool
            )
            self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info("created database tables")

    def drop_tables(self):
        if self.engine is None:
            return
        Base.metadata.drop_all(self.engine)
        logger.info("dropped database tables")

    def get_group_by_hash(self, integrand_hash: str, exclude_curated: bool = True) -> Optional[IntegrandGroup]:
        if self.Session is None:
            return None
        with self.Session() as session:
            group_model = session.query(IntegrandGroupModel).filter_by(
                integrand_hash=integrand_hash
            ).first()
            if group_model is None:
                return None
            removed_ids = set(self.get_removed_instance_ids()) if exclude_curated else set()
            return self._model_to_group(group_model, session, removed_ids)

    def get_group_by_id(self, group_id: str, exclude_curated: bool = True) -> Optional[IntegrandGroup]:
        if self.Session is None:
            return None
        with self.Session() as session:
            group_model = session.query(IntegrandGroupModel).filter_by(id=group_id).first()
            if group_model is None:
                return None
            removed_ids = set(self.get_removed_instance_ids()) if exclude_curated else set()
            return self._model_to_group(group_model, session, removed_ids)

    def get_groups_by_family(self, family: str, limit: int = 100, exclude_curated: bool = True) -> List[IntegrandGroup]:
        if self.Session is None:
            return []
        with self.Session() as session:
            group_models = session.query(IntegrandGroupModel).filter_by(
                integrand_family=family
            ).limit(limit).all()
            removed_ids = set(self.get_removed_instance_ids()) if exclude_curated else set()
            groups = [self._model_to_group(g, session, removed_ids) for g in group_models]
            return [g for g in groups if g is not None]

    def get_all_groups(self, limit: int = 1000, offset: int = 0, exclude_curated: bool = True) -> List[IntegrandGroup]:
        if self.Session is None:
            return []
        with self.Session() as session:
            group_models = session.query(IntegrandGroupModel).limit(limit).offset(offset).all()
            removed_ids = set(self.get_removed_instance_ids()) if exclude_curated else set()
            groups = [self._model_to_group(g, session, removed_ids) for g in group_models]
            return [g for g in groups if g is not None]

    def get_integral_instance(self, instance_id: str, exclude_curated: bool = True) -> Optional[IntegralFormula]:
        if self.Session is None:
            return None
        if exclude_curated and instance_id in self.get_removed_instance_ids():
            return None
        with self.Session() as session:
            instance_model = session.query(IntegralInstanceModel).filter_by(id=instance_id).first()
            if instance_model is None:
                return None
            return self._model_to_instance(instance_model, session)

    def count_groups(self, exclude_curated: bool = False) -> int:
        if self.Session is None:
            return 0
        with self.Session() as session:
            query = session.query(func.count(IntegrandGroupModel.id))
            if exclude_curated:
                removed_instance_ids = session.query(CurationLogModel.target_id).filter(
                    CurationLogModel.action == 'remove_instance'
                ).subquery()
                query = query.filter(~IntegrandGroupModel.instances.any(
                    IntegralInstanceModel.id.in_(removed_instance_ids)
                ))
            return query.scalar()

    def count_instances(self, exclude_curated: bool = False) -> int:
        if self.Session is None:
            return 0
        with self.Session() as session:
            query = session.query(func.count(IntegralInstanceModel.id))
            if exclude_curated:
                removed_ids = session.query(CurationLogModel.target_id).filter(
                    CurationLogModel.action == 'remove_instance'
                ).all()
                removed_ids = [r[0] for r in removed_ids]
                if removed_ids:
                    query = query.filter(~IntegralInstanceModel.id.in_(removed_ids))
            return query.scalar()

    def sample_groups(self, n: int, seed: int = 42, exclude_curated: bool = True) -> List[IntegrandGroup]:
        if self.Session is None:
            return []
        random.seed(seed)
        with self.Session() as session:
            all_group_models = session.query(IntegrandGroupModel).all()
            removed_ids = set(self.get_removed_instance_ids()) if exclude_curated else set()
            all_groups = [self._model_to_group(g, session, removed_ids) for g in all_group_models]
            all_groups = [g for g in all_groups if g is not None]
            n_sample = min(n, len(all_groups))
            return random.sample(all_groups, n_sample)

    def mark_instance_removed(self, instance_id: str, reason: str = "", context: Dict[str, Any] = None):
        if self.Session is None:
            return
        with self.Session() as session:
            curation_entry = CurationLogModel(
                action='remove_instance',
                target_type='instance',
                target_id=instance_id,
                reason=reason,
                context=json.dumps(context) if context else None
            )
            session.add(curation_entry)
            session.commit()
            logger.info(f"marked instance {instance_id} as removed: {reason}")

    def get_curation_log(self) -> List[Dict[str, Any]]:
        if self.Session is None:
            return []
        with self.Session() as session:
            log_entries = session.query(CurationLogModel).order_by(
                CurationLogModel.created_at.desc()
            ).all()
            return [
                {
                    'id': entry.id,
                    'action': entry.action,
                    'target_type': entry.target_type,
                    'target_id': entry.target_id,
                    'reason': entry.reason,
                    'context': json.loads(entry.context) if entry.context else None,
                    'created_at': entry.created_at.isoformat()
                }
                for entry in log_entries
            ]

    def get_removed_instance_ids(self) -> List[str]:
        if self.Session is None:
            return []
        with self.Session() as session:
            removed = session.query(CurationLogModel.target_id).filter(
                CurationLogModel.action == 'remove_instance'
            ).all()
            return [r[0] for r in removed]

    def rebuild_from_jsonl(self, groups_path: Path, integrals_path: Path):
        """Args:
            groups_path: path to integrand_groups.jsonl
            integrals_path: path to integrals_all.jsonl
        """
        logger.info(f"rebuilding database from {groups_path} and {integrals_path}")
        # drop and recreate tables
        self.drop_tables()
        self.create_tables()
        # load groups
        groups_loaded = 0
        with self.Session() as session:
            with open(groups_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    group_model = self._dict_to_group_model(data)
                    session.add(group_model)
                    groups_loaded += 1
                    if groups_loaded % 1000 == 0:
                        session.commit()
                        logger.info(f"loaded {groups_loaded} groups...")
            session.commit()
        logger.info(f"loaded {groups_loaded} integrand groups")
        # load integrals
        instances_loaded = 0
        with self.Session() as session:
            with open(integrals_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    instance_model, mse_model = self._dict_to_instance_models(data)
                    session.add(instance_model)
                    if mse_model:
                        session.add(mse_model)
                    instances_loaded += 1
                    if instances_loaded % 1000 == 0:
                        session.commit()
                        logger.info(f"loaded {instances_loaded} instances...")
            session.commit()
        logger.info(f"loaded {instances_loaded} integral instances")

    def export_curation_log(self) -> List[Dict[str, Any]]:
        return self.get_curation_log()

    def import_curation_log(self, records: List[Dict[str, Any]]):
        if self.Session is None:
            return
        with self.Session() as session:
            for record in records:
                curation_entry = CurationLogModel(
                    action=record['action'],
                    target_type=record['target_type'],
                    target_id=record['target_id'],
                    reason=record.get('reason'),
                    context=json.dumps(record['context']) if record.get('context') else None
                )
                session.add(curation_entry)
            session.commit()
            logger.info(f"imported {len(records)} curation log entries")

    def _model_to_group(self, group_model: IntegrandGroupModel, session: Session, removed_instance_ids: set) -> Optional[IntegrandGroup]:
        """Convert SQLAlchemy model to IntegrandGroup dataclass
        Args:
            group_model: SQLAlchemy group model
            session: database session
            removed_instance_ids: set of instance IDs that are marked as removed
        Returns:
            IntegrandGroup or None if group becomes orphaned (all instances removed)
        """
        definite_instances = [
            inst.id for inst in group_model.instances
            if inst.integral_type == 'definite' and inst.id not in removed_instance_ids
        ]
        indefinite_instances = [
            inst.id for inst in group_model.instances
            if inst.integral_type == 'indefinite' and inst.id not in removed_instance_ids
        ]

        if not definite_instances and not indefinite_instances:
            return None
        unique_mse_questions = set()
        latex_variants = []
        for inst in group_model.instances:
            if inst.id in removed_instance_ids:
                continue
            if inst.mse_metadata and inst.mse_metadata.mse_question_id:
                unique_mse_questions.add(inst.mse_metadata.mse_question_id)
            if inst.normalized_latex:
                latex_variants.append(inst.normalized_latex)
        latex_variants = list(set(latex_variants))[:10]
        return IntegrandGroup(
            id=group_model.id,
            integrand_canonical=group_model.integrand_canonical,
            integrand_hash=group_model.integrand_hash,
            indefinite_instances=indefinite_instances,
            definite_instances=definite_instances,
            unique_mse_questions=unique_mse_questions,
            latex_variants=latex_variants,
            integrand_family=group_model.integrand_family,
            checksum=group_model.checksum,
            created_at=group_model.created_at.isoformat() if group_model.created_at else None
        )

    def _model_to_instance(self, instance_model: IntegralInstanceModel, session: Session) -> IntegralFormula:
        mse_question_id = None
        mse_answer_id = None
        source_url = ""
        author_name = None
        author_link = None

        if instance_model.mse_metadata:
            mse_question_id = instance_model.mse_metadata.mse_question_id
            mse_answer_id = instance_model.mse_metadata.mse_answer_id
            source_url = instance_model.mse_metadata.source_url or ""
            author_name = instance_model.mse_metadata.author_name
            author_link = instance_model.mse_metadata.author_link

        equivalent_forms = []
        if instance_model.equivalent_forms:
            try:
                equivalent_forms = json.loads(instance_model.equivalent_forms)
            except (json.JSONDecodeError, TypeError):
                equivalent_forms = []

        return IntegralFormula(
            id=instance_model.id,
            raw_latex=instance_model.raw_latex,
            source_id=instance_model.source_id,
            normalized_source_id=instance_model.normalized_source_id,
            chain_position=instance_model.chain_position,
            mse_question_id=mse_question_id or 0,
            mse_answer_id=mse_answer_id,
            source_url=source_url,
            normalized_latex=instance_model.normalized_latex,
            sympy_integrand=instance_model.sympy_integrand,
            sympy_variable=instance_model.sympy_variable,
            sympy_lower_bound=instance_model.sympy_lower_bound,
            sympy_upper_bound=instance_model.sympy_upper_bound,
            integral_type=instance_model.integral_type,
            parsing_success=instance_model.parsing_success,
            parsing_error=instance_model.parsing_error,
            integrand_canonical=instance_model.integrand_canonical,
            integrand_hash=instance_model.integrand_hash,
            integrand_family=instance_model.integrand_family,
            equivalent_forms=equivalent_forms,
            author_name=author_name,
            author_link=author_link,
            checksum=instance_model.checksum,
            normalized_content_checksum=instance_model.normalized_content_checksum,
            created_at=instance_model.created_at.isoformat() if instance_model.created_at else None
        )

    def _dict_to_group_model(self, data: Dict[str, Any]) -> IntegrandGroupModel:
        return IntegrandGroupModel(
            id=data['id'],
            integrand_canonical=data['integrand_canonical'],
            integrand_hash=data['integrand_hash'],
            integrand_family=data.get('integrand_family'),
            checksum=data.get('checksum'),
        )

    def _dict_to_instance_models(self, data: Dict[str, Any]) -> tuple[IntegralInstanceModel, Optional[MSEMetadataModel]]:
        equivalent_forms_json = None
        if 'equivalent_forms' in data and data['equivalent_forms']:
            equivalent_forms_json = json.dumps(data['equivalent_forms'])

        instance = IntegralInstanceModel(
            id=data['id'],
            integrand_hash=data['integrand_hash'],
            source_id=data['source_id'],
            normalized_source_id=data['normalized_source_id'],
            chain_position=data['chain_position'],
            raw_latex=data['raw_latex'],
            normalized_latex=data['normalized_latex'],
            equivalent_forms=equivalent_forms_json,
            sympy_integrand=data['sympy_integrand'],
            sympy_variable=data['sympy_variable'],
            sympy_lower_bound=data.get('sympy_lower_bound'),
            sympy_upper_bound=data.get('sympy_upper_bound'),
            integral_type=data['integral_type'],
            parsing_success=data['parsing_success'],
            parsing_error=data.get('parsing_error'),
            integrand_canonical=data.get('integrand_canonical'),
            integrand_family=data.get('integrand_family'),
            checksum=data.get('checksum'),
            normalized_content_checksum=data.get('normalized_content_checksum'),
        )
        mse_metadata = None
        if data.get('mse_question_id'):
            mse_metadata = MSEMetadataModel(
                instance_id=data['id'],
                mse_question_id=data.get('mse_question_id'),
                mse_answer_id=data.get('mse_answer_id'),
                source_url=data.get('source_url'),
                author_name=data.get('author_name'),
                author_link=data.get('author_link'),
            )
        return instance, mse_metadata
