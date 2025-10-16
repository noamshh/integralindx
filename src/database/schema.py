"""Tables:
- integrand_groups: unique integrand groups
- integral_instances: individual integral records
- mse_metadata: math stackexchange metadata
- curation_log: manual curation actions"""

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class IntegrandGroupModel(Base):
    """Unique integrand groups (deduplicated by integrand_hash)
    Maps to corpus.formula_models.IntegrandGroup"""
    __tablename__ = 'integrand_groups'
    id = Column(String, primary_key=True)  # integrand-group-{hash}
    integrand_canonical = Column(Text, nullable=False)
    integrand_hash = Column(String, nullable=False, unique=True, index=True)
    integrand_family = Column(String, nullable=True, index=True)
    checksum = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    instances = relationship('IntegralInstanceModel', back_populates='group', cascade='all, delete-orphan')
    __table_args__ = (
        Index('idx_group_integrand_hash', 'integrand_hash'),
        Index('idx_group_integrand_family', 'integrand_family'),
    )


class IntegralInstanceModel(Base):
    """Individual integral instances
    Maps to corpus.formula_models.IntegralFormula"""
    __tablename__ = 'integral_instances'
    id = Column(String, primary_key=True)  # integral-{hash}
    integrand_hash = Column(String, ForeignKey('integrand_groups.integrand_hash'), nullable=False, index=True)
    source_id = Column(String, nullable=False)  # original RawFormula ID
    normalized_source_id = Column(String, nullable=False)  # NormalizedFormula ID
    chain_position = Column(Integer, nullable=False, default=0)
    raw_latex = Column(Text, nullable=False)
    normalized_latex = Column(Text, nullable=False)
    equivalent_forms = Column(Text, nullable=True)  # json array of equivalent forms
    sympy_integrand = Column(Text, nullable=False)
    sympy_variable = Column(String, nullable=False)
    sympy_lower_bound = Column(String, nullable=True)
    sympy_upper_bound = Column(String, nullable=True)
    integral_type = Column(String, nullable=False)  # 'definite' or 'indefinite'
    parsing_success = Column(Boolean, nullable=False, default=True)
    parsing_error = Column(Text, nullable=True)
    integrand_canonical = Column(Text, nullable=True)
    integrand_family = Column(String, nullable=True)
    checksum = Column(String, nullable=True)
    normalized_content_checksum = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    group = relationship('IntegrandGroupModel', back_populates='instances')
    mse_metadata = relationship('MSEMetadataModel', back_populates='instance', uselist=False, cascade='all, delete-orphan')
    __table_args__ = (
        Index('idx_instance_integral_type', 'integral_type'),
        Index('idx_instance_integrand_hash', 'integrand_hash'),
        Index('idx_instance_source_id', 'source_id'),
    )


class MSEMetadataModel(Base):
    __tablename__ = 'mse_metadata'
    instance_id = Column(String, ForeignKey('integral_instances.id'), primary_key=True)
    mse_question_id = Column(Integer, nullable=True, index=True)
    mse_answer_id = Column(Integer, nullable=True)
    source_url = Column(Text, nullable=True)
    author_name = Column(String, nullable=True)
    author_link = Column(Text, nullable=True)
    instance = relationship('IntegralInstanceModel', back_populates='mse_metadata')
    __table_args__ = (
        Index('idx_mse_metadata_question_id', 'mse_question_id'),
    )


class CurationLogModel(Base):
    """Manual curation actions (removals, edits)
    Preserved across database rebuilds"""
    __tablename__ = 'curation_log'
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String, nullable=False)  # 'remove_instance', 'edit_group', etc.
    target_type = Column(String, nullable=False)  # 'instance', 'group'
    target_id = Column(String, nullable=False, index=True)
    reason = Column(Text, nullable=True)
    context = Column(Text, nullable=True)  # json string for additional data
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    __table_args__ = (
        Index('idx_curation_target_id', 'target_id'),
        Index('idx_curation_action', 'action'),
    )
