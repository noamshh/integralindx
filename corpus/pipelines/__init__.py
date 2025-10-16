"""
Pipeline modules for IntegralIndx corpus processing.

Contains all data processing pipelines:
- scrape_pipeline: MSE data scraping
- normalize_pipeline: LaTeX normalization
- filter_pipeline: Integral extraction and filtering
- refilter_contributors_pipeline: Reprocess contributors with updated normalization
- variable_normalization_pipeline: Variable and parameter normalization
- smart_variable_normalization: Comparison-based efficient normalization
- integrand_group_pipeline: Grouping and deduplication
"""
