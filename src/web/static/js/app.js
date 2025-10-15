
class IntegralSearchApp {
    constructor() {
        this.currentQuery = null;
        this.currentK = 6;
        this.initializeEventListeners();
        this.restoreSearchState();
    }

    initializeEventListeners() {
        document.getElementById('searchForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.performSearch(true); // true = new search, reset to 6 results
        });
        document.getElementById('logoLink').addEventListener('click', (e) => {
            e.preventDefault();
            this.clearSearch();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        document.getElementById('moreResultsBtn').addEventListener('click', () => {
            this.loadMoreResults();
        });
        this.checkServerHealth();
    }

    async checkServerHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            if (!health.search_engine_available) {
                this.showError('Search engine is not available. Please ensure the embeddings database is set up.');
            } else if (!health.inference_available) {
                console.warn('Inference engine not available - new queries will use fallback method');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.showError('Unable to connect to server');
        }
    }

    async validateQuery(query) {
        try {
            const response = await fetch('/api/validate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ expression: query })
            });
            return await response.json();
        } catch (error) {
            console.error('Validation error:', error);
            return { valid: false, error: 'Validation request failed' };
        }
    }

    async performSearch(isNewSearch = true) {
        const queryInput = document.getElementById('queryInput');
        const query = queryInput.value.trim();
        if (!query) {
            this.showError('Please enter an integral expression');
            return;
        }
        const validation = await this.validateQuery(query);
        if (!validation.valid) {
            queryInput.classList.add('is-invalid');
            this.showError(validation.error || 'Invalid SymPy expression');
            return;
        }
        queryInput.classList.remove('is-invalid');
        if (isNewSearch) {
            this.currentK = 6;
        }
        this.currentQuery = query;
        this.setLoadingState(true);
        this.hideError();
        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, k: this.currentK })
            });

            const result = await response.json();

            if (result.success) {
                // Save search state to sessionStorage
                this.saveSearchState({
                    query: query,
                    k: this.currentK,
                    results: result.results
                });

                this.displayResults(result.query, result.results);
            } else {
                this.showError(result.message || 'Search failed');
            }
        } catch (error) {
            console.error('Search error:', error);
            this.showError('Network error occurred during search');
        } finally {
            this.setLoadingState(false);
        }
    }

    async loadMoreResults() {
        this.currentK = 18;
        await this.performSearch(false); // false = not a new search
    }

    displayResults(query, results) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContainer = document.getElementById('resultsContainer');
        const resultsCount = document.getElementById('resultsCount');
        const moreResultsContainer = document.getElementById('moreResultsContainer');

        if (results.length === 0) {
            this.showError('No similar integrals found');
            return;
        }
        resultsCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
        resultsContainer.innerHTML = '';
        results.forEach((result, index) => {
            const resultCard = this.createResultCard(result, index + 1);
            resultsContainer.appendChild(resultCard);
        });

        if (this.currentK === 6 && results.length === 6) {
            moreResultsContainer.classList.remove('d-none');
        } else {
            moreResultsContainer.classList.add('d-none');
        }
        resultsSection.classList.remove('d-none');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
            MathJax.typesetPromise([resultsContainer]).catch(() => {
            });
        } else if (window.MathJax) {
            window.MathJax.startup?.promise?.then(() => {
                if (typeof MathJax.typesetPromise === 'function') {
                    MathJax.typesetPromise([resultsContainer]).catch(() => {});
                }
            });
        }
    }

    createResultCard(result, rank) {
        const col = document.createElement('div');
        col.className = 'col-md-6 col-lg-4 mb-3';
        const similarityClass = this.getSimilarityClass(result.similarity_score);
        const similarityPercent = (result.similarity_score * 100).toFixed(1);
        const displayMath = result.integrand_latex || result.integrand_canonical || result.latex || 'N/A';
        col.innerHTML = `
            <div class="result-card">
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <span class="badge bg-primary">#${rank}</span>
                    <span class="similarity-score ${similarityClass}">
                        ${similarityPercent}% similar
                    </span>
                </div>
                <div class="result-math">\\[${displayMath}\\]</div>
                <div class="mt-2">
                    <small class="text-muted">
                        <strong>Family:</strong> ${result.integrand_family || 'Unknown'}
                        <br>
                        <strong>Instances:</strong> ${result.total_instances || 0}
                        (${result.definite_count || 0} definite, ${result.indefinite_count || 0} indefinite)
                        <br>
                        <strong>MSE Questions:</strong> ${result.unique_mse_questions || 0}
                    </small>
                </div>
                ${result.latex_variants && result.latex_variants.length > 0 ? `
                    <div class="mt-2">
                        <small class="text-muted">
                            <strong>Example integrals:</strong>
                            <div class="mt-1">
                                ${result.latex_variants.slice(0, 2).map(latex =>
                                    `<div class="equivalent-form">\\(${latex}\\)</div>`
                                ).join('')}
                                ${result.total_instances > 2 ?
                                    `<div class="text-muted">... +${result.total_instances - 2} more</div>`
                                    : ''}
                            </div>
                        </small>
                    </div>
                ` : ''}
            </div>
        `;

        col.addEventListener('click', (e) => {
            if (e.target.tagName === 'A' || e.target.closest('a')) {
                return;
            }
            window.location.href = `/integrand/${result.integrand_hash}`;
        });

        return col;
    }

    getSimilarityClass(score) {
        if (score > 0.85) return 'high-similarity';
        if (score > 0.6) return 'medium-similarity';
        return 'low-similarity';
    }

    setLoadingState(loading) {
        try {
            const spinner = document.getElementById('searchSpinner');
            const button = document.querySelector('#searchForm button[type="submit"]');
            if (!button) {
                console.error('Search button not found');
                return;
            }
            if (loading) {
                if (spinner) spinner.classList.remove('d-none');
                button.disabled = true;
                button.innerHTML = '<span id="searchSpinner" class="spinner-border spinner-border-sm me-2"></span>Searching...';
            } else {
                if (spinner) spinner.classList.add('d-none');
                button.disabled = false;
                button.innerHTML = '<span id="searchSpinner" class="spinner-border spinner-border-sm d-none me-2"></span>Find Similar Integrals';
            }
        } catch (error) {
            console.error('Error in setLoadingState:', error);
        }
    }

    showError(message) {
        const errorAlert = document.getElementById('errorAlert');
        const errorMessage = document.getElementById('errorMessage');
        const displayMessage = window.DEV_MODE ? message : 'Invalid expression';
        errorMessage.textContent = displayMessage;
        errorAlert.classList.remove('d-none');
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        const errorAlert = document.getElementById('errorAlert');
        errorAlert.classList.add('d-none');
    }

    clearSearch() {
        const queryInput = document.getElementById('queryInput');
        queryInput.value = '';
        queryInput.classList.remove('is-invalid');
        this.currentK = 6;
        this.currentQuery = null;
        document.getElementById('resultsSection').classList.add('d-none');
        document.getElementById('moreResultsContainer').classList.add('d-none');
        document.getElementById('resultsContainer').innerHTML = '';
        sessionStorage.removeItem('lastSearch');
        this.hideError();
    }

    saveSearchState(searchData) {
        try {
            sessionStorage.setItem('lastSearch', JSON.stringify(searchData));
        } catch (e) {
            console.error('Failed to save search state:', e);
        }
    }

    restoreSearchState() {
        try {
            const saved = sessionStorage.getItem('lastSearch');
            if (!saved) return;
            const searchData = JSON.parse(saved);
            document.getElementById('queryInput').value = searchData.query || '';
            this.currentK = searchData.k || 6;
            this.currentQuery = searchData.query || null;
            if (searchData.results && searchData.results.length > 0) {
                this.displayResults(searchData.query, searchData.results);
            }
        } catch (e) {
            console.error('Failed to restore search state:', e);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new IntegralSearchApp();
});