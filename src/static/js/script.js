/**
 * Frontend logic for the SoluGen AI Recruitment RAG system.
 * * This module handles the asynchronous communication with the FastAPI 
 * backend, manages the UI state transitions, and renders the 
 * retrieved job descriptions.
 * * @author Your Name
 * @version 1.0.0
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Element References
    const searchBtn = document.getElementById('searchBtn');
    const queryInput = document.getElementById('queryInput');
    const resultsContainer = document.getElementById('resultsContainer');
    const statusMessage = document.getElementById('statusMessage');

    /**
     * Executes the semantic search by sending the user query to the backend.
     * Manages UI loading states and handles potential network errors.
     * * @async
     * @function performSearch
     * @returns {Promise<void>}
     */
    const performSearch = async () => {
        const query = queryInput.value.trim();
        
        // Prevent empty searches
        if (!query) return;

        // --- UI State: Loading ---
        resultsContainer.innerHTML = '';
        statusMessage.className = 'status-visible';
        searchBtn.disabled = true;

        try {
            /** @type {Response} */
            const response = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query, 
                    k: 5 // Defaulting to top 5 most relevant matches
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            renderResults(data.results);
            
        } catch (error) {
            console.error('Retrieval Error:', error);
            resultsContainer.innerHTML = `
                <div class="error-message">
                    <p>⚠️ Failed to retrieve results. Please ensure the backend is running.</p>
                </div>
            `;
        } finally {
            // --- UI State: Idle ---
            statusMessage.className = 'status-hidden';
            searchBtn.disabled = false;
        }
    };

    /**
 * Processes and renders search results into the DOM.
 * * @function renderResults
 * @param {Array<Object>} results - Array of result objects from the API.
 * @param {string} results[].job_title - Title of the job.
 * @param {number} results[].score - Normalized similarity score (0-1).
 * @param {string} results[].content - The text chunk content.
 * @param {string|number} results[].id - The database index/chunk ID.
 * @param {string} results[].source - The source filename.
 */
const renderResults = (results) => {
    // --- Edge Case: No Results Above Threshold ---
    if (!results || results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="no-results-state">
                <p>No matches found that meet the relevance threshold. Try rephrasing your search.</p>
            </div>`;
        return;
    }

    results.forEach(res => {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        // --- Component Construction ---
        // Header: Title and Similarity Score
        // Content: Main text
        // Footer: Chunk ID and Source identifier
        card.innerHTML = `
            <div class="result-header">
                <span class="job-title">${res.job_title}</span>
                <span class="score-pill">Match: ${(res.score * 100).toFixed(1)}%</span>
            </div>
            <div class="content">${res.content.replace(/\n/g, '<br>')}</div>
            <div class="result-footer">
                <span class="id-tag">Document Index: ${res.id}</span>
                <span class="source-tag">Source: ${res.source}</span>
            </div>
        `;
        resultsContainer.appendChild(card);
    });
};

    // --- Event Listeners ---

    /** Trigger search on button click */
    searchBtn.addEventListener('click', performSearch);

    /** Trigger search on 'Enter' key press in the input field */
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
});