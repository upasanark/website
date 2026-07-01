// Chapter navigation for Reveal.js slide pages
// Usage: <script src="../chapter-nav.js" data-course="generative-rec" data-chapter="1"></script>

(function() {
  const script = document.currentScript;
  const course = script.getAttribute('data-course');
  const chapterIdx = parseInt(script.getAttribute('data-chapter'), 10);

  // Course chapter definitions
  const courses = {
    'generative-rec': {
      title: 'Generative Retrieval for Recommendations',
      chapters: [
        { file: '01_sequential_rec_foundations.html', title: 'Ch 1: Sequential Recommendation Foundations' },
        { file: '02_generative_retrieval.html', title: 'Ch 2: Generative Retrieval' },
        { file: '03_tokenization_methods.html', title: 'Ch 3: Tokenization Methods' },
        { file: '04_tiger_and_foundations.html', title: 'Ch 4: TIGER and Foundations' },
        { file: '05_llm_training_techniques.html', title: 'Ch 5: LLM Training Techniques' },
        { file: '06_beam_search_inference.html', title: 'Ch 6: Beam Search Inference' },
        { file: '07_applications_and_applied.html', title: 'Ch 7: Applications and Applied Methods' }
      ]
    },
    'graph-methods': {
      title: 'Graph Methods for Recommendations',
      chapters: [
        { file: '01_graph_foundations.html', title: 'Ch 1: Foundations of Graph Theory for ML' },
        { file: '01b_spectral_geometry.html', title: 'Ch 1b: Spectral Geometry' },
        { file: '02_gnn_and_gcn.html', title: 'Ch 2: GNN and Graph Convolution' },
        { file: '03_graphs_for_recommendations.html', title: 'Ch 3: Graphs for Recommendations' },
        { file: '04_multi_interest_graphs.html', title: 'Ch 4: Multi-Interest Graph Methods' },
        { file: '05_graph_transformers.html', title: 'Ch 5: Graph Transformers' },
        { file: '06_scaling_graphs.html', title: 'Ch 6: Scaling Graph Methods' },
        { file: '07_knowledge_graphs.html', title: 'Ch 7: Knowledge Graphs' },
        { file: '08_applied_graph_methods.html', title: 'Ch 8: Applied Graph Methods' },
        { file: '09_interview_prep.html', title: 'Ch 9: Interview Prep' }
      ]
    }
  };

  const courseData = courses[course];
  if (!courseData) return;

  const chapters = courseData.chapters;
  const prev = chapterIdx > 0 ? chapters[chapterIdx - 1] : null;
  const next = chapterIdx < chapters.length - 1 ? chapters[chapterIdx + 1] : null;

  // Root path: number of directories deep from site root
  // Chapter files are 3 levels deep: Website/DeepLearning/CourseName/chapter.html
  // So we need "../../" to get back to site root
  const path = window.location.pathname;
  // Count directory depth from path segments (file is last segment)
  const segments = path.split('/').filter(Boolean);
  const depth = segments.length - 1; // -1 for the filename itself
  const rootPath = depth <= 0 ? './' : '../'.repeat(depth);

  // Build chapter nav bar
  const nav = document.createElement('div');
  nav.className = 'chapter-top-nav';
  nav.innerHTML = `
    <div class="chapter-nav-inner">
      <a href="index.html" class="chapter-back-link">&larr; Back to ${courseData.title}</a>
      <span class="chapter-nav-sep">|</span>
      <a href="${rootPath}index.html" class="chapter-home-link">Home</a>
      <span class="chapter-nav-spacer"></span>
      <span class="chapter-position">${chapterIdx + 1} / ${chapters.length}</span>
    </div>
  `;

  // Build prev/next footer
  const footer = document.createElement('div');
  footer.className = 'chapter-bottom-nav';
  let footerHTML = '<div class="chapter-bottom-inner">';
  if (prev) {
    footerHTML += `<a href="${prev.file}" class="chapter-prev">&larr; ${prev.title}</a>`;
  } else {
    footerHTML += `<span></span>`;
  }
  if (next) {
    footerHTML += `<a href="${next.file}" class="chapter-next">${next.title} &rarr;</a>`;
  } else {
    footerHTML += `<span></span>`;
  }
  footerHTML += '</div>';
  footer.innerHTML = footerHTML;

  // Insert at top and bottom of body
  document.body.insertBefore(nav, document.body.firstChild);
  document.body.appendChild(footer);
})();