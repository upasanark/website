// Shared navigation bar for all pages
// Include with: <script src="/nav.js"></script> or <script src="../nav.js"></script> etc.

(function() {
  // Determine the site root path based on current page depth
  const path = window.location.pathname;
  const parts = path.replace('/Website/', '').split('/').filter(Boolean);
  
  // Calculate relative path to root
  const depth = parts.length - 1; // -1 because the filename itself is a part
  const rootPath = depth <= 0 ? '.' : '../'.repeat(depth).replace(/\/$/, '');
  
  // Determine active section
  const activeSection = parts[0] || 'home';
  const sectionMap = {
    'DeepLearning': 'Research',
    'Projects': 'Projects',
    'BookReviews': 'Book Reviews',
    'Writings': 'Writings'
  };
  
  // Build nav HTML
  const nav = document.createElement('nav');
  nav.className = 'site-nav';
  nav.innerHTML = `
    <a href="${rootPath}/index.html" class="site-title">Upasana Ramakrishnan</a>
    <ul>
      <li><a href="${rootPath}/DeepLearning/" class="${activeSection === 'DeepLearning' ? 'active' : ''}">Research</a></li>
      <li><a href="${rootPath}/Projects/" class="${activeSection === 'Projects' ? 'active' : ''}">Projects</a></li>
      <li><a href="${rootPath}/BookReviews/" class="${activeSection === 'BookReviews' ? 'active' : ''}">Book Reviews</a></li>
      <li><a href="${rootPath}/Writings/" class="${activeSection === 'Writings' ? 'active' : ''}">Writings</a></li>
    </ul>
  `;
  
  // Insert at the top of body
  document.body.insertBefore(nav, document.body.firstChild);
})();