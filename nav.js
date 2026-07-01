// Shared navigation bar for all pages
// Include with: <script src="../nav.js"></script> etc.

(function() {
  // Get the current page path and determine depth from root
  const path = window.location.pathname;
  
  // Remove leading/trailing slashes, split into segments
  const segments = path.split('/').filter(Boolean);
  
  // The last segment is the filename (e.g., "episode1.html" or "index.html")
  // Everything before it is directory depth
  const depth = segments.length - 1;
  
  // Build relative path to root
  const rootPath = depth <= 0 ? '.' : '../'.repeat(depth).replace(/\/$/, '');
  
  // Determine active section from the first directory segment
  const firstDir = segments[0] || '';
  const activeSection = firstDir;
  
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