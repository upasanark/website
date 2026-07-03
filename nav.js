// Shared navigation bar for all pages
// Include with: <script src="nav.js"></script> or <script src="../nav.js"></script> etc.

(function() {
  const scriptEl = document.currentScript;
  // Determine how deep we are by counting '../' in the script src attribute
  const src = scriptEl.getAttribute('src') || 'nav.js';
  const depth = (src.match(/\.\.\//g) || []).length;
  const root = depth === 0 ? '' : '../'.repeat(depth);

  // Determine active section from URL path
  const path = window.location.pathname;
  let activeSection = '';
  if (path.includes('/DeepLearning')) activeSection = 'DeepLearning';
  else if (path.includes('/Career')) activeSection = 'Career';
  else if (path.includes('/Projects')) activeSection = 'Career';
  else if (path.includes('/BookReviews')) activeSection = 'BookReviews';
  else if (path.includes('/Writings')) activeSection = 'Writings';

  const nav = document.createElement('nav');
  nav.className = 'site-nav';
  nav.innerHTML = `
    <a href="${root}index.html" class="site-title">Upasana Ramakrishnan</a>
    <ul>
      <li><a href="${root}DeepLearning/" class="${activeSection === 'DeepLearning' ? 'active' : ''}">Learnings</a></li>
      <li><a href="${root}Career/" class="${activeSection === 'Career' ? 'active' : ''}">Career</a></li>
      <li><a href="${root}Writings/" class="${activeSection === 'Writings' ? 'active' : ''}">Writings</a></li>
      <li><a href="${root}BookReviews/" class="${activeSection === 'BookReviews' ? 'active' : ''}">Readings</a></li>
    </ul>
  `;

  document.body.insertBefore(nav, document.body.firstChild);
})();
