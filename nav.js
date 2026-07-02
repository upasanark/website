// Shared navigation bar for all pages
// Include with: <script src="../nav.js"></script> etc.

(function() {
  const scriptUrl = new URL(document.currentScript.src);
  const pageUrl = new URL(window.location.href);

  const scriptParts = scriptUrl.pathname.split('/').filter(Boolean); // e.g. ['website', 'nav.js']
  const pageParts = pageUrl.pathname.split('/').filter(Boolean);     // e.g. ['website', 'Writings', 'Mind-Body Problem', 'episode1.html']

  // Compare directory paths (exclude filenames)
  const scriptBase = scriptParts.slice(0, -1);
  const pageBase = pageParts.slice(0, -1);

  // Count matching path segments from root
  let matchDepth = 0;
  for (let i = 0; i < scriptBase.length && i < pageBase.length; i++) {
    if (scriptBase[i] === pageBase[i]) matchDepth++;
    else break;
  }

  // How many levels up from page to common root
  const up = pageBase.length - matchDepth;
  const rootPath = up <= 0 ? '.' : '../'.repeat(up);

  // Determine active section from the first directory segment after site root
  const activeSection = pageBase[matchDepth] || '';

  // Build nav HTML
  const nav = document.createElement('nav');
  nav.className = 'site-nav';
  nav.innerHTML = `
    <a href="${rootPath}/index.html" class="site-title">Upasana Ramakrishnan</a>
    <ul>
      <li><a href="${rootPath}/DeepLearning/" class="${activeSection === 'DeepLearning' ? 'active' : ''}">Learnings</a></li>
      <li><a href="${rootPath}/Projects/" class="${activeSection === 'Projects' ? 'active' : ''}">Projects</a></li>
      <li><a href="${rootPath}/BookReviews/" class="${activeSection === 'BookReviews' ? 'active' : ''}">Readings</a></li>
      <li><a href="${rootPath}/Writings/" class="${activeSection === 'Writings' ? 'active' : ''}">Writings</a></li>
    </ul>
  `;

  // Insert at the top of body
  document.body.insertBefore(nav, document.body.firstChild);
})();