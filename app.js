const copyCitationButton = document.getElementById("copy-citation-button");
const citationCode = document.getElementById("language-bibtex");
const scrollTopButton = document.getElementById("scroll-top-button");
const themeToggle = document.getElementById("theme-toggle");

const lightbox = document.getElementById("image-lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const captionText = document.getElementById("lightbox-caption");
const closeButton = document.getElementsByClassName("lightbox-close")[0];
const contentImages = document.querySelectorAll(".section img");

const sideNav = document.getElementById("side-nav");
const sideNavLinks = document.querySelectorAll(".side-nav a");
const sections = [];

let isDragging = false;
let startX, startY, scrollLeft, scrollTop;
let hasMoved = false;
let scrollTimeout = null;

function getStoredTheme() {
  return localStorage.getItem("theme");
}

function setStoredTheme(theme) {
  localStorage.setItem("theme", theme);
}

function getPreferredTheme() {
  const storedTheme = getStoredTheme();
  if (storedTheme) {
    return storedTheme;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  setStoredTheme(theme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme");
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  setTheme(newTheme);
}

function initializeTheme() {
  const preferredTheme = getPreferredTheme();
  setTheme(preferredTheme);

  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", (e) => {
      const storedTheme = getStoredTheme();
      if (!storedTheme) {
        setTheme(e.matches ? "dark" : "light");
      }
    });
}

function toggleScrollButton() {
  if (
    document.body.scrollTop > 300 ||
    document.documentElement.scrollTop > 300
  ) {
    scrollTopButton.style.display = "block";
  } else {
    scrollTopButton.style.display = "none";
  }
}

function toggleSideNav() {
  const scrollPosition = window.scrollY || document.documentElement.scrollTop;
  const headerHeight = document.querySelector(".title").offsetHeight + 100;

  if (scrollPosition > headerHeight) {
    sideNav.classList.add("visible");
  } else {
    sideNav.classList.remove("visible");
  }
}

function initializeSections() {
  sideNavLinks.forEach((link) => {
    const sectionId = link.getAttribute("data-section");
    const section = document.getElementById(sectionId);
    if (section) {
      sections.push({ id: sectionId, element: section, link: link });
    }
  });
}

function updateActiveSection() {
  const scrollPosition = window.scrollY + window.innerHeight / 3;

  let currentSection = null;

  for (let i = sections.length - 1; i >= 0; i--) {
    const section = sections[i];
    const sectionTop =
      section.element.getBoundingClientRect().top + window.scrollY;

    if (scrollPosition >= sectionTop) {
      currentSection = section;
      break;
    }
  }

  sideNavLinks.forEach((link) => link.classList.remove("active"));

  if (currentSection) {
    currentSection.link.classList.add("active");
  }
}

function handleScroll() {
  if (scrollTimeout) {
    cancelAnimationFrame(scrollTimeout);
  }

  scrollTimeout = requestAnimationFrame(() => {
    toggleScrollButton();
    toggleSideNav();
    updateActiveSection();
  });
}

function handleSideNavClick(event) {
  const link = event.target.closest("a");
  if (!link) return;

  event.preventDefault();
  const targetId = link.getAttribute("data-section");
  const targetSection = document.getElementById(targetId);

  if (targetSection) {
    const offset = 20;
    const targetPosition =
      targetSection.getBoundingClientRect().top + window.scrollY - offset;

    window.scrollTo({
      top: targetPosition,
      behavior: "smooth",
    });
  }
}

function fallbackCopyToClipboard(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;

  textArea.style.cssText =
    "position:fixed;top:0;left:0;width:2em;height:2em;padding:0;border:none;outline:none;box-shadow:none;background:transparent;";

  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    const successful = document.execCommand("copy");
    return successful;
  } finally {
    document.body.removeChild(textArea);
  }
}

function copyToClipboard() {
  const textToCopy = citationCode.innerText;

  if (navigator.clipboard) {
    navigator.clipboard
      .writeText(textToCopy)
      .then(() => {
        showCopyFeedback(true);
      })
      .catch(() => {
        const success = fallbackCopyToClipboard(textToCopy);
        showCopyFeedback(success);
      });
  } else {
    const success = fallbackCopyToClipboard(textToCopy);
    showCopyFeedback(success);
  }
}

function showCopyFeedback(success) {
  if (!success) return;

  const iconCopy = copyCitationButton.querySelector(".icon-copy");
  const iconCheck = copyCitationButton.querySelector(".icon-check");

  copyCitationButton.classList.add("success");

  iconCopy.style.display = "none";
  iconCheck.style.display = "block";

  setTimeout(() => {
    copyCitationButton.classList.remove("success");
    iconCopy.style.display = "block";
    iconCheck.style.display = "none";
  }, 2000);
}

function closeLightbox() {
  lightbox.style.display = "none";
  lightboxImg.classList.remove("zoomed");
}

function openLightbox(imgElement) {
  lightbox.style.display = "block";
  lightboxImg.src = imgElement.src;
  captionText.innerHTML = imgElement.alt || imgElement.title || "";
  lightboxImg.classList.remove("zoomed");
}

function handleLightboxClick(event) {
  if (hasMoved) {
    return;
  }
  if (event.target === lightbox) {
    closeLightbox();
  }
}

function handleLightboxImgClick(event) {
  event.stopPropagation();
  if (hasMoved) {
    return;
  }
  lightboxImg.classList.toggle("zoomed");
}

function handleKeydown(event) {
  if (event.key === "Escape" && lightbox.style.display === "block") {
    closeLightbox();
  }
}

function handleMouseDown(event) {
  if (!lightboxImg.classList.contains("zoomed")) {
    return;
  }

  isDragging = true;
  hasMoved = false;
  lightbox.classList.add("grabbing");

  startX = event.pageX - lightbox.offsetLeft;
  startY = event.pageY - lightbox.offsetTop;
  scrollLeft = lightbox.scrollLeft;
  scrollTop = lightbox.scrollTop;

  event.preventDefault();
}

function handleMouseMove(event) {
  if (!isDragging) {
    return;
  }
  event.preventDefault();

  const x = event.pageX - lightbox.offsetLeft;
  const y = event.pageY - lightbox.offsetTop;

  const walkX = x - startX;
  const walkY = y - startY;

  if (Math.abs(walkX) > 5 || Math.abs(walkY) > 5) {
    hasMoved = true;
  }

  lightbox.scrollLeft = scrollLeft - walkX;
  lightbox.scrollTop = scrollTop - walkY;
}

function stopDragging() {
  isDragging = false;
  lightbox.classList.remove("grabbing");
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth",
  });
}

function formatBibtexSyntax() {
  citationCode.innerHTML = citationCode.textContent
    // Entry type like @InProceedings
    .replace(/(@\w+)/g, '<span class="bibtex-type">$1</span>')
    // Citation key (first thing after {)
    .replace(/(@\w+<\/span>)\{(\w+)/, '$1{<span class="bibtex-key">$2</span>')
    // Field names (preserve spaces before =)
    .replace(/^(\s+)(\w+)(\s*=)/gm, '$1<span class="bibtex-field">$2</span>$3')
    // Values in braces
    .replace(/= \{([^}]+)\}/g, '= {<span class="bibtex-value">$1</span>}');
}

initializeTheme();
formatBibtexSyntax();
initializeSections();

window.addEventListener("scroll", handleScroll, { passive: true });

window.onscroll = toggleScrollButton;

themeToggle.addEventListener("click", toggleTheme);
copyCitationButton.addEventListener("click", copyToClipboard);
scrollTopButton.addEventListener("click", scrollToTop);
closeButton.addEventListener("click", closeLightbox);
sideNav.addEventListener("click", handleSideNavClick);

contentImages.forEach((img) => {
  img.addEventListener("click", function () {
    openLightbox(this);
  });
});

lightbox.addEventListener("click", handleLightboxClick);
lightboxImg.addEventListener("click", handleLightboxImgClick);
lightbox.addEventListener("mousedown", handleMouseDown);
lightbox.addEventListener("mousemove", handleMouseMove);
lightbox.addEventListener("mouseup", stopDragging);
lightbox.addEventListener("mouseleave", stopDragging);

document.addEventListener("keydown", handleKeydown);