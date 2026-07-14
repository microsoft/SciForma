$(document).ready(function() {
  // Initialize image carousel
  var options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 3500,
  };
  var carousels = bulmaCarousel.attach('#results-carousel', options);
  var carousels2 = bulmaCarousel.attach('#comparison-carousel', options);

  // Scroll to top
  window.addEventListener('scroll', function() {
    var btn = document.querySelector('.scroll-to-top');
    if (btn) btn.classList.toggle('visible', window.scrollY > 300);
  });
});

function scrollToTop() { window.scrollTo({top:0,behavior:'smooth'}); }
function toggleMoreWorks() {
  var dd = document.getElementById('moreWorksDropdown');
  var btn = document.querySelector('.more-works-btn');
  dd.classList.toggle('show');
  btn.classList.toggle('active');
}
function copyBibTeX() {
  var code = document.getElementById('bibtex-code').innerText;
  navigator.clipboard.writeText(code);
  var btn = document.querySelector('.copy-bibtex-btn');
  btn.classList.add('copied');
  btn.querySelector('.copy-text').textContent = 'Copied!';
  setTimeout(function(){ btn.classList.remove('copied'); btn.querySelector('.copy-text').textContent = 'Copy'; }, 2000);
}
