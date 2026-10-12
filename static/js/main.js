/* main.js - small site-wide behaviours: mobile menu, flash messages, confirmations. */
(function () {
  "use strict";

  // Responsive navigation toggle (keeps aria-expanded in sync for screen readers).
  var toggle = document.getElementById("nav-toggle");
  var nav = document.getElementById("site-nav");
  if (toggle && nav) {
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  // Success messages disappear by themselves after a few seconds.
  document.querySelectorAll(".alert[data-dismiss]").forEach(function (el) {
    setTimeout(function () { el.remove(); }, 6000);
  });

  // Forms with data-confirm ask "are you sure?" before deleting anything.
  document.querySelectorAll("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      if (!window.confirm(form.getAttribute("data-confirm"))) {
        event.preventDefault();
      }
    });
  });
})();
