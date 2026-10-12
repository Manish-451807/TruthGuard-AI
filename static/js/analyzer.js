/* analyzer.js - character counter and a real "working" state for analysis forms.
   The spinner only shows while the browser is genuinely waiting for the server. */
(function () {
  "use strict";

  // Live character counter for textareas that declare data-counter="<id>".
  document.querySelectorAll("textarea[data-counter]").forEach(function (area) {
    var output = document.getElementById(area.getAttribute("data-counter"));
    if (!output) { return; }
    var max = area.maxLength > 0 ? area.maxLength : null;
    function update() {
      output.textContent = area.value.length + (max ? " / " + max : "") + " characters";
    }
    area.addEventListener("input", update);
    update();
  });

  // Disable the submit button once the form is valid and being sent (stops double submits).
  document.querySelectorAll("form[data-analyze-form]").forEach(function (form) {
    var button = form.querySelector("button[type='submit']");
    if (!button) { return; }
    var label = button.querySelector(".btn-label");
    var original = label ? label.textContent : "";
    form.addEventListener("submit", function () {
      if (!form.checkValidity()) { return; }
      button.classList.add("loading");
      if (label) { label.textContent = button.getAttribute("data-loading") || "Working..."; }
      // Disable after the submit event finishes so the form data is still sent.
      setTimeout(function () { button.disabled = true; }, 0);
    });
    // If the user returns with the Back button, restore the button.
    window.addEventListener("pageshow", function () {
      button.disabled = false;
      button.classList.remove("loading");
      if (label) { label.textContent = original; }
    });
  });
})();
